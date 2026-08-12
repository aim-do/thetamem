"""Configurable token-mixer shell around :class:`ThetaMemory`.

The default frontend is the one every recorded run used::

    q, k = Linear(h) -> Conv -> SiLU -> RoPE -> L2
    v    = Linear(h) -> Conv -> SiLU

The operation tuples make ablations explicit and ordered.  The
``direct_hadamard`` projection instead maps the token directly to two branches
per head and multiplies them before the operation pipeline; with no explicit
``lift`` it then uses ``key()`` as the memory lift, so no intermediate
key-to-branch matrix is introduced.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .lift import Lift, key
from .memory import ThetaMemory

__all__ = ["ThetaMemLayer", "CausalDepthwiseConv1d", "HeadRMSNorm"]

_QK_PROJECTIONS = ("linear", "direct_hadamard")
_QK_OPS = {"conv", "silu", "rope", "center", "l1", "l2", "rms"}
_VALUE_OPS = {"conv", "silu", "center", "l1", "l2", "rms"}


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise causal convolution over the time axis, without bias."""

    def __init__(self, channels: int, width: int) -> None:
        super().__init__()
        self.width = width
        self.conv = nn.Conv1d(
            channels, channels, width, groups=channels, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x.transpose(1, 2), (self.width - 1, 0))
        return self.conv(padded).transpose(1, 2)


class HeadRMSNorm(nn.Module):
    """Per-head RMS normalization with learned ``[heads, width]`` weight."""

    def __init__(self, heads: int, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(heads, width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(
            x.square().mean(-1, keepdim=True) + self.eps
        )
        heads, width = self.weight.shape
        return normalized * self.weight.to(x.dtype).view(1, heads, 1, width)


def _rope_phases(
    time: int, rotated_dim: int, base: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(time, device=device, dtype=torch.float32)
    frequencies = base ** (
        -torch.arange(0, rotated_dim, 2, device=device, dtype=torch.float32)
        / rotated_dim
    )
    phase = torch.outer(positions, frequencies)
    return phase.cos(), phase.sin()


def _apply_rope(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotated_dim: int,
) -> torch.Tensor:
    rotated = tensor[..., :rotated_dim]
    rest = tensor[..., rotated_dim:]
    even = rotated[..., 0::2]
    odd = rotated[..., 1::2]
    cos = cos.to(tensor.dtype)
    sin = sin.to(tensor.dtype)
    mixed_even = even * cos - odd * sin
    mixed_odd = even * sin + odd * cos
    mixed = torch.stack((mixed_even, mixed_odd), dim=-1).flatten(-2)
    return torch.cat((mixed, rest), dim=-1)


def _validated_ops(
    name: str, operations: Sequence[str], allowed: set[str]
) -> tuple[str, ...]:
    result = tuple(operations)
    unknown = [op for op in result if op not in allowed]
    if unknown:
        raise ValueError(
            f"unknown {name} operation(s) {unknown}; expected {sorted(allowed)}"
        )
    if result.count("conv") > 1:
        raise ValueError(f"{name} operations may contain at most one conv")
    return result


class ThetaMemLayer(nn.Module):
    """Token mixer: ``[B, T, d_model] -> [B, T, d_model]``.

    ``qk_ops`` and ``value_ops`` execute from left to right.  Their defaults
    preserve the original frontend.  Removing all operations gives a direct
    projected stream; including ``"conv"`` creates and accounts for the
    corresponding causal convolution cache.

    ``qk_projection="linear"`` produces one ``key_dim`` vector per head.
    ``"direct_hadamard"`` produces two such vectors directly from the token
    and multiplies them before ``qk_ops``.  Unless a custom ``lift`` is
    supplied explicitly, the direct projection is fed raw to the SSM through
    ``key()``; the linear projection retains the canonical Hadamard lift.

    ``position="none"`` is a master switch that disables every ``"rope"``
    operation.  Otherwise RoPE is applied exactly where ``"rope"`` appears in
    ``qk_ops``.
    """

    def __init__(
        self,
        d_model: int = 128,
        *,
        heads: int = 4,
        key_dim: int = 32,
        value_dim: int = 64,
        state: str | None = None,
        lift: Lift | None = None,
        feature_width: int | None = None,
        update: str = "sum",
        passes: int = 2,
        value_center: str = "none",
        backend: str = "chunked",
        chunk: int = 256,
        feature_norm: str = "none",
        qk_projection: str = "linear",
        qk_ops: Sequence[str] = ("conv", "silu", "rope", "l2"),
        value_ops: Sequence[str] = ("conv", "silu"),
        position: str = "rope",
        rope_fraction: float = 0.5,
        rope_base: float = 10_000.0,
        conv_width: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if qk_projection not in _QK_PROJECTIONS:
            raise ValueError(
                f"unknown qk_projection {qk_projection!r}; expected "
                f"{_QK_PROJECTIONS}"
            )
        if position not in {"rope", "none"}:
            raise ValueError(f"unknown position {position!r}")
        self.qk_ops = _validated_ops("qk", qk_ops, _QK_OPS)
        self.value_ops = _validated_ops("value", value_ops, _VALUE_OPS)
        self.d_model = d_model
        self.heads = heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.qk_projection = qk_projection
        self.position = position
        self.rope_base = rope_base
        self.rotated_dim = (
            int(key_dim * rope_fraction) // 2 * 2 if position == "rope" else 0
        )
        self.eps = eps

        qk_multiplier = 2 if qk_projection == "direct_hadamard" else 1
        self.q_proj = nn.Linear(
            d_model, heads * key_dim * qk_multiplier, bias=False
        )
        self.k_proj = nn.Linear(
            d_model, heads * key_dim * qk_multiplier, bias=False
        )
        self.v_proj = nn.Linear(d_model, heads * value_dim, bias=False)

        if "conv" in self.qk_ops:
            self.q_conv: CausalDepthwiseConv1d | None = CausalDepthwiseConv1d(
                heads * key_dim, conv_width
            )
            self.k_conv: CausalDepthwiseConv1d | None = CausalDepthwiseConv1d(
                heads * key_dim, conv_width
            )
        else:
            self.q_conv = None
            self.k_conv = None
        self.v_conv: CausalDepthwiseConv1d | None = (
            CausalDepthwiseConv1d(heads * value_dim, conv_width)
            if "conv" in self.value_ops
            else None
        )

        resolved_state = "hadamard" if state is None else state
        resolved_lift = lift
        if qk_projection == "direct_hadamard" and lift is None:
            if state not in {None, "hadamard"}:
                raise ValueError(
                    "direct_hadamard feeds its product directly to the SSM; "
                    "use lift=... to compose an additional custom lift"
                )
            if feature_width not in {None, key_dim}:
                raise ValueError(
                    "direct_hadamard without a custom lift has width key_dim; "
                    "feature_width cannot change it"
                )
            resolved_lift = key()
        self.memory = ThetaMemory(
            key_dim,
            value_dim,
            heads,
            state=resolved_state,
            lift=resolved_lift,
            feature_width=feature_width,
            update=update,
            passes=passes,
            value_center=value_center,
            backend=backend,
            chunk=chunk,
            feature_norm=feature_norm,
            eps=eps,
        )
        self.z_proj = nn.Sequential(
            nn.Linear(d_model, value_dim, bias=False),
            nn.Linear(value_dim, heads * value_dim, bias=True),
        )
        self.out_norm = HeadRMSNorm(heads, value_dim, eps)
        self.out_proj = nn.Linear(heads * value_dim, d_model, bias=False)
        self.apply(self._initialize_linear)

    @staticmethod
    def _initialize_linear(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2**-2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _heads(self, tensor: torch.Tensor, width: int) -> torch.Tensor:
        batch, time, _ = tensor.shape
        return tensor.view(batch, time, self.heads, width).transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, heads, time, width = tensor.shape
        return tensor.transpose(1, 2).reshape(batch, time, heads * width)

    def _project_qk(
        self, hidden_states: torch.Tensor, projection: nn.Linear
    ) -> torch.Tensor:
        projected = projection(hidden_states)
        if self.qk_projection == "linear":
            return self._heads(projected, self.key_dim)
        batch, time, _ = projected.shape
        branches = projected.view(
            batch, time, self.heads, 2, self.key_dim
        )
        multiplied = branches[:, :, :, 0] * branches[:, :, :, 1]
        return multiplied.transpose(1, 2)

    def _normalize_stream(self, tensor: torch.Tensor, op: str) -> torch.Tensor:
        if op == "center":
            return tensor - tensor.mean(-1, keepdim=True)
        if op == "l1":
            return tensor / tensor.abs().sum(-1, keepdim=True).clamp_min(
                self.eps
            )
        if op == "l2":
            return F.normalize(tensor, p=2, dim=-1, eps=self.eps)
        if op == "rms":
            return tensor * torch.rsqrt(
                tensor.square().mean(-1, keepdim=True) + self.eps
            )
        raise ValueError(f"unknown normalization operation {op!r}")

    def _apply_qk_ops(
        self, query: torch.Tensor, key_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time = query.shape[2]
        for op in self.qk_ops:
            if op == "conv":
                assert self.q_conv is not None and self.k_conv is not None
                query = self._heads(
                    self.q_conv(self._merge_heads(query)), self.key_dim
                )
                key_tensor = self._heads(
                    self.k_conv(self._merge_heads(key_tensor)), self.key_dim
                )
            elif op == "silu":
                query = F.silu(query)
                key_tensor = F.silu(key_tensor)
            elif op == "rope":
                if self.rotated_dim > 0:
                    cos, sin = _rope_phases(
                        time,
                        self.rotated_dim,
                        self.rope_base,
                        query.device,
                    )
                    query = _apply_rope(query, cos, sin, self.rotated_dim)
                    key_tensor = _apply_rope(
                        key_tensor, cos, sin, self.rotated_dim
                    )
            else:
                query = self._normalize_stream(query, op)
                key_tensor = self._normalize_stream(key_tensor, op)
        return query, key_tensor

    def _apply_value_ops(self, value: torch.Tensor) -> torch.Tensor:
        for op in self.value_ops:
            if op == "conv":
                assert self.v_conv is not None
                value = self._heads(
                    self.v_conv(self._merge_heads(value)), self.value_dim
                )
            elif op == "silu":
                value = F.silu(value)
            else:
                value = self._normalize_stream(value, op)
        return value

    @property
    def state_size(self) -> int:
        """Memory-state and active convolution-cache floats."""
        conv_cache = 0
        if self.q_conv is not None:
            conv_cache += 2 * self.heads * self.key_dim * (
                self.q_conv.width - 1
            )
        if self.v_conv is not None:
            conv_cache += self.heads * self.value_dim * (
                self.v_conv.width - 1
            )
        return self.heads * self.memory.state_size + conv_cache

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"expected hidden rank 3 [B,T,D], got rank {hidden_states.ndim}"
            )
        if hidden_states.shape[-1] != self.d_model:
            raise ValueError(
                f"expected hidden width {self.d_model}, got "
                f"{hidden_states.shape[-1]}"
            )
        batch, time, _ = hidden_states.shape
        query = self._project_qk(hidden_states, self.q_proj)
        key_tensor = self._project_qk(hidden_states, self.k_proj)
        value = self._heads(self.v_proj(hidden_states), self.value_dim)
        query, key_tensor = self._apply_qk_ops(query, key_tensor)
        value = self._apply_value_ops(value)
        memory = self.memory(query, key_tensor, value)
        gate = F.silu(self.z_proj(hidden_states))
        gate = gate.view(batch, time, self.heads, self.value_dim).transpose(1, 2)
        output = self.out_norm(memory) * gate
        output = output.transpose(1, 2).reshape(
            batch, time, self.heads * self.value_dim
        )
        return self.out_proj(output)
