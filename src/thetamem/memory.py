"""ThetaMemory: a bounded-state associative memory over a lifted key."""

from __future__ import annotations

import math

import torch
from torch import nn

from . import scan
from .lift import Lift, LiftModule, state_lift

__all__ = ["ThetaMemory", "UPDATES", "BACKENDS", "VALUE_CENTERS"]

UPDATES = ("sum", "second_pass", "multi_pass", "delta")
BACKENDS = ("naive", "chunked", "fla")
VALUE_CENTERS = ("none", "running_mean", "exact_mean")


def _causal_mean(values: torch.Tensor) -> torch.Tensor:
    """Inclusive causal running mean of the value stream, per channel."""
    dtype = torch.promote_types(values.dtype, torch.float32)
    counts = torch.arange(
        1, values.shape[2] + 1, device=values.device, dtype=dtype
    ).view(1, 1, -1, 1)
    return (values.to(dtype).cumsum(2) / counts).to(values.dtype)


class ThetaMemory(nn.Module):
    """Signed multiplicative-lift memory.

    Takes projected queries, keys, and values of shape ``[B, H, T, *]`` and
    returns reads of shape ``[B, H, T, value_dim]``. The module owns only the
    lift parameters (the branch projections) and the update controls; query
    and key share the same lift weights, so the read kernel is symmetric.

    Reads carry no accumulated-mass denominator. Key/lift normalization is
    always explicit, and :class:`thetamem.layer.ThetaMemLayer` applies a
    per-head RMSNorm after the read.

    Args:
        key_dim: width of the projected key and query.
        value_dim: width of the values.
        heads: number of heads.
        state: one of ``"hadamard"``, ``"concat"``, ``"outer"`` — the three
            canonical state kinds. Ignored when ``lift`` is given.
        lift: a custom lift specification built from
            :func:`thetamem.branch` / :func:`thetamem.key` /
            :func:`thetamem.key_part` / :func:`thetamem.normalize` /
            :func:`thetamem.hadamard` / :func:`thetamem.outer` /
            :func:`thetamem.concat`.
        feature_width: default width of ``branch()`` factors; ``None`` means
            ``key_dim``.
        update: ``"sum"`` (pure additive), ``"second_pass"`` (additive base
            plus a correction state written from strict-prefix residuals),
            ``"multi_pass"`` (causal Richardson iterations over the inclusive
            lower-triangular key Gram), or ``"delta"`` (sequential delta
            rule).
        passes: number of Richardson steps for ``update="multi_pass"``.
            Each step carries one complete iterate and has a learned per-head
            step size. The first step starts at 0.1, making one pass identical
            to the default 90/10 blend of ``"second_pass"``; the final iterate
            is read directly.
        value_center: ``"none"``, ``"running_mean"``, or ``"exact_mean"``.
            Both options remove the value mean; the qualifier says how.
            ``"running_mean"`` writes values centered by their causal running
            mean and adds the current mean back after the read, so the
            earliest writes are centered by a poor estimate. ``"exact_mean"``
            removes the mean exactly: a ones channel rides along the values,
            so the same scan also accumulates the signed key mass, and the
            read subtracts ``mean * ones_read`` before adding the mean back.
            Both are subtractive; nothing is divided.
        backend: ``"naive"``, ``"chunked"``, or ``"fla"``.
        chunk: chunk length for the chunked backend.
        feature_norm: ``"none"``, ``"rms"``, or ``"l2"`` applied after the
            complete flat lift, or independently to each top-level outer
            factor.
    """

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        heads: int,
        *,
        state: str = "hadamard",
        lift: Lift | None = None,
        feature_width: int | None = None,
        update: str = "sum",
        passes: int = 2,
        value_center: str = "none",
        backend: str = "chunked",
        chunk: int = 256,
        feature_norm: str = "none",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if update not in UPDATES:
            raise ValueError(f"unknown update {update!r}; expected {UPDATES}")
        if value_center not in VALUE_CENTERS:
            raise ValueError(
                f"unknown value_center {value_center!r}; expected "
                f"{VALUE_CENTERS}"
            )
        if backend not in BACKENDS:
            raise ValueError(
                f"unknown backend {backend!r}; expected {BACKENDS}"
            )
        if chunk < 1:
            raise ValueError("chunk must be a positive integer")
        if isinstance(passes, bool) or not isinstance(passes, int) or passes < 1:
            raise ValueError("passes must be a positive integer")
        spec = lift if lift is not None else state_lift(state, feature_width)
        self.lift = LiftModule(
            spec,
            heads=heads,
            key_dim=key_dim,
            default_width=feature_width,
            feature_norm=feature_norm,
            eps=eps,
        )
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.heads = heads
        self.update = update
        self.passes = passes
        self.value_center = value_center
        self.backend = backend
        self.chunk = chunk
        self.eps = eps
        if update == "second_pass":
            # Blend weight between the base and correction reads;
            # sigmoid(log 9) = 0.9 starts the read close to the plain sum.
            self.mix_logit = nn.Parameter(
                torch.full((heads,), math.log(9.0))
            )
        else:
            self.register_parameter("mix_logit", None)
        if update == "multi_pass":
            # eta=0.1 matches the second-pass module's initial 90/10 blend:
            # one Richardson step is (1-eta)*base + eta*correction.
            self.pass_strength_logit = nn.Parameter(
                torch.full((passes, heads), -math.log(9.0))
            )
        else:
            self.register_parameter("pass_strength_logit", None)
        if update == "delta":
            # Raw learned write strength beta = sigmoid(logit).  Key/lift
            # normalization is deliberately explicit rather than hidden in
            # the update rule.  The conservative dimension-aware initial
            # value is 1 / (feature_width + 1).
            self.strength_logit = nn.Parameter(
                torch.full(
                    (heads,), -math.log(float(self.lift.feature_width))
                )
            )
        else:
            self.register_parameter("strength_logit", None)

    @property
    def state_shape(self) -> tuple[int, ...]:
        """Per-head shape of one memory state, value axis last."""
        return self.lift.state_shape(self.value_dim)

    @property
    def state_size(self) -> int:
        """Streaming state floats per head, all states included."""
        per_state = math.prod(self.state_shape)
        if self.value_center == "exact_mean":
            # The ones channel rides along the values in every state.
            per_state += math.prod(self.state_shape[:-1])
        if self.update == "second_pass":
            states = 2
        elif self.update == "multi_pass":
            states = 1 + self.passes
        else:
            states = 1
        total = states * per_state
        if self.value_center != "none":
            # Running value sum and its count.
            total += self.value_dim + 1
        return total

    @property
    def feature_width(self) -> int:
        return self.lift.feature_width

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("query, key, and value must have rank 4 [B,H,T,D]")
        if query.shape != key.shape:
            raise ValueError("query and key must share one shape")
        if value.shape[:-1] != key.shape[:-1]:
            raise ValueError("value must match key on batch/head/time axes")
        if query.shape[1] != self.heads:
            raise ValueError(f"expected {self.heads} heads, got {query.shape[1]}")
        if query.shape[-1] != self.key_dim:
            raise ValueError(
                f"expected query/key width {self.key_dim}, got "
                f"{query.shape[-1]}"
            )
        if value.shape[-1] != self.value_dim:
            raise ValueError(
                f"expected value width {self.value_dim}, got {value.shape[-1]}"
            )
        q_factors = self.lift(query)
        k_factors = self.lift(key)
        if self.value_center == "none":
            return self._read(q_factors, k_factors, value)
        mean = _causal_mean(value)
        if self.value_center == "running_mean":
            return self._read(q_factors, k_factors, value - mean) + mean
        # "exact_mean": a ones channel rides along the values so the same scan
        # also accumulates the signed key mass; the mean is removed exactly,
        # for every update, because each update is linear in its written
        # values. The mass is subtracted, never divided by.
        extended = torch.cat((value, torch.ones_like(value[..., :1])), dim=-1)
        read = self._read(q_factors, k_factors, extended)
        read_values = read[..., : self.value_dim]
        read_mass = read[..., self.value_dim :]
        return read_values - mean * read_mass + mean

    def _read(
        self,
        q_factors: tuple[torch.Tensor, ...],
        k_factors: tuple[torch.Tensor, ...],
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch one update over already-lifted factors; blend its reads."""
        if self.update == "sum":
            return scan.sum_read(
                q_factors,
                k_factors,
                values,
                backend=self.backend,
                chunk=self.chunk,
            )
        if self.update == "second_pass":
            base, correction = scan.second_pass_read(
                q_factors,
                k_factors,
                values,
                backend=self.backend,
                chunk=self.chunk,
            )
            weight = torch.sigmoid(self.mix_logit).to(base.dtype)
            weight = weight.view(1, -1, 1, 1)
            return weight * base + (1.0 - weight) * correction
        if self.update == "multi_pass":
            strengths = torch.sigmoid(self.pass_strength_logit)
            return scan.multi_pass_read(
                q_factors,
                k_factors,
                values,
                strengths,
                backend=self.backend,
                chunk=self.chunk,
            )
        reference = k_factors[0]
        strength = torch.sigmoid(self.strength_logit).to(reference.dtype)
        beta = strength.view(1, -1, 1).expand(
            reference.shape[0], -1, reference.shape[2]
        )
        return scan.delta_read(
            q_factors,
            k_factors,
            values,
            beta,
            backend=self.backend,
            chunk=self.chunk,
        )
