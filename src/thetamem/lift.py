"""Declarative key-lift algebra.

A lift describes how a projected key is transformed into the feature that
addresses the memory.  Flat expressions return one feature vector; ``outer``
returns one vector per state axis so the scan can keep the factorization until
an executor explicitly materializes the Kronecker feature.

The constructors are intentionally small and composable:

- ``key()`` and ``key_part()`` select the full projected key or an equal,
  contiguous part of it without adding parameters.
- ``branch()`` applies a learned per-head projection, optionally to a selected
  and normalized key part.
- ``normalize()`` applies local operations before an enclosing product.
- ``hadamard()`` multiplies equal-width blocks elementwise.
- ``concat()`` forms a direct sum of flat blocks.
- ``outer()`` makes two or more flat blocks separate state axes.

Nested ``outer`` expressions are forbidden.  They would not add expressive
power, while making the state-axis contract ambiguous.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn

__all__ = [
    "Lift",
    "branch",
    "key",
    "key_part",
    "normalize",
    "hadamard",
    "outer",
    "concat",
    "state_lift",
    "LiftModule",
    "STATE_KINDS",
]


@dataclass(frozen=True)
class Lift:
    """Base class for immutable lift specifications."""


@dataclass(frozen=True)
class Key(Lift):
    pass


@dataclass(frozen=True)
class KeyPart(Lift):
    index: int
    groups: int = 2


@dataclass(frozen=True)
class Normalize(Lift):
    source: Lift
    ops: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Branch(Lift):
    width: int | None = None
    source: Lift | None = None


@dataclass(frozen=True)
class Hadamard(Lift):
    factors: tuple[Lift, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Outer(Lift):
    factors: tuple[Lift, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Concat(Lift):
    blocks: tuple[Lift, ...] = field(default_factory=tuple)


_LOCAL_NORMALIZATIONS = {"center", "l1", "l2", "rms"}
_FEATURE_NORMS = {"none", "center", "rms", "l2"}


class _LiftedFactors(tuple):
    """Tuple-compatible factor bundle with scan metadata.

    A globally centered outer product is not itself one Kronecker product.
    Keeping the raw factors plus this flag lets the scan apply the exact
    rank-one centering correction without materializing the full feature.
    """

    def __new__(
        cls,
        factors: tuple[torch.Tensor, ...],
        *,
        centered: bool = False,
    ) -> "_LiftedFactors":
        instance = super().__new__(cls, factors)
        instance.centered = centered
        return instance


def key() -> Key:
    """The projected key itself as a feature block (no parameters)."""
    return Key()


def key_part(index: int, groups: int = 2) -> KeyPart:
    """One equal contiguous part of the projected key (no parameters).

    ``key_dim`` must be divisible by ``groups`` when the lift is materialized.
    """
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("key_part index must be an integer")
    if isinstance(groups, bool) or not isinstance(groups, int):
        raise TypeError("key_part groups must be an integer")
    if groups < 1:
        raise ValueError("key_part groups must be positive")
    if not 0 <= index < groups:
        raise ValueError(
            f"key_part index must be in [0, {groups}), got {index}"
        )
    return KeyPart(index, groups)


def _branch_source_is_supported(source: Lift) -> bool:
    if isinstance(source, (Key, KeyPart)):
        return True
    if isinstance(source, Normalize):
        return _branch_source_is_supported(source.source)
    return False


def branch(
    width: int | None = None,
    *,
    source: Lift | None = None,
) -> Branch:
    """A learned per-head projection to ``width`` features.

    ``source=None`` projects the full key and exactly preserves the original
    ``branch(width)`` behavior.  A ``key_part`` (optionally wrapped in local
    normalization) can be supplied to project only that part.  Every call is
    independently parameterized.
    """
    if width is not None and (
        isinstance(width, bool) or not isinstance(width, int) or width < 1
    ):
        raise ValueError("branch width must be a positive integer")
    if source is not None:
        if not isinstance(source, Lift):
            raise TypeError("branch source must be a lift expression")
        if not _branch_source_is_supported(source):
            raise TypeError(
                "branch source must be key(), key_part(), or local "
                "normalization around either"
            )
    return Branch(width, source)


def normalize(source: Lift, *ops: str) -> Lift:
    """Apply local normalizations to a flat lift block, from left to right.

    Supported operations are ``"center"``, ``"l1"``, ``"l2"``, and
    ``"rms"``.  ``"none"`` is accepted as an explicit no-op.  Omitting this
    constructor leaves the source raw.
    """
    if not isinstance(source, Lift):
        raise TypeError("normalize source must be a lift expression")
    if isinstance(source, Outer):
        raise TypeError("normalize cannot wrap outer; normalize its factors")
    if not ops:
        raise ValueError("normalize needs at least one operation")
    filtered: list[str] = []
    for op in ops:
        if op == "none":
            continue
        if op not in _LOCAL_NORMALIZATIONS:
            raise ValueError(
                f"unknown local normalization {op!r}; expected one of "
                f"{sorted(_LOCAL_NORMALIZATIONS)}"
            )
        filtered.append(op)
    if not filtered:
        return source
    return Normalize(source, tuple(filtered))


def _require_flat(node: Lift, context: str) -> None:
    if not isinstance(node, Lift):
        raise TypeError(f"{context} entries must be lift expressions")
    if isinstance(node, Outer):
        raise TypeError(f"{context} cannot contain outer")


def hadamard(*factors: Lift) -> Hadamard:
    """Elementwise product of two or more equal-width flat blocks."""
    if len(factors) < 2:
        raise ValueError("hadamard needs at least two factors")
    for factor in factors:
        _require_flat(factor, "hadamard")
    return Hadamard(tuple(factors))


def outer(*factors: Lift) -> Outer:
    """Tensor product of two or more blocks, one state axis per factor."""
    if len(factors) < 2:
        raise ValueError("outer needs at least two factors")
    for factor in factors:
        _require_flat(factor, "outer")
    return Outer(tuple(factors))


def concat(*blocks: Lift) -> Concat:
    """Direct sum (concatenation) of two or more flat feature blocks."""
    if len(blocks) < 2:
        raise ValueError("concat needs at least two blocks")
    for block in blocks:
        _require_flat(block, "concat")
    return Concat(tuple(blocks))


STATE_KINDS = ("hadamard", "concat", "outer")


def state_lift(state: str, feature_width: int | None = None) -> Lift:
    """Return one of the three named canonical lift trees."""
    if state == "hadamard":
        return hadamard(branch(feature_width), branch(feature_width))
    if state == "concat":
        return concat(key(), hadamard(branch(feature_width), branch(feature_width)))
    if state == "outer":
        return outer(branch(feature_width), branch(feature_width))
    raise ValueError(
        f"unknown state kind {state!r}; expected one of {STATE_KINDS}"
    )


def _part_width(node: KeyPart, key_dim: int) -> int:
    if key_dim % node.groups != 0:
        raise ValueError(
            f"key_dim {key_dim} must be divisible by key_part groups "
            f"{node.groups}"
        )
    return key_dim // node.groups


def _node_width(node: Lift, key_dim: int, default_width: int) -> int:
    if isinstance(node, Key):
        return key_dim
    if isinstance(node, KeyPart):
        return _part_width(node, key_dim)
    if isinstance(node, Normalize):
        return _node_width(node.source, key_dim, default_width)
    if isinstance(node, Branch):
        return node.width if node.width is not None else default_width
    if isinstance(node, Hadamard):
        widths = {_node_width(f, key_dim, default_width) for f in node.factors}
        if len(widths) != 1:
            raise ValueError(
                f"hadamard factors must share one width, got {sorted(widths)}"
            )
        return widths.pop()
    if isinstance(node, Concat):
        return sum(_node_width(b, key_dim, default_width) for b in node.blocks)
    raise TypeError(f"unexpected lift node {type(node).__name__}")


def axis_widths(spec: Lift, key_dim: int, default_width: int) -> tuple[int, ...]:
    """Widths of the state's key axes: one per outer factor, else one."""
    if isinstance(spec, Outer):
        return tuple(_node_width(f, key_dim, default_width) for f in spec.factors)
    return (_node_width(spec, key_dim, default_width),)


def _collect_branches(node: Lift) -> list[Branch]:
    if isinstance(node, Branch):
        return [node]
    if isinstance(node, (Key, KeyPart)):
        return []
    if isinstance(node, Normalize):
        return _collect_branches(node.source)
    if isinstance(node, Hadamard):
        return [b for f in node.factors for b in _collect_branches(f)]
    if isinstance(node, Concat):
        return [b for block in node.blocks for b in _collect_branches(block)]
    if isinstance(node, Outer):
        return [b for f in node.factors for b in _collect_branches(f)]
    raise TypeError(f"unexpected lift node {type(node).__name__}")


def _validate_branch_sources(node: Lift) -> None:
    """Defend materialization when internal dataclasses bypass factories."""
    if isinstance(node, Branch):
        if node.source is not None and not _branch_source_is_supported(
            node.source
        ):
            raise TypeError(
                "branch source must be key(), key_part(), or local "
                "normalization around either"
            )
        return
    if isinstance(node, Normalize):
        _validate_branch_sources(node.source)
        return
    if isinstance(node, Hadamard):
        for factor in node.factors:
            _validate_branch_sources(factor)
        return
    if isinstance(node, Concat):
        for block in node.blocks:
            _validate_branch_sources(block)
        return
    if isinstance(node, Outer):
        for factor in node.factors:
            _validate_branch_sources(factor)


class LiftModule(nn.Module):
    """Materialized lift that owns branch weights and evaluates features."""

    def __init__(
        self,
        spec: Lift,
        *,
        heads: int,
        key_dim: int,
        default_width: int | None = None,
        feature_norm: str = "none",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if feature_norm not in _FEATURE_NORMS:
            raise ValueError(f"unknown feature_norm {feature_norm!r}")
        default_width = key_dim if default_width is None else default_width
        _validate_branch_sources(spec)
        self.spec = spec
        self.heads = heads
        self.key_dim = key_dim
        self.default_width = default_width
        self.feature_norm = feature_norm
        self.eps = eps
        self.axis_widths = axis_widths(spec, key_dim, default_width)
        weights = []
        for node in _collect_branches(spec):
            width = node.width if node.width is not None else default_width
            source_width = (
                key_dim
                if node.source is None
                else _node_width(node.source, key_dim, default_width)
            )
            scale = 1 / math.sqrt(source_width)
            weights.append(
                nn.Parameter(
                    torch.randn(heads, width, source_width) * scale
                )
            )
        self.branch_weights = nn.ParameterList(weights)

    @property
    def feature_width(self) -> int:
        """Flat feature width (the product over all state key axes)."""
        return math.prod(self.axis_widths)

    @property
    def order(self) -> int:
        """Number of state key axes (one for a flat lift)."""
        return len(self.axis_widths)

    def state_shape(self, value_dim: int) -> tuple[int, ...]:
        """Per-head shape of one memory state, value axis last."""
        return (*self.axis_widths, value_dim)

    def _apply_ops(
        self, features: torch.Tensor, ops: tuple[str, ...]
    ) -> torch.Tensor:
        for op in ops:
            if op == "center":
                features = features - features.mean(-1, keepdim=True)
            elif op == "l1":
                features = features / features.abs().sum(
                    -1, keepdim=True
                ).clamp_min(self.eps)
            elif op == "l2":
                features = features / features.norm(
                    dim=-1, keepdim=True
                ).clamp_min(self.eps)
            elif op == "rms":
                features = features * torch.rsqrt(
                    features.square().mean(-1, keepdim=True) + self.eps
                )
            else:  # factories validate specs; retain a defensive boundary.
                raise ValueError(f"unknown local normalization {op!r}")
        return features

    def _normalize_top_level(self, features: torch.Tensor) -> torch.Tensor:
        if self.feature_norm == "center":
            return self._apply_ops(features, ("center",))
        if self.feature_norm == "rms":
            return self._apply_ops(features, ("rms",))
        if self.feature_norm == "l2":
            return self._apply_ops(features, ("l2",))
        return features

    def _evaluate(
        self, node: Lift, x: torch.Tensor, cursor: list[int]
    ) -> torch.Tensor:
        if isinstance(node, Key):
            return x
        if isinstance(node, KeyPart):
            width = _part_width(node, self.key_dim)
            start = node.index * width
            return x[..., start : start + width]
        if isinstance(node, Normalize):
            return self._apply_ops(
                self._evaluate(node.source, x, cursor), node.ops
            )
        if isinstance(node, Branch):
            source = x if node.source is None else self._evaluate(
                node.source, x, cursor
            )
            weight = self.branch_weights[cursor[0]]
            cursor[0] += 1
            return torch.einsum("hfi,bhti->bhtf", weight, source)
        if isinstance(node, Hadamard):
            product = self._evaluate(node.factors[0], x, cursor)
            for factor in node.factors[1:]:
                product = product * self._evaluate(factor, x, cursor)
            return product
        if isinstance(node, Concat):
            return torch.cat(
                [self._evaluate(block, x, cursor) for block in node.blocks],
                dim=-1,
            )
        raise TypeError(f"unexpected lift node {type(node).__name__}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Evaluate a ``[B, H, T, key_dim]`` projected key tensor."""
        if x.ndim != 4:
            raise ValueError(
                f"expected key rank 4 [B,H,T,D], got rank {x.ndim}"
            )
        if x.shape[1] != self.heads:
            raise ValueError(f"expected {self.heads} heads, got {x.shape[1]}")
        if x.shape[-1] != self.key_dim:
            raise ValueError(
                f"expected key width {self.key_dim}, got {x.shape[-1]}"
            )
        cursor = [0]
        if isinstance(self.spec, Outer):
            raw_factors = tuple(
                self._evaluate(factor, x, cursor)
                for factor in self.spec.factors
            )
            if self.feature_norm == "center":
                # Centering the complete outer feature is represented lazily.
                # The scan uses K_c = K - sum(q)sum(k)/feature_width and keeps
                # the original tensor-state axes.
                return _LiftedFactors(raw_factors, centered=True)
            factors = tuple(
                self._normalize_top_level(factor) for factor in raw_factors
            )
        else:
            factors = (
                self._normalize_top_level(self._evaluate(self.spec, x, cursor)),
            )
        return factors
