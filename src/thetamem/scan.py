"""Causal scans over lifted features.

Every memory in this package is a causal sum of feature/value writes,

    S_t = sum_{i<=t} Phi(k_i) (x) v_i,        y_t = <Phi(q_t), S_t>,

optionally followed by one correction (see ``second_pass``), by several causal
triangular Richardson iterates (see ``multi_pass``), or replaced by the
sequential delta rule (see ``delta``). Completed record sets can also be fit
with the non-causal replay solvers below. Features arrive as a tuple of factor
tensors — one ``[B, H, T, width]`` tensor per state key axis. Flat lifts have
one factor; ``outer`` lifts have two or more, and their scores factorize as

    <a_q (x) b_q, a_k (x) b_k> = (a_q . a_k)(b_q . b_k),

so the chunked additive executors avoid the full product width for local score
tiles. Exact carried-state work still touches that width; delta and FLA
materialize it explicitly.

Three executors:

- ``naive`` — masked quadratic reference. Obviously correct, O(T^2) memory.
- ``chunked`` — chunk-diagonal score tiles plus an explicit inter-chunk state.
  Flat lifts batch all tiles in one call; factorized lifts walk chunks
  sequentially so the product width appears only in the carry.
- ``fla`` — optional delegation to the ``flash-linear-attention`` package for
  flat features. Fails closed when the package or a kernel is unavailable.

The inter-chunk state is always an *exclusive* prefix — computed by shifting
the inclusive cumulative sum, never by subtracting a chunk's own update from
it, because that subtraction is exact only up to rounding and would let a
chunk's own tokens leak into the state it reads.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "flatten_factors",
    "sum_read",
    "second_pass_read",
    "multi_pass_read",
    "delta_read",
    "REPLAY_SOLVERS",
    "replay_fit",
    "replay_read",
]

Factors = tuple[torch.Tensor, ...]
REPLAY_SOLVERS = ("cg", "richardson", "heavy_ball", "delta")


def _features_centered(factors: Factors) -> bool:
    """Whether factors lazily represent a globally centered outer feature."""
    return bool(getattr(factors, "centered", False))


def _matching_center_mode(q_factors: Factors, k_factors: Factors) -> bool:
    q_centered = _features_centered(q_factors)
    k_centered = _features_centered(k_factors)
    if q_centered != k_centered:
        raise ValueError("query and key features must use the same centering")
    return q_centered


def _flatten_raw_factors(factors: Factors) -> torch.Tensor:
    """Materialize a Kronecker feature without applying lazy centering."""
    flat = factors[0]
    for factor in factors[1:]:
        flat = (flat.unsqueeze(-1) * factor.unsqueeze(-2)).reshape(
            *flat.shape[:-1], -1
        )
    return flat


def flatten_factors(factors: Factors) -> torch.Tensor:
    """Materialize the represented feature vector per token.

    A factor bundle produced by ``feature_norm="center"`` is centered only
    here. Factorized chunked paths use the equivalent rank-one score
    correction and do not call this function for their local score tiles.
    """
    flat = _flatten_raw_factors(factors)
    if _features_centered(factors):
        flat = flat - flat.mean(-1, keepdim=True)
    return flat


def _state_dtype(reference: torch.Tensor) -> torch.dtype:
    return torch.promote_types(reference.dtype, torch.float32)


def _state_update(
    keys: torch.Tensor,
    values: torch.Tensor,
    dtype: torch.dtype,
    centered: bool = False,
) -> torch.Tensor:
    """Compute a carried-state update in the state accumulation dtype."""
    with torch.autocast(device_type=keys.device.type, enabled=False):
        update = torch.matmul(
            keys.to(dtype).transpose(-1, -2), values.to(dtype)
        )
        if centered:
            # Project the complete feature axis, C=I-11^T/F.  The state keeps
            # its original width and a raw query reads it exactly because the
            # projected state's feature-axis sum is zero.
            update = update - update.mean(-2, keepdim=True)
        return update


def _state_read(
    query: torch.Tensor,
    state: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Read a carried state in its dtype, casting only the finished read."""
    with torch.autocast(device_type=query.device.type, enabled=False):
        return torch.matmul(query.to(state.dtype), state).to(output_dtype)


def _flatten_state_factors(
    factors: Factors, dtype: torch.dtype
) -> torch.Tensor:
    """Materialize factors after promotion, outside mixed-precision autocast."""
    with torch.autocast(device_type=factors[0].device.type, enabled=False):
        return _flatten_raw_factors(
            tuple(factor.to(dtype) for factor in factors)
        )


def _factor_scores(
    q_factors: Factors,
    k_factors: Factors,
    centered: bool = False,
) -> torch.Tensor:
    """Product of per-factor score matrices: ``[..., Tq, Tk]``."""
    score = torch.matmul(q_factors[0], k_factors[0].transpose(-1, -2))
    for q_factor, k_factor in zip(q_factors[1:], k_factors[1:]):
        score = score * torch.matmul(q_factor, k_factor.transpose(-1, -2))
    if centered:
        q_sum = q_factors[0].sum(-1)
        k_sum = k_factors[0].sum(-1)
        width = q_factors[0].shape[-1]
        for q_factor, k_factor in zip(q_factors[1:], k_factors[1:]):
            q_sum = q_sum * q_factor.sum(-1)
            k_sum = k_sum * k_factor.sum(-1)
            width *= q_factor.shape[-1]
        score = score - (
            q_sum.unsqueeze(-1) * k_sum.unsqueeze(-2) / width
        )
    return score


def _pad_time(tensor: torch.Tensor, pad: int) -> torch.Tensor:
    if pad == 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, 0, 0, pad))


def _tiles(tensor: torch.Tensor, tiles: int, width: int) -> torch.Tensor:
    batch, heads, _, dim = tensor.shape
    return tensor.view(batch, heads, tiles, width, dim)


def _exclusive_cumsum(update: torch.Tensor) -> torch.Tensor:
    inclusive = update.cumsum(2)
    return torch.cat(
        [torch.zeros_like(update[:, :, :1]), inclusive[:, :, :-1]], dim=2
    )


# ---------------------------------------------------------------------------
# sum
# ---------------------------------------------------------------------------


def _sum_naive(
    q_factors: Factors,
    k_factors: Factors,
    values: torch.Tensor,
    centered: bool = False,
) -> torch.Tensor:
    score = _factor_scores(q_factors, k_factors, centered).tril()
    return torch.matmul(score, values)


def _sum_chunked_flat(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    batch, heads, time, _ = query.shape
    width = min(chunk, time)
    tiles = -(-time // width)
    if tiles == 1:
        return _sum_naive((query,), (keys,), values)
    pad = tiles * width - time
    qt = _tiles(_pad_time(query, pad), tiles, width)
    kt = _tiles(_pad_time(keys, pad), tiles, width)
    vt = _tiles(_pad_time(values, pad), tiles, width)
    score = torch.matmul(qt, kt.transpose(-1, -2))
    out = torch.matmul(score.tril(), vt)
    update = _state_update(kt, vt, _state_dtype(query))
    states = _exclusive_cumsum(update)
    out = out + _state_read(qt, states, out.dtype)
    return out.reshape(batch, heads, tiles * width, -1)[:, :, :time]


def _sum_chunked_factored(
    q_factors: Factors,
    k_factors: Factors,
    values: torch.Tensor,
    chunk: int,
    centered: bool = False,
) -> torch.Tensor:
    time = values.shape[2]
    state: torch.Tensor | None = None
    state_dtype = _state_dtype(values)
    outputs = []
    for start in range(0, time, chunk):
        stop = min(start + chunk, time)
        bq = tuple(f[:, :, start:stop] for f in q_factors)
        bk = tuple(f[:, :, start:stop] for f in k_factors)
        bv = values[:, :, start:stop]
        block = torch.matmul(_factor_scores(bq, bk, centered).tril(), bv)
        if state is not None:
            block = block + _state_read(
                _flatten_state_factors(bq, state_dtype), state, block.dtype
            )
        outputs.append(block)
        if stop < time:
            flat_keys = _flatten_state_factors(bk, state_dtype)
            update = _state_update(flat_keys, bv, state_dtype, centered)
            state = update if state is None else state + update
    return torch.cat(outputs, dim=2)


def sum_read(
    q_factors: Factors,
    k_factors: Factors,
    values: torch.Tensor,
    *,
    backend: str = "chunked",
    chunk: int = 256,
) -> torch.Tensor:
    """Read ``y_t = <Phi(q_t), S_t>`` from the causal sum state."""
    centered = _matching_center_mode(q_factors, k_factors)
    if backend == "naive":
        return _sum_naive(q_factors, k_factors, values, centered)
    if backend == "chunked":
        if len(q_factors) == 1:
            query = flatten_factors(q_factors) if centered else q_factors[0]
            keys = flatten_factors(k_factors) if centered else k_factors[0]
            return _sum_chunked_flat(query, keys, values, chunk)
        return _sum_chunked_factored(
            q_factors, k_factors, values, chunk, centered
        )
    if backend == "fla":
        return _sum_fla(flatten_factors(q_factors), flatten_factors(k_factors), values)
    raise ValueError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# second-pass correction
# ---------------------------------------------------------------------------


def _second_pass_naive(
    q_factors: Factors,
    k_factors: Factors,
    values: torch.Tensor,
    centered: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_score = _factor_scores(k_factors, k_factors, centered)
    prediction = torch.matmul(key_score.tril(-1), values)
    residual = values - prediction
    read_score = _factor_scores(q_factors, k_factors, centered).tril()
    return torch.matmul(read_score, values), torch.matmul(read_score, residual)


def _second_pass_chunked_flat(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, time, value_dim = values.shape
    width = min(chunk, time)
    tiles = -(-time // width)
    if tiles == 1:
        return _second_pass_naive((query,), (keys,), values)
    pad = tiles * width - time
    qt = _tiles(_pad_time(query, pad), tiles, width)
    kt = _tiles(_pad_time(keys, pad), tiles, width)
    vt = _tiles(_pad_time(values, pad), tiles, width)
    state_dtype = _state_dtype(query)
    key_score = torch.matmul(kt, kt.transpose(-1, -2))
    base_update = _state_update(kt, vt, state_dtype)
    base_states = _exclusive_cumsum(base_update)
    prediction = _state_read(kt, base_states, kt.dtype)
    prediction = prediction + torch.matmul(key_score.tril(-1), vt)
    residual = vt - prediction
    residual_update = _state_update(kt, residual, state_dtype)
    residual_states = _exclusive_cumsum(residual_update)
    packed_values = torch.cat((vt, residual), dim=-1)
    packed_states = torch.cat((base_states, residual_states), dim=-1)
    read_score = torch.matmul(qt, kt.transpose(-1, -2)).tril()
    out = torch.matmul(read_score, packed_values)
    out = out + _state_read(qt, packed_states, out.dtype)
    out = out.reshape(batch, heads, tiles * width, -1)[:, :, :time]
    return out[..., :value_dim], out[..., value_dim:]


def _second_pass_chunked_factored(
    q_factors: Factors,
    k_factors: Factors,
    values: torch.Tensor,
    chunk: int,
    centered: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    time = values.shape[2]
    value_dim = values.shape[-1]
    state_dtype = _state_dtype(values)
    base_state: torch.Tensor | None = None
    residual_state: torch.Tensor | None = None
    outputs = []
    for start in range(0, time, chunk):
        stop = min(start + chunk, time)
        bq = tuple(f[:, :, start:stop] for f in q_factors)
        bk = tuple(f[:, :, start:stop] for f in k_factors)
        bv = values[:, :, start:stop]
        flat_keys = _flatten_state_factors(bk, state_dtype)
        key_score = _factor_scores(bk, bk, centered)
        prediction = torch.matmul(key_score.tril(-1), bv)
        if base_state is not None:
            prediction = prediction + _state_read(
                flat_keys, base_state, prediction.dtype
            )
        residual = bv - prediction
        packed = torch.cat((bv, residual), dim=-1)
        block = torch.matmul(
            _factor_scores(bq, bk, centered).tril(), packed
        )
        if base_state is not None:
            packed_states = torch.cat((base_state, residual_state), dim=-1)
            block = block + _state_read(
                _flatten_state_factors(bq, state_dtype),
                packed_states,
                block.dtype,
            )
        outputs.append(block)
        if stop < time:
            base_update = _state_update(
                flat_keys, bv, state_dtype, centered
            )
            residual_update = _state_update(
                flat_keys, residual, state_dtype, centered
            )
            base_state = (
                base_update if base_state is None else base_state + base_update
            )
            residual_state = (
                residual_update
                if residual_state is None
                else residual_state + residual_update
            )
    out = torch.cat(outputs, dim=2)
    return out[..., :value_dim], out[..., value_dim:]


def second_pass_read(
    q_factors: Factors,
    k_factors: Factors,
    values: torch.Tensor,
    *,
    backend: str = "chunked",
    chunk: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Base and correction reads of the second-pass corrected memory.

    The first pass accumulates the base state ``M_t = sum Phi(k_i) (x) v_i``
    and forms every strict-prefix prediction ``p_t = <Phi(k_t), M_{t-1}>``.
    The second pass accumulates the correction state from the finalized
    residuals ``r_t = v_t - p_t``. Both reads are returned; the caller blends
    them. The strict-prefix boundary is mandatory — an inclusive prediction
    would leak the token's own value into its residual.
    """
    centered = _matching_center_mode(q_factors, k_factors)
    if backend == "naive":
        return _second_pass_naive(
            q_factors, k_factors, values, centered
        )
    if backend == "chunked":
        if len(q_factors) == 1:
            query = flatten_factors(q_factors) if centered else q_factors[0]
            keys = flatten_factors(k_factors) if centered else k_factors[0]
            return _second_pass_chunked_flat(
                query, keys, values, chunk
            )
        return _second_pass_chunked_factored(
            q_factors, k_factors, values, chunk, centered
        )
    if backend == "fla":
        return _second_pass_fla(
            flatten_factors(q_factors), flatten_factors(k_factors), values
        )
    raise ValueError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# multi-pass correction
# ---------------------------------------------------------------------------


def _multi_pass_naive(
    q_factors: Factors,
    k_factors: Factors,
    values: torch.Tensor,
    strengths: torch.Tensor,
    centered: bool = False,
) -> torch.Tensor:
    read_score = _factor_scores(q_factors, k_factors, centered).tril()
    key_score = _factor_scores(k_factors, k_factors, centered).tril(-1)
    iterate = values
    for index in range(strengths.shape[0]):
        prediction = iterate + torch.matmul(key_score, iterate)
        strength = strengths[index].view(1, -1, 1, 1).to(values.dtype)
        iterate = iterate + strength * (values - prediction)
    return torch.matmul(read_score, iterate)


def _multi_pass_chunked(
    q_factors: Factors,
    k_factors: Factors,
    values: torch.Tensor,
    strengths: torch.Tensor,
    chunk: int,
    centered: bool = False,
) -> torch.Tensor:
    time = values.shape[2]
    passes = strengths.shape[0]
    state_dtype = _state_dtype(values)
    states: list[torch.Tensor | None] = [None] * (passes + 1)
    outputs = []
    for start in range(0, time, chunk):
        stop = min(start + chunk, time)
        bq = tuple(f[:, :, start:stop] for f in q_factors)
        bk = tuple(f[:, :, start:stop] for f in k_factors)
        bv = values[:, :, start:stop]
        read_score = _factor_scores(bq, bk, centered).tril()
        key_score = _factor_scores(bk, bk, centered).tril(-1)
        carried = states[0] is not None
        flat_keys = (
            _flatten_state_factors(bk, state_dtype)
            if carried or stop < time
            else None
        )
        flat_queries = (
            _flatten_state_factors(bq, state_dtype) if carried else None
        )
        iterates = [bv]
        iterate = bv
        for index in range(passes):
            # Read the completed previous iterate. Passes are sequential;
            # time within each pass remains a causal scan.
            prediction = iterate + torch.matmul(key_score, iterate)
            if carried:
                prediction = prediction + _state_read(
                    flat_keys, states[index], prediction.dtype
                )
            strength = strengths[index].view(1, -1, 1, 1).to(bv.dtype)
            iterate = iterate + strength * (bv - prediction)
            iterates.append(iterate)
        block = torch.matmul(read_score, iterate)
        if carried:
            block = block + _state_read(
                flat_queries, states[-1], block.dtype
            )
        outputs.append(block)
        if stop < time:
            for index, iterate_values in enumerate(iterates):
                update = _state_update(
                    flat_keys, iterate_values, state_dtype, centered
                )
                states[index] = (
                    update if states[index] is None else states[index] + update
                )
    return torch.cat(outputs, dim=2)


def _multi_pass_fla(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    strengths: torch.Tensor,
) -> torch.Tensor:
    iterate = values
    prior_keys = torch.nn.functional.pad(keys[:, :, :-1], (0, 0, 1, 0))
    for index in range(strengths.shape[0]):
        prior_iterate = torch.nn.functional.pad(
            iterate[:, :, :-1], (0, 0, 1, 0)
        )
        prediction = iterate + _sum_fla(keys, prior_keys, prior_iterate)
        strength = strengths[index].view(1, -1, 1, 1).to(values.dtype)
        iterate = iterate + strength * (values - prediction)
    return _sum_fla(query, keys, iterate)


def multi_pass_read(
    q_factors: Factors,
    k_factors: Factors,
    values: torch.Tensor,
    strengths: torch.Tensor,
    *,
    backend: str = "chunked",
    chunk: int = 256,
) -> torch.Tensor:
    """Read the final causal Richardson iterate.

    Let ``A = I + tril(Phi(K) Phi(K)^T, -1)`` be the unit-diagonal causal
    overlap operator and initialize the coefficient iterate with the ordinary
    additive write, ``W_0 = V``. Pass ``p`` computes
    ``W_{p+1} = W_p + eta_p (V - A W_p)``.

    The prediction reads the prior *completed* pass, including its write at
    the current position. This is a scan-parallel approximation to the causal
    triangular system, not a least-squares solver. A single unit-strength pass
    is exactly the correction returned by :func:`second_pass_read`; identity
    overlap leaves ``W_0`` unchanged instead of adding one value copy per pass.

    ``strengths`` has shape ``[passes, heads]``. The returned tensor is the
    read of the last full iterate, shape ``[B, H, T, V]``. Each pass needs a
    separate carried state in the chunked implementation.
    """
    centered = _matching_center_mode(q_factors, k_factors)
    if backend == "naive":
        return _multi_pass_naive(
            q_factors, k_factors, values, strengths, centered
        )
    if backend == "chunked":
        if len(q_factors) == 1 and centered:
            q_factors = (flatten_factors(q_factors),)
            k_factors = (flatten_factors(k_factors),)
            centered = False
        return _multi_pass_chunked(
            q_factors, k_factors, values, strengths, chunk, centered
        )
    if backend == "fla":
        return _multi_pass_fla(
            flatten_factors(q_factors),
            flatten_factors(k_factors),
            values,
            strengths,
        )
    raise ValueError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# delta rule
# ---------------------------------------------------------------------------


def _delta_naive(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    batch, heads, time, width = keys.shape
    value_dim = values.shape[-1]
    state_dtype = _state_dtype(query)
    state = torch.zeros(
        batch, heads, width, value_dim, dtype=state_dtype, device=keys.device
    )
    outputs = []
    for t in range(time):
        key_t = keys[:, :, t].to(state_dtype)
        value_t = values[:, :, t].to(state_dtype)
        beta_t = beta[:, :, t].unsqueeze(-1).to(state_dtype)
        prediction = torch.einsum("bhw,bhwv->bhv", key_t, state)
        state = state + torch.einsum(
            "bhw,bhv->bhwv", beta_t * key_t, value_t - prediction
        )
        outputs.append(
            torch.einsum("bhw,bhwv->bhv", query[:, :, t].to(state_dtype), state)
        )
    return torch.stack(outputs, dim=2).to(query.dtype)


def _delta_chunked(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    batch, heads, time, width = keys.shape
    value_dim = values.shape[-1]
    state_dtype = _state_dtype(query)
    state = torch.zeros(
        batch, heads, width, value_dim, dtype=state_dtype, device=keys.device
    )
    outputs = []
    for start in range(0, time, chunk):
        stop = min(start + chunk, time)
        block_q = query[:, :, start:stop].to(state_dtype)
        block_k = keys[:, :, start:stop].to(state_dtype)
        block_v = values[:, :, start:stop].to(state_dtype)
        block_beta = beta[:, :, start:stop].unsqueeze(-1).to(state_dtype)
        scaled_keys = block_beta * block_k
        # Solve (I + tril(BKK^T, -1)) U = B (V - K S) for the pseudo values U
        # such that S_next = S + K^T U reproduces the sequential recurrence.
        interaction = torch.matmul(
            scaled_keys, block_k.transpose(-1, -2)
        ).tril(-1)
        identity = torch.eye(
            interaction.shape[-1], dtype=state_dtype, device=keys.device
        )
        rhs = block_beta * (block_v - torch.matmul(block_k, state))
        pseudo = torch.linalg.solve_triangular(
            identity + interaction, rhs, upper=False
        )
        read_score = torch.matmul(block_q, block_k.transpose(-1, -2)).tril()
        block_out = torch.matmul(block_q, state) + torch.matmul(
            read_score, pseudo
        )
        outputs.append(block_out)
        state = state + torch.matmul(block_k.transpose(-1, -2), pseudo)
    return torch.cat(outputs, dim=2).to(query.dtype)


def delta_read(
    q_factors: Factors,
    k_factors: Factors,
    values: torch.Tensor,
    beta: torch.Tensor,
    *,
    backend: str = "chunked",
    chunk: int = 256,
) -> torch.Tensor:
    """Sequential delta rule over the lifted key.

    ``S_t = S_{t-1} + beta_t Phi_t (v_t - S_{t-1}^T Phi_t)^T`` with the write
    strength ``beta`` supplied per token and head. The delta rule couples the
    state axes, so factorized lifts are materialized to their flat features
    first; the flat width is the practical bound for this update.
    """
    _matching_center_mode(q_factors, k_factors)
    query = flatten_factors(q_factors)
    keys = flatten_factors(k_factors)
    if backend == "naive":
        return _delta_naive(query, keys, values, beta)
    if backend == "chunked":
        return _delta_chunked(query, keys, values, beta, chunk)
    if backend == "fla":
        return _delta_fla(query, keys, values, beta)
    raise ValueError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# offline replay solvers
# ---------------------------------------------------------------------------


def _validate_replay_inputs(
    factors: Factors, values: torch.Tensor | None = None
) -> None:
    if not factors:
        raise ValueError("at least one factor is required")
    reference = factors[0]
    if reference.ndim != 4:
        raise ValueError("factors must have shape [B,H,T,F]")
    prefix = reference.shape[:3]
    for factor in factors[1:]:
        if factor.ndim != 4 or factor.shape[:3] != prefix:
            raise ValueError("all factors must share [B,H,T]")
        if factor.device != reference.device:
            raise ValueError("all factors must be on one device")
    if values is not None:
        if values.ndim != 4 or values.shape[:3] != prefix:
            raise ValueError("values must share factor axes [B,H,T]")
        if values.device != reference.device:
            raise ValueError("factors and values must be on one device")


def _factorized_write(
    factors: Factors,
    values: torch.Tensor,
    dtype: torch.dtype,
    centered: bool = False,
) -> torch.Tensor:
    """Reduce factorized feature/value writes into a dense tensor state.

    Global feature centering is the projection ``C = I - 11^T/R`` over all
    factor axes together.  Applying it to the completed state is exactly the
    same as centering every Kronecker feature, without flattening the factors.
    """
    factor_axes = list(range(3, 3 + len(factors)))
    value_axis = 3 + len(factors)
    operands: list[object] = []
    with torch.autocast(device_type=values.device.type, enabled=False):
        for factor, axis in zip(factors, factor_axes):
            operands.extend((factor.to(dtype), [0, 1, 2, axis]))
        operands.extend((values.to(dtype), [0, 1, 2, value_axis]))
        state = torch.einsum(
            *operands, [0, 1, *factor_axes, value_axis]
        )
        if centered:
            feature_dims = tuple(range(2, state.ndim - 1))
            state = state - state.mean(dim=feature_dims, keepdim=True)
        return state


def _factorized_read(
    factors: Factors,
    state: torch.Tensor,
    output_dtype: torch.dtype,
    centered: bool = False,
) -> torch.Tensor:
    """Contract factorized features with a dense tensor state."""
    factor_axes = list(range(3, 3 + len(factors)))
    value_axis = 3 + len(factors)
    operands: list[object] = []
    with torch.autocast(device_type=state.device.type, enabled=False):
        for factor, axis in zip(factors, factor_axes):
            operands.extend((factor.to(state.dtype), [0, 1, 2, axis]))
        operands.extend(
            (state, [0, 1, *factor_axes, value_axis])
        )
        output = torch.einsum(
            *operands, [0, 1, 2, value_axis]
        )
        if centered:
            # (C phi)^T S = phi^T S - mean(phi) * 1^T S.  Fitted centered
            # states already have zero feature sum; retaining the correction
            # also makes replay_read exact for an externally supplied state.
            feature_sum = factors[0].to(state.dtype).sum(-1)
            width = factors[0].shape[-1]
            for factor in factors[1:]:
                feature_sum = feature_sum * factor.to(state.dtype).sum(-1)
                width *= factor.shape[-1]
            feature_dims = tuple(range(2, state.ndim - 1))
            state_sum = state.sum(dim=feature_dims)
            output = output - (
                feature_sum.unsqueeze(-1)
                * state_sum.unsqueeze(2)
                / width
            )
        return output.to(output_dtype)


def _state_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Independent inner products for every batch/head/value channel."""
    factor_axes = tuple(range(2, left.ndim - 1))
    return (left * right).sum(dim=factor_axes)


def _state_scale(scalar: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    factor_dims = state.ndim - 3
    return scalar.view(
        scalar.shape[0], scalar.shape[1], *([1] * factor_dims), scalar.shape[2]
    )


def replay_read(q_factors: Factors, state: torch.Tensor) -> torch.Tensor:
    """Read an offline fitted dense state without flattening its factors.

    ``state`` is the tensor returned by :func:`replay_fit`, with shape
    ``[B, H, F1, ..., Fm, V]``. Queries may have a different time length from
    the fitted keys but must have the same batch, head, and factor widths.
    """
    _validate_replay_inputs(q_factors)
    expected = (
        q_factors[0].shape[0],
        q_factors[0].shape[1],
        *(factor.shape[-1] for factor in q_factors),
    )
    if state.ndim != len(q_factors) + 3 or state.shape[:-1] != expected:
        raise ValueError(
            "state must have shape [B,H,F1,...,Fm,V] matching the factors"
        )
    if state.device != q_factors[0].device:
        raise ValueError("queries and state must be on one device")
    return _factorized_read(
        q_factors,
        state,
        q_factors[0].dtype,
        _features_centered(q_factors),
    )


def replay_fit(
    k_factors: Factors,
    values: torch.Tensor,
    *,
    solver: str = "cg",
    iterations: int = 8,
    strength: float | None = None,
    momentum: float | None = None,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    """Fit one non-causal memory state by replaying a completed record set.

    This is an offline/prefill solver: every iteration may read every record,
    so it must not be used to produce causal outputs inside the fitted span.
    It never constructs a ``T x T`` Gram matrix, and the persistent state keeps
    its factor axes, shape ``[B, H, F1, ..., Fm, V]``.

    Peak memory is not proportional to that state alone. The write and read
    contractions sum over the token axis, which no pairwise contraction path
    can do without an intermediate that still carries ``T``: the chosen path
    materializes a temporary of order ``T * prod(F_i)`` or ``T * F1 * V``,
    whichever the einsum optimizer prefers. At ``T=512`` with ``32 x 32``
    factors that is a ``512 x 1024`` temporary per call. Replay is therefore
    memory-bounded by the context length and can exhaust device memory on a
    long one; the causal :func:`sum_read` path has no such intermediate because
    it tiles the token axis by construction.

    ``solver="cg"`` runs independent CGLS/CGNR iterations per value channel
    and is the default. ``"richardson"`` requires an explicit positive
    ``strength``. ``"heavy_ball"`` additionally requires explicit
    ``momentum`` in ``[0, 1)``; neither parameter is made stable
    automatically. ``"delta"`` performs cyclic sequential delta sweeps,
    reusing the final state between sweeps; its raw ``strength`` defaults to
    one and is not divided by feature norm.
    """
    _validate_replay_inputs(k_factors, values)
    if solver not in REPLAY_SOLVERS:
        raise ValueError(
            f"unknown replay solver {solver!r}; expected {REPLAY_SOLVERS}"
        )
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise ValueError("iterations must be a positive integer")
    if iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")

    if solver in ("richardson", "heavy_ball") and strength is None:
        raise ValueError(f"solver={solver!r} requires an explicit strength")
    if solver == "delta" and strength is None:
        strength = 1.0
    if strength is not None:
        strength = float(strength)
        if not math.isfinite(strength) or strength <= 0:
            raise ValueError("strength must be finite and positive")
    if solver == "heavy_ball":
        if momentum is None:
            raise ValueError("solver='heavy_ball' requires explicit momentum")
        momentum = float(momentum)
        if not math.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError("momentum must be finite and in [0, 1)")
    elif momentum is not None:
        raise ValueError("momentum is only valid for solver='heavy_ball'")

    dtype = _state_dtype(values)
    centered = _features_centered(k_factors)
    rhs = _factorized_write(k_factors, values, dtype, centered)
    state = torch.zeros_like(rhs)

    if solver == "delta":
        assert strength is not None
        for _ in range(iterations):
            for index in range(values.shape[2]):
                factors_t = tuple(
                    factor[:, :, index : index + 1] for factor in k_factors
                )
                prediction = _factorized_read(
                    factors_t, state, dtype, centered
                )
                residual = strength * (
                    values[:, :, index : index + 1].to(dtype) - prediction
                )
                state = state + _factorized_write(
                    factors_t, residual, dtype, centered
                )
        return state

    def normal_action(candidate: torch.Tensor) -> torch.Tensor:
        prediction = _factorized_read(
            k_factors, candidate, dtype, centered
        )
        return _factorized_write(
            k_factors, prediction, dtype, centered
        )

    if solver in ("richardson", "heavy_ball"):
        assert strength is not None
        velocity = torch.zeros_like(state)
        momentum_value = 0.0 if momentum is None else momentum
        for _ in range(iterations):
            residual = rhs - normal_action(state)
            velocity = momentum_value * velocity + strength * residual
            state = state + velocity
        return state

    residual = rhs
    direction = residual
    squared = _state_inner(residual, residual)
    initial_squared = squared
    threshold = initial_squared * (tolerance * tolerance)
    tiny = torch.finfo(dtype).eps
    for _ in range(iterations):
        mapped = normal_action(direction)
        denominator = _state_inner(direction, mapped)
        active = (squared > threshold) & (denominator > tiny)
        alpha = torch.where(
            active, squared / denominator.clamp_min(tiny), torch.zeros_like(squared)
        )
        state = state + _state_scale(alpha, state) * direction
        residual_next = residual - _state_scale(alpha, state) * mapped
        squared_next = _state_inner(residual_next, residual_next)
        active_next = squared_next > threshold
        beta = torch.where(
            active & active_next,
            squared_next / squared.clamp_min(tiny),
            torch.zeros_like(squared),
        )
        direction_next = residual_next + _state_scale(beta, state) * direction
        direction = torch.where(
            _state_scale(active_next, state),
            direction_next,
            torch.zeros_like(direction_next),
        )
        residual = residual_next
        squared = squared_next
    return state


# ---------------------------------------------------------------------------
# fla delegation (optional dependency, fail-closed)
# ---------------------------------------------------------------------------


def _fla_unavailable(error: Exception) -> RuntimeError:
    return RuntimeError(
        "backend='fla' requires the flash-linear-attention package and a "
        "supported GPU kernel path; install with `pip install thetamem[fla]` "
        f"or use backend='chunked'. Underlying error: {error}"
    )


def _sum_fla(
    query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    try:
        from fla.ops.linear_attn import chunk_linear_attn
    except Exception as error:  # pragma: no cover - optional dependency
        raise _fla_unavailable(error)
    output, _ = chunk_linear_attn(
        query.transpose(1, 2),
        keys.transpose(1, 2),
        values.transpose(1, 2),
        scale=1.0,
        normalize=False,
    )
    return output.transpose(1, 2)


def _second_pass_fla(
    query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    prior_keys = torch.nn.functional.pad(keys[:, :, :-1], (0, 0, 1, 0))
    prior_values = torch.nn.functional.pad(values[:, :, :-1], (0, 0, 1, 0))
    prediction = _sum_fla(keys, prior_keys, prior_values)
    residual = values - prediction
    packed = _sum_fla(query, keys, torch.cat((values, residual), dim=-1))
    value_dim = values.shape[-1]
    return packed[..., :value_dim], packed[..., value_dim:]


def _delta_fla(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    try:
        from fla.ops.delta_rule import fused_recurrent_delta_rule
    except Exception as error:  # pragma: no cover - optional dependency
        raise _fla_unavailable(error)
    output, _ = fused_recurrent_delta_rule(
        query.transpose(1, 2),
        keys.transpose(1, 2),
        values.transpose(1, 2),
        beta.transpose(1, 2),
        scale=1.0,
    )
    return output.transpose(1, 2)
