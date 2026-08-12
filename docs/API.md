# API reference

Names listed in `thetamem.__all__` are the supported top-level API.

```python
from thetamem import (
    __version__, Lift,
    branch, key, key_part, normalize,
    hadamard, outer, concat, state_lift,
    ThetaMemory, ThetaMemLayer,
    STATE_KINDS, UPDATES, BACKENDS, VALUE_CENTERS,
    REPLAY_SOLVERS, replay_fit, replay_read,
    data,
)
```

## Lift algebra

A lift is an immutable expression describing how a projected query or key
becomes the feature that addresses the memory. Parameters are created when a
`ThetaMemory` materializes the expression. The same learned branch weights are
used for queries and keys.

### `key()`

Returns the complete projected key without adding parameters. Its width is
`key_dim`.

### `key_part(index, groups=2)`

Returns equal contiguous part `index` of the projected key without adding
parameters. `index` is zero-based and `key_dim` must be divisible by `groups`.

```python
left = key_part(0, 2)
right = key_part(1, 2)
```

### `branch(width=None, *, source=None)`

Creates a learned per-head projection. `width=None` resolves to
`feature_width`, whose default is `key_dim`. Every call creates independent
weights.

By default the input is the complete key. `source` may select `key()`,
`key_part()`, or local normalization around either:

```python
branch(32)  # weights [heads, 32, key_dim]
branch(16, source=key_part(0, 2))
branch(16, source=normalize(key_part(1, 2), "center", "l2"))
```

### `normalize(source, *ops)`

Applies local operations from left to right before an enclosing Hadamard,
concatenation, or outer product. Supported operations are:

- `"center"`: subtract the feature-axis mean;
- `"l1"`: divide by the feature-axis L1 norm;
- `"l2"`: divide by the feature-axis L2 norm;
- `"rms"`: divide by the feature-axis RMS;
- `"none"`: explicit no-op.

Zero denominators are clamped by the module's `eps`. Omitting `normalize`
leaves a block raw. It can wrap any flat lift expression but not `outer`.

```python
processed = normalize(key_part(0, 2), "center", "l2")
```

This local node is different from `ThetaMemory(feature_norm=...)`:
`normalize()` runs exactly where it appears in the expression, while
`feature_norm` is a whole-lift shorthand applied after the complete flat lift,
or independently after each top-level outer factor. `normalize()` is the
general form and supports `"center"` and `"l1"` as well; prefer it, and read
`feature_norm` as a convenience for the common case.

### `hadamard(*factors)`

Elementwise product of two or more flat, equal-width lift blocks. For example,

```python
hadamard(branch(32), branch(32))
```

implements the signed multiplicative feature
`phi(x) = (W_a x) * (W_b x)`.

Independent preprocessing of key parts is explicit:

```python
hadamard(
    normalize(key_part(0, 2), "center", "l2"),
    normalize(key_part(1, 2), "l1"),
)
```

### `concat(*blocks)`

Direct sum of two or more flat blocks. Its width is the sum of the block
widths. An `outer` cannot be nested inside a concatenation.

### `outer(*factors)`

Tensor product of two or more flat factors. Every factor becomes one state
axis; nested `outer` expressions are not allowed.

```python
# Signed 16 x 16 address from two halves of a 32-wide key.
split_outer = outer(key_part(0, 2), key_part(1, 2))
```

The flat feature width and carried-state work are the product of all factor
widths. The chunked sum/second-pass executors keep factors separate for the
within-chunk score and avoid a full product-feature tensor there, but exact
carried-state updates and reads still touch the product width. Delta and FLA
materialize the flat Kronecker feature.

Allocated width is not automatically functional rank. With two bias-free
linear branches of the same `d`-wide key,
`outer(branch(Fa), branch(Fb))` has `Fa*Fb` physical rows but rank at most
`min(Fa*Fb, d*(d+1)/2)`. The default `32 -> 32x32` outer therefore allocates
1,024 rows but spans at most 528 quadratic functions. A disjoint
`64 -> 32+32 -> 32x32` split can generically span all 1,024. The generic DSL
accepts more than two factors, but higher-degree constructions are research
expressions rather than benchmarked v0.1 presets.

### `state_lift(state, feature_width=None)`

The three named presets, each a shorthand for the lift expression beside it:

| name | lift | per-head state |
|---|---|---|
| `"hadamard"` | `hadamard(branch(F), branch(F))` | `F x V` |
| `"concat"` | `concat(key(), hadamard(branch(F), branch(F)))` | `(Dk + F) x V` |
| `"outer"` | `outer(branch(F), branch(F))` | `F x F x V` |

## `ThetaMemory`

```python
ThetaMemory(
    key_dim, value_dim, heads, *,
    state="hadamard",        # preset; ignored when lift is supplied
    lift=None,               # custom Lift expression
    feature_width=None,      # default branch width; None -> key_dim
    update="sum",            # "sum" | "second_pass" | "multi_pass" | "delta"
    passes=2,                # correction passes for update="multi_pass"
    value_center="none",     # "none" | "running_mean" | "exact_mean"
    backend="chunked",       # "naive" | "chunked" | "fla"
    chunk=256,
    feature_norm="none",     # "none" | "rms" | "l2"
    eps=1e-6,
)
```

`forward(query, key, value)` accepts query/key tensors of shape
`[B, H, T, key_dim]` and values of shape `[B, H, T, value_dim]`. It returns
`[B, H, T, value_dim]`. Shape widths and head count are checked explicitly.
Reads have no accumulated-mass denominator. Signed coordinates do not by
themselves guarantee a centered kernel; normalization and centering remain
explicit experimental choices.

Updates:

- `"sum"`: pure additive memory;
- `"second_pass"`: additive base plus a correction state written from
  strict-prefix residuals. A learned per-head mixture starts at 0.9 toward the
  base read;
- `"multi_pass"`: `passes=P` repeats the causal correction. With
  `A_c = I + tril(Phi(K)Phi(K)^T,-1)`, it starts at `W_0=V` and applies
  `W_{p+1}=W_p+eta_p(V-A_c W_p)`, then reads the final iterate. Every pass
  has a learned per-head `eta_p`, initialized at 0.1, and one full carried
  state. One pass at `eta=1` is exactly the raw `second_pass` correction;
  the default one-pass initialization exactly matches its 90/10 blend.
  This approximates a causal triangular system, not global least squares;
  three to five passes are the intended practical range;
- `"delta"`: sequential raw delta rule with learned per-head
  `beta = sigmoid(strength_logit)`. It does not divide by `||Phi||^2`;
  normalization is used only when explicitly requested in the frontend or
  lift. The initial strength is conservatively set to
  `1 / (feature_width + 1)` and remains learned independently per head.

Value centering (`value_center`, orthogonal to the update). Both options
remove the value mean and both are subtractive — nothing is divided. The
qualifier says how the mean is obtained:

- `"none"`: values are written raw;
- `"running_mean"`: values are written centered by their causal running mean,
  and the current mean is added back after the read. Cheap; the earliest
  writes are centered by a poor mean estimate;
- `"exact_mean"`: exact removal of the value mean. A ones channel rides along
  the values, so the same scan also accumulates the **signed key mass**, and
  the read returns `read(v) - mean * read(1) + mean`. Because every update is
  linear in its written values, this equals centering all values by the
  *final* running mean, retroactively, for every update and backend. Centered
  reads commute with a constant value shift; raw reads do not.

The signed key mass is read and **subtracted** here. It is not a denominator:
positive-feature designs must divide by their accumulated mass, and a signed
mass can pass through zero, so dividing by it would be unsafe.

Properties:

- `state_shape`: per-head state shape, value axis last;
- `state_size`: state floats per head - all full causal-iterate/correction
  states, the ones channel, and the running value sum included;
- `feature_width`: flattened feature width.

## Offline replay fitting

`replay_fit` and `replay_read` are lower-level APIs over already lifted
factors. They fit and read one **completed** record set:

```python
from thetamem import ThetaMemory, replay_fit, replay_read

memory = ThetaMemory(32, 64, 4, state="outer")
k_factors = memory.lift(keys)
q_factors = memory.lift(queries)

state = replay_fit(k_factors, values, solver="cg", iterations=5)
reads = replay_read(q_factors, state)
```

```python
replay_fit(
    k_factors, values, *,
    solver="cg",           # "cg" | "richardson" | "heavy_ball" | "delta"
    iterations=8,
    strength=None,
    momentum=None,
    tolerance=1e-6,
)
```

`tolerance` is the relative residual stopping threshold for `solver="cg"`;
the fixed-iteration Richardson, heavy-ball, and delta paths do not use it.

The returned state has shape `[B,H,F1,...,Fm,V]`. Outer factors keep their
separate axes and no `T x T` Gram is built. **Peak memory is nevertheless
proportional to the context length**: the write and read contractions sum over
the token axis, and no pairwise path does that without an intermediate that
still carries `T`, so the einsum materializes a temporary of order
`T * prod(F_i)` or `T * F1 * V` — at `T=512` with `32 x 32` factors, a
`512 x 1024` temporary per call. A long context can therefore exhaust device
memory here, unlike the causal chunked path, which tiles the token axis. Every
iteration also replays the complete keys/values, so this is an offline/prefill
solver rather than a streaming `forward`.

- `"cg"` is the default CGLS/CGNR solver and chooses per-value step scalars
  from global reductions.
- `"richardson"` requires an explicit positive `strength`; stability requires
  an appropriate spectral bound.
- `"heavy_ball"` additionally requires `momentum` in `[0,1)`.
- `"delta"` performs cyclic sequential sweeps; `strength` defaults to 1.
  Sweeps after the first revisit the whole record set. Unit strength has the
  standard projection interpretation only for unit-norm lifted rows; otherwise
  choose a stable explicit strength.

Two iterations are the smallest useful replay-correction experiment; three to
five are the practical starting range. Richardson, heavy-ball, and CG are
finite approximations to the global fitted memory. Replayed delta is instead a
cyclic projection method: it can approach interpolation for a consistent,
properly scaled system, but need not return the least-squares solution when the
records are inconsistent. Convergence depends on rank, conditioning,
parameters, and numerical precision. See
[Algorithms](ALGORITHMS.md#6-global-replay-solvers) and the deterministic
[`offline_replay_solvers.py`](../examples/capacity/offline_replay_solvers.py).

## `ThetaMemLayer`

```python
ThetaMemLayer(
    d_model=128, *,
    heads=4,
    key_dim=32,
    value_dim=64,
    state=None,
    lift=None,
    feature_width=None,
    update="sum",
    passes=2,
    value_center="none",
    backend="chunked",
    chunk=256,
    feature_norm="none",
    qk_projection="linear",  # "linear" | "direct_hadamard"
    qk_ops=("conv", "silu", "rope", "l2"),
    value_ops=("conv", "silu"),
    position="rope",         # master switch: "none" disables every "rope" op
    rope_fraction=0.5,
    rope_base=10_000.0,
    conv_width=4,
    eps=1e-6,
)
```

`forward(hidden_states)` maps `[B, T, d_model]` to the same shape. The output
shell remains per-head RMSNorm, a low-rank SiLU gate, and a bias-free output
projection.

### Q/K projection modes

`"linear"` projects each token to one `key_dim` vector per head. When neither
`state` nor `lift` is supplied, it retains the canonical learned Hadamard lift.

`"direct_hadamard"` uses one fused Q projection and one fused K projection.
Each produces two `key_dim` branches per head directly from the token; the two
branches are multiplied immediately. Unless a custom `lift` is supplied, the
result goes to the SSM through `key()`, so there is no intermediate projected
key and no additional lift matrix. With `state="hadamard"` the direct product
*is* the lift; any other preset must be expressed explicitly as a custom
`lift`. The direct-product width is `key_dim`.

### Ordered frontend operations

`qk_ops` execute from left to right and support `"conv"`, `"silu"`, `"rope"`,
`"center"`, `"l1"`, `"l2"`, and `"rms"`. `value_ops` support the same list
except RoPE. Operations absent from a tuple are not run and their convolution
modules are not created.

The defaults are the frontend used by every recorded run:

```text
Q/K: projection -> causal Conv -> SiLU -> RoPE -> L2 -> memory
V:   projection -> causal Conv -> SiLU -> memory
```

Examples:

```python
# Projected Q/K go directly to the chosen lift; V keeps its tested frontend.
raw_qk = ThetaMemLayer(
    qk_ops=(),
    value_ops=("conv", "silu"),
)

# No Q/K/V SiLU and no Q/K normalization.
no_activation_or_qk_norm = ThetaMemLayer(
    qk_ops=("conv", "rope"),
    value_ops=("conv",),
)

# Direct token-to-two-branches Hadamard, then the selected operations and SSM.
direct = ThetaMemLayer(
    qk_projection="direct_hadamard",
    qk_ops=("conv", "rope", "l2"),
    value_ops=("conv", "silu"),
)

# Direct Hadamard Q/K go immediately to the SSM. V retains Conv -> SiLU.
direct_raw_qk = ThetaMemLayer(
    qk_projection="direct_hadamard",
    qk_ops=(),
)

# Remove the V frontend as well when that is the intended ablation.
fully_raw = ThetaMemLayer(
    qk_projection="direct_hadamard",
    qk_ops=(),
    value_ops=(),
)
```

`position="none"` makes every `"rope"` operation a no-op for compatibility
with previous configurations. Otherwise its exact location is determined by
`qk_ops`. `state_size` includes only the convolution caches whose `"conv"`
operation is active.

## Backends

- `"naive"`: masked quadratic semantic reference, O(T^2) time and memory.
- `"chunked"`: chunk-local score tiles plus an explicit inter-chunk state.
  Carried-state `K^T V` updates and `Q S` reads accumulate in at least FP32;
  within-chunk score tiles remain in the input dtype.
- `"fla"`: optional delegation to `flash-linear-attention`. It is never
  selected implicitly and fails closed if its dependency or kernel is
  unavailable.

For outer lifts, flattening computes the exact Kronecker feature, not an
approximation: its dot product equals the product of the factor dot products.
The tradeoff is temporary memory proportional to the product width.

## Data generators

### `thetamem.data.mqar`

- `generate(num_examples, seq_len, kv_pairs, *, vocab_size=8192,
  power_a=0.01, seed=0, random_fillers=True)` creates one MQAR segment
  (`inputs`, `labels`, with `-100` off answer positions).
- `mixture(segments, *, vocab_size, seed)` creates a list of segments;
  `TRAIN_MIXTURE` and `EVAL_SLICES` contain the recorded protocol.

### `thetamem.data.mad`

- `fuzzy_recall(num_examples, *, seq_len=128, vocab_size=16, k_motif=3,
  v_motif=3, train=True, seed=0)` creates fuzzy recall.
- `selective_copy(num_examples, *, seq_len=256, vocab_size=16,
  tokens_to_copy=16, seed=0)` creates selective copying.
- `generate(task, num_examples, *, train=True, seed=0, **overrides)`
  dispatches by `"fuzzy"` or `"selective"`.

All data draws use a local seeded NumPy generator.
