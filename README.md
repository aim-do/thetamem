# ΘetaMem

**A PyTorch token mixer with a signed, multiplicative key: linear time, a
bounded state, and no erasure.**

*In search of the most efficient state: signed tensor-product keys for
linear-time sequence models.*

## Abstract

> Fixed-state token mixers process a sequence in linear time through a state
> of constant size, and a constant state forgets: every write spends the same
> shared feature directions, so records blur into one another long before the
> query that needs them arrives. ΘetaMem fights that interference in the key
> instead of with erasure. It lifts the projected key with a **signed
> quadratic Hadamard product**; in its simplest form the key is split into
> halves `a` and `b` and addresses the memory through
> `[a ⊙ b, a ⊙ roll(b, 1)]`. The lift adds no parameters and keeps the width
> of the key, so the state is exactly as large as a linear key's — but the
> similarity of two tokens becomes a signed degree-two polynomial instead of a
> dot product, which separates nearby keys more sharply and lets the
> cross-talk of unrelated records cancel instead of piling up. Writes are
> purely additive: no decay, no erasure, no gates. The outer product of two
> learned branches spends the same multiplication on a larger tensor state
> instead.
>
> At a state matched to Gated DeltaNet-2's 8,192 floats, with 4.5% fewer
> parameters, the additive quadratic memory scores **0.434** on multi-query
> associative recall over 1,024–4,096-token contexts against the baseline's
> **0.195** (2.2×), and **0.273** against **0.087** at 4,096 tokens (3.1×);
> the same memory with a linear key scores 0.140, so the gain is the lift's. A
> purely additive state has no clock and cannot copy selectively on its own
> (0.186 against 0.919 on MAD); with learned Fourier position features in
> front of the lift — a laboratory frontend — the additive memory passes the
> baseline on all three tasks at once: recall 0.249 against 0.195, fuzzy
> recall 0.344 against 0.272, selective copying 0.995 against 0.919. At a 16×
> smaller state of 512 floats the ordering holds (0.344 / 0.306 / 0.990
> against 0.141 / 0.148 / 0.828). Every number is a single-seed, best-of-grid
> synthetic result; together they motivate the quadratic signed key, not yet a
> language-model claim.

— quoted from the
[paper](paper/ThetaMem-Signed-Multiplicative-Lifts.pdf) (citation markers and
math typesetting aside); every number above is tabulated, with what it is
matched on and what it is not, in
[the snapshot and its caveats](#current-research-snapshot).

## What it is

ΘetaMem replaces the linear key of a fixed-state memory with a product of two
projections of it. In its parameter-free form — the arm behind the v0.2
results — each projected key is split into halves `a` and `b` and addresses
the state through `[a ⊙ b, a ⊙ roll(b, 1)]`: the feature has the width of the
key, so the state does not grow, but every coordinate is a product of one
coordinate from each half, so the similarity of two tokens is a signed
degree-two polynomial rather than a dot product. In its learned form the two
factors are independently learned branches, and taking their **outer**
product instead of the elementwise one turns them into two axes of a tensor
state — a much larger memory bought without widening a single projection
around it. Writes are additive; nothing is erased.

**Public Preview v0.2 · 2026-10-22.** The implementation and reasoning are
public; the benchmark study is in progress — multi-seed replication, a
position signal for the additive state, benchmarking the corrected and
centered memories, fused tensor-state kernels, and language-model-scale
validation are open.

[Paper: *ΘetaMem: Signed Multiplicative Lifts for Fixed-State Sequence Memory* (PDF)](paper/ThetaMem-Signed-Multiplicative-Lifts.pdf) ·
[Paper source and versions](paper/README.md) ·
[API reference](docs/API.md) ·
[Algorithms](docs/ALGORITHMS.md) ·
[Reproduce experiments](docs/EXPERIMENTS.md) ·
[Capacity simulations](examples/capacity/README.md) ·
[Release notes](CHANGELOG.md)

## News

- **2026-10-22 — Public Preview v0.2: a parameter-free quadratic address beats
  Gated DeltaNet-2 at matched state.**

  *Library.* A `roll()` lift node and the `state="shifted_hadamard"` preset —
  the address of the new headline arm, which the library now rebuilds
  exactly; `thetamem.data.mqar.LONG_EVAL_SLICES`; example flags `--qk-ops`,
  `--position`, and `--eval long`.

  *Results.* On one H100 at the baseline's 8,192 state floats and with 4.5%
  fewer parameters, the additive shifted-split memory scores **0.434** on
  long-context MQAR (1,024–4,096 tokens) against Gated DeltaNet-2's
  **0.195**; a linear key in the same shell scores 0.140. With no position
  signal it cannot copy selectively (0.186 vs 0.919); a laboratory
  learned-Fourier position frontend makes the additive memory beat the
  baseline on MQAR, fuzzy recall, and selective copying at once (0.249 /
  0.344 / 0.995 vs 0.195 / 0.272 / 0.919). The ordering holds at 512 state
  floats on Kaggle T4s. Single seed throughout.

  *Paper.* Public Preview v0.2: a two-paragraph abstract; the matched-state
  suite is Section 7.5 and the small-state league Section 7.6.

- **2026-08-12 — Public Preview v0.1: first public release.**

  *Library.* The lift algebra (branches, key parts, local normalization,
  Hadamard, direct sum, outer product); signed no-mass reads with value
  centering (`running_mean` and the exact `exact_mean`); updates `sum`,
  `second_pass`, experimental causal `multi_pass`, and a sequential `delta`
  reference; offline replay solvers (Richardson, heavy-ball, CG, cyclic
  delta); naive/chunked/FLA executors; MQAR and MAD generators with minimal
  training harnesses; three deterministic capacity simulations.

  *Results.* At matched state the additive Hadamard memory beats Gated
  DeltaNet-2 on MQAR extrapolation (0.668 vs 0.567) with no erasure, and the
  tensor state holds that slice at 0.976. Trained at equal key width and equal
  parameters, a signed outer state holds a 4,096-token slice at 0.742 where an
  exact grouped PSD self-product collapses to 0.024. On MAD fuzzy recall the
  constant-state variant **loses** to the baseline (0.181 vs 0.323); only the
  tensor state passes it (0.714), and the baseline's wider-key control reaches
  0.596. Single seed throughout.

Each future snapshot adds a dated entry here; details land in
[Release notes](CHANGELOG.md).

## Why ΘetaMem?

Every bounded associative memory returns, for a query, the record it stored
plus a weighted sum of every other record. The established families each fix
one property of that budget and pay with another. Plain linear attention
mixes nearby keys, cannot separate them with any linear map, and its state is
chained to the physical key width. Delta-style memories (DeltaNet, Gated
DeltaNet-2) write residuals and forget through decay and erasure, which keeps
a crowded state usable at the price of the records they remove; a single
causal delta pass is not a least-squares solver, and corrective writes still
share the available feature directions. Positive-diagonal self-product
memories take another route: grouped self-outers make the query/key score a
nonnegative sum of squared similarities and give the additive read a positive
mass mode.

ΘetaMem changes the weights themselves. A product key makes the read weight a
degree-two function of the key, so it separates near-duplicate keys that a
linear key mixes; the product is **signed**, so the cross-talk of unrelated
records can cancel instead of accumulating with one sign; and it does this
either at constant state (`shifted_hadamard`, `hadamard`) or with a
deliberately larger one (`outer`). The cancellation is conditional: it needs
decorrelated factors and centered unrelated values, and adversarially
sign-aligned values remove it. Separate maps from the same source are not
statistically independent, and a larger physical outer state need not have
equally large functional rank. The
[paper](paper/ThetaMem-Signed-Multiplicative-Lifts.pdf) develops the argument
and reports, claim by claim, what currently supports it.

The **product mechanisms** share one scan:

| state | lift | what it buys |
|---|---|---|
| `"shifted_hadamard"` | each coordinate of one key half times a coordinate of the other, twice, with a one-step circular shift | a signed quadratic similarity at exactly a linear key's state, with no lift parameters |
| `"hadamard"` | corresponding coordinates of two learned branches are multiplied | a product similarity at constant state, with learned branch geometry |
| `"outer"` | every coordinate of one learned branch pairs with every coordinate of the other | a much larger tensor state and a product similarity |

The learned presets build both branches from one projected key. The lift
algebra also supports disjoint key parts. Those choices allocate the same
kind of tensor array but do not have the same functional rank: separate
matrices are not the same thing as independent information. The
[`rank_geometry.py`](examples/capacity/rank_geometry.py) surrogate isolates
that distinction; neither construction promises that training will occupy its
algebraic upper bound.

Beyond the presets, a small **lift algebra** composes branches, the raw key
and its parts, circular shifts, elementwise products, direct sums, and outer
products into custom state layouts, with validation rules that keep every
instance executable by the same scan machinery.

Factorization keeps projection and chunk-local score formation compact, but an
exact read and write still touches the complete tensor state. For a fixed
state the sequence cost remains linear in context length; growing the state
until it rivals the context eventually gives back the same broad cost pressure
as attention. Exact accounting is in [Algorithms](docs/ALGORITHMS.md).

`feature_norm="center"` removes the coordinate mean from the **completed** key
feature. For an outer lift this is global centering of the full tensor product,
not separate centering of its factors. The chunked `sum`, `second_pass`, and
`multi_pass` paths apply the exact rank-one correction lazily, preserving the
factorized local score and the original tensor-state shape and size. The
`delta` and optional `fla` paths flatten outer features as they already do.

The default `update="sum"` read has **no accumulated-mass denominator**.
Optional `value_center="running_mean"` subtracts a causal running mean;
`value_center="exact_mean"` removes the mean exactly through a signed key-mass
read — subtracted, never used as a divisor. `update="second_pass"` stores measured strict-prefix
corrections in a second state. `update="multi_pass", passes=P` repeats the
correction as Richardson steps for a **causal triangular** system and reads
the final iterate; one unit-strength pass is exactly the raw second pass.
This is not global least squares. `update="delta"` is the one-sweep causal
delta reference. The separate `replay_fit` API provides global
Richardson/heavy-ball/CG and repeated delta, but every iteration rereads the
completed context. A second sweep is the smallest meaningful replay
correction; two to five passes are the practical preliminary range at moderate
load, while CG is the better candidate near the fitted-state ceiling.

There are two independent value-centering controls. A final
`value_ops=(..., "center")` centers every per-token value vector across its
coordinates after the preceding ordered frontend transforms.
`value_center="running_mean"|"exact_mean"` instead centers each channel across
causal records inside the memory algorithm. They act on different axes.

## Current research snapshot

### Matched-state suite (one H100, 2026-08)

Single seed, best over the learning-rate grid {3.16e-4, 1e-3, 3.16e-3}; two-layer
models at width 128 (gated short-convolution mixer, then the memory layer);
every memory at 4 heads, key width 32, value width 64, and exactly **8,192 core
state floats**. ΘetaMem uses a raw Q/K path (projection → causal convolution →
lift) and writes additively; Gated DeltaNet-2 runs the official
flash-linear-attention Triton kernels at a pinned revision. MQAR trains on
lengths 64-256 (vocabulary 8,192) and is scored as the mean over three long
slices; MAD runs at its standard settings. Every cell and the full protocol:
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#matched-state-suite-2026-08).

| memory | params | MQAR score | 1,024 / 256 | 2,048 / 512 | 4,096 / 512 | fuzzy | selective |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gated DeltaNet-2 | 1,453,764 | 0.195 | 0.400 | 0.097 | 0.087 | 0.272 | 0.919 |
| ΘetaMem, linear key (`lift=key()`) | 1,388,288 | 0.140 | 0.279 | 0.074 | 0.068 | 0.384 | 0.185 |
| **ΘetaMem `shifted_hadamard`** | 1,388,288 | **0.434** | **0.737** | **0.292** | **0.273** | 0.235 | 0.186 |
| ΘetaMem `shifted_hadamard` + learned Fourier positions † | 1,398,536 | 0.249 | 0.332 | 0.220 | 0.197 | 0.344 | 0.995 |

Parameters are those of the MQAR models; on MAD they are 407,236 for the
baseline and 341,760 for the ΘetaMem shells. † Laboratory frontend, not
shipped: learned Fourier position features added to the Q/K input, then
convolution → SiLU → L2 → identity-initialized per-head 32 × 32 maps before the
same shifted-split lift.

### Small states (Kaggle T4, 2026-08)

Width 64, 2 heads, key width 16, value width 16 — **512 core state floats** for
every memory; FP32; MQAR at vocabulary 2,048, scored as the mean over five
slices from 64 tokens / 16 pairs to 1,024 / 256; best over five (MQAR) or three
(MAD) learning rates; one seed. The baseline is a PyTorch reference
implementation of the Gated DeltaNet-2 recurrence.

| memory | params | MQAR | fuzzy | selective |
|---|---:|---:|---:|---:|
| Gated DeltaNet-2 | 200,786 | 0.141 | 0.148 | 0.828 |
| ΘetaMem shifted-split, raw Q/K | 194,880 | 0.305 | 0.318 | 0.199 |
| ΘetaMem shifted-split + learned Fourier positions † | 196,936 | 0.344 | 0.306 | 0.990 |

† The laboratory frontend above, without the Q/K convolution and with 16 × 16
maps.

### How to read the snapshot

**The lift is the gain.** Same shell, same state, same parameter count: a
linear key scores 0.140 on MQAR and the parameter-free quadratic address
0.434 — 2.2× the erasing baseline on the mean and 3.1× at 4,096 tokens (0.273
vs 0.087), with no decay, erasure, gate, or correction. The lift has no
parameters, so the comparison isolates geometry: the extra separability is
bought with degree, not with parameters or state.

**Where we lose.** An additive state has no clock. On selective copying the
quadratic memory stays at 0.186 against the baseline's 0.919, because nothing
in an additive state records when a record arrived; on fuzzy recall —
overlapping, rewritten motifs, where erasure genuinely helps — it trails 0.235
to 0.272. Learned position features fix copying and pass the baseline on all
three tasks at once, but give back part of the recall gain (0.249), and that
frontend is not shipped yet. Deletion replays and a learned delta rule,
reported in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#matched-state-suite-2026-08),
trade recall for order in the same way.

**What is matched and what is not.** Core state floats and key width are
matched exactly, and the ΘetaMem shells carry fewer trainable parameters than
the baseline (4.5% fewer on MQAR, 16% fewer on MAD). Every number is
single-seed and the best of a three- to five-point learning-rate grid; the
ΘetaMem selective-copying optima sit at the grid's edge, while both headline
MQAR cells peak at the central rate. MQAR's long slices draw fillers from the
full vocabulary, so they mix retention with collision distractors; every arm
saw identical tensors. ΘetaMem compiled end to end while the baseline ran its
fused kernels eagerly, so the step-time observation (12.3 ms vs 48.0 ms median
per MQAR step) crosses executors.

**What this repository rebuilds.** The `shifted_hadamard` row is
`ThetaMemLayer(128, heads=4, key_dim=32, value_dim=64, state="shifted_hadamard",
qk_ops=("conv",), position="none", eps=1e-8)`, and the linear-key row is the
same call with `lift=thetamem.key()`; both have the recorded arms' parameter
sets and initialization from the same seed and return identical outputs and
gradients. The baseline and the † frontends come from the project's laboratory
harness and are not built here.

<details>
<summary><b>Earlier snapshot (v0.1, 2026-08-12)</b> — learned Hadamard, concatenated, and outer lifts; signed versus PSD self-product</summary>

Single seed, best over the stated learning-rate grids, one H100; two-layer
models (gated short-convolution mixer, then the memory layer); all memories
at projected key width 32 x 4 heads. Key params count the trained maps that
produce the key features per memory layer (token-to-key projection plus lift
branches, shared with the query side). The study is in progress; full
protocol in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

**MQAR** (vocabulary 8,192; trained on lengths 64-256; hardest slices shown):

| memory | core state (floats) | key params | len 256 / kv 64 | len 1,024 / kv 256 |
|---|---:|---:|---:|---:|
| Gated DeltaNet-2 | 8,192 | 16,384 | 0.998 | 0.567 |
| Gated DeltaNet-2, physical key 48 | 12,288 | 24,576 | 1.000 | 0.814 |
| **ΘetaMem hadamard** | 8,192 | 24,576 | 1.000 | 0.668 |
| ΘetaMem hadamard + second_pass | 16,384 | 24,576 | 1.000 | 0.662 |
| ΘetaMem concat | 16,384 | 24,576 | 1.000 | 0.781 |
| **ΘetaMem outer** | 262,144 | 24,576 | 1.000 | 0.976 |

**MAD** (fuzzy in-context recall and selective copying):

| memory | core state (floats) | key params | fuzzy | selective |
|---|---:|---:|---:|---:|
| Gated DeltaNet-2 | 8,192 | 16,384 | 0.323 | 0.869 |
| Gated DeltaNet-2, physical key 48 | 12,288 | 24,576 | 0.596 | 0.914 |
| **ΘetaMem hadamard** | 8,192 | 24,576 | 0.181 | 0.988 |
| ΘetaMem hadamard + second_pass | 16,384 | 24,576 | 0.264 | 0.982 |
| ΘetaMem concat | 16,384 | 24,576 | 0.398 | 0.983 |
| **ΘetaMem outer** | 262,144 | 24,576 | 0.714 | 0.982 |

**Signed versus a positive geometry, trained at equal key width and equal
parameters** (1,388,288 parameters per arm, bit-identical data, one seed, one
learning-rate cell; evaluated frozen at four lengths):

| memory | core state (floats) | 1,024 / 256 | 2,048 / 512 | 4,096 / 1,024 |
|---|---:|---:|---:|---:|
| grouped PSD self-product + positive mass | 35,360 | 0.983781 | 0.484236 | 0.024242 |
| **signed outer of the two key halves** | 65,536 | 0.999961 | **0.986990** | **0.742194** |
| signed outer, per-half L1/L2 normalization | 65,536 | 0.999945 | 0.981914 | 0.699979 |

**How to read the v0.1 snapshot.**

**Where we win and where we lose.** At matched state the signed additive
memory beats the erasing baseline on MQAR extrapolation with no correction
and no erasure (0.668 vs 0.567). Given a wider physical key that same
baseline reaches 0.814, ahead of every ΘetaMem arm except the tensor state —
it is the comparator to quote, not the narrow one. On MAD fuzzy recall the
constant-state variant **loses outright** (0.181 vs 0.323): overlapping,
rewritten motifs are the regime where erasure genuinely helps. Only the
tensor state passes both baseline configurations there.

**What is matched and what is not.** The first two tables match physical key
width and (for the Hadamard lift) core state floats, **not** trainable
parameters — the theta shell carries roughly 11% more on MQAR. The third
table is the reverse: parameters and key width match exactly, while the
signed outer carries 1.853× the comparator's core state. Its comparator is
our own independent implementation of the published equations, not its
authors' code, and it is not shipped here. Its long slices draw fillers from
the full vocabulary, so they mix retention with collision distractors.

**What these numbers do not cover.** One seed everywhere. The multi-pass and
value-centering arms are implemented and tested but not benchmarked. The
theta arms compiled end to end while the baseline ran its own fused kernels
eagerly, so the step-time observation (theta 19-24 ms vs. 29-33 ms per step
on the MAD runs) crosses executors. The baseline and every archived control
were run in the project's benchmark harness; none of them is built by the
code in this repository, which ships the ΘetaMem layers and the data
generators. The examples below re-run the protocol, not the bit pattern.

</details>

## Install and first forward pass

```bash
pip install torch numpy
pip install -e .
```

Start here. Every argument below has a default, and the defaults are the v0.1
recorded configuration: the learned Hadamard lift, a pure additive `sum`, no
value centering, and the chunked executor.

```python
import torch
import thetamem

layer = thetamem.ThetaMemLayer(128, heads=4, key_dim=32, value_dim=64)
y = layer(torch.randn(2, 1024, 128))   # [batch, time, d_model]

layer.state_size    # 9,728 streaming floats per layer
```

The v0.2 headline arm is the parameter-free shifted-split address with the raw
Q/K frontend it was measured with — the same state, no lift parameters:

```python
layer = thetamem.ThetaMemLayer(
    128, heads=4, key_dim=32, value_dim=64,
    state="shifted_hadamard",   # [a*b, a*roll(b, 1)] of the key halves a, b
    qk_ops=("conv",),           # Q/K: projection -> causal conv -> lift
    position="none",
)
layer.state_size    # 9,728 — exactly what a linear key would use
```

It has no clock of its own: on selective copying it needs a position signal
(see the snapshot). Change one thing at a time. `state="outer"` is the tensor
memory of the v0.1 tables — the strongest arm measured there, at 27× the
streaming state (263,680 floats per layer), so it is a deliberate choice
rather than a default:

```python
layer = thetamem.ThetaMemLayer(
    128, heads=4, key_dim=32, value_dim=64,
    state="outer",              # "hadamard" | "concat" | "outer" | "shifted_hadamard"
    update="second_pass",       # "sum" | "second_pass" | "multi_pass" | "delta"
    feature_norm="center",      # final whole-feature coordinate centering
    value_ops=("conv", "silu", "center"),  # center V after its frontend
    value_center="exact_mean",  # "none" | "running_mean" | "exact_mean"
    backend="chunked",          # "naive" | "chunked" | "fla"
    chunk=256,
)
```

The memory alone (no projections, no gates) is `thetamem.ThetaMemory`. The
four presets are shorthands for lift expressions — `state="outer"` is
`outer(branch(32), branch(32))`, and `state="shifted_hadamard"` is
`concat(hadamard(a, b), hadamard(a, roll(b, 1)))` with
`a, b = key_part(0, 2), key_part(1, 2)` — and the algebra builds the rest:

```python
import torch
from thetamem import ThetaMemory, branch, key_part, normalize, outer

memory = ThetaMemory(
    key_dim=64, value_dim=64, heads=4,
    lift=outer(
        normalize(key_part(0, 2), "l2"),   # disjoint halves of the key:
        normalize(key_part(1, 2), "l2"),   # no lift parameters at all
    ),
)

q = k = torch.randn(2, 4, 128, 64)   # [batch, heads, time, key_dim]
v = torch.randn(2, 4, 128, 64)       # [batch, heads, time, value_dim]
reads = memory(q, k, v)              # [2, 4, 128, 64]
```

## Embedding in your stack

The installed package is self-contained and import-light (torch and numpy).
Three integration points:

- **As a token mixer**: `ThetaMemLayer` maps `[B, T, d_model]` to the same
  shape — drop it in wherever an attention or SSM block sits, residual and
  norm outside. `layer.state_size` reports the streaming state floats per
  layer (memory states, the ones channel, and convolution caches included) for
  memory budgeting.
- **As the memory primitive**: `ThetaMemory` consumes already-projected
  `[B, H, T, *]` queries/keys/values and returns reads — use it inside your
  own shell if you have your own projections, gating, or normalization.
- **As an offline fitted memory**: `replay_fit` and `replay_read` apply
  Richardson, heavy-ball, CG, or repeated delta to a completed set of lifted
  keys. The state keeps its outer factor axes and no `T x T` Gram is built,
  but each iteration replays the context and its contractions materialize a
  temporary proportional to the context length, so this path is memory-bounded
  by `T` and is a prefill tool, not a long-context one.

Everything is standard PyTorch and compiles end to end with
`torch.compile`; the chunked executor is portable (CPU/GPU). A streaming
decode path (stateful step-by-step inference) is not shipped yet — see
Scope.

## Reproduce the experiments

### Library examples

Install the package first. The library ships the data generators and a
minimal harness — no benchmark framework is required. The v0.2 matched-state
arm, with the long MQAR slices and their mean (`--qk-ops` and `--position`
select the raw Q/K frontend; with them the example shell has exactly the
recorded parameter counts):

```bash
python examples/mqar.py --state shifted_hadamard --qk-ops conv --position none --eval long --lr 1e-3
python examples/mad.py --task fuzzy     --state shifted_hadamard --qk-ops conv --position none --lr 1e-3
python examples/mad.py --task selective --state shifted_hadamard --qk-ops conv --position none --lr 3.16e-3
```

The v0.1 snapshot — MQAR with the three learned-lift state kinds:

```bash
python examples/mqar.py --state hadamard --lr 3.16e-3
python examples/mqar.py --state concat   --lr 1e-3
python examples/mqar.py --state outer    --lr 1e-3
```

MAD fuzzy recall and selective copying:

```bash
python examples/mad.py --task fuzzy     --state outer    --lr 3.16e-3
python examples/mad.py --task fuzzy     --state concat   --lr 1e-3
python examples/mad.py --task fuzzy     --state hadamard --lr 3.16e-3
python examples/mad.py --task selective --state hadamard --lr 1e-3
```

Add `--update second_pass` for one correction, or
`--update multi_pass --passes 3` to repeat it — two is the smallest
meaningful budget and three to five is the practical range.

`--smoke` runs a two-epoch CPU pass to check that the wiring holds. **It is
not expected to learn anything**: it prints an accuracy near `0.0000`, which
at that budget is the correct result and not a failure. A healthy smoke run
is one that completes, prints the constructor call it built, and reports the
state floats. The numbers above need the full protocol and a GPU.

### NumPy surrogate experiments

These are standalone, deterministic checks of selected theoretical claims.
They do not train a model and do not add another architecture to the library:

```bash
python examples/capacity/rank_geometry.py
python examples/capacity/psd_self_product_capacity.py
python examples/capacity/offline_replay_solvers.py
```

[`rank_geometry.py`](examples/capacity/rank_geometry.py) separates physical
cells from functional rank;
[`psd_self_product_capacity.py`](examples/capacity/psd_self_product_capacity.py)
implements an exact grouped PSD self-product and an ideal independent signed
comparator; and
[`offline_replay_solvers.py`](examples/capacity/offline_replay_solvers.py)
compares replayed Richardson, heavy-ball, and CG without making a claim about
the library's causal `multi_pass`. Fixed-seed outputs and assumptions are
recorded in the [capacity simulation guide](examples/capacity/README.md) and
[experiment protocol](docs/EXPERIMENTS.md#capacity-simulations).

Data generation is importable directly:

```python
from thetamem.data import mqar, mad

train_segments = mqar.mixture(mqar.TRAIN_MIXTURE, vocab_size=8192, seed=123)
eval_segments  = mqar.mixture(mqar.EVAL_SLICES,  vocab_size=8192, seed=1123)
long_segments  = mqar.mixture(mqar.LONG_EVAL_SLICES, vocab_size=8192, seed=1123)
inputs, targets = mad.generate("fuzzy", 6400, train=True, seed=123)
```

Full protocols, grids, and expected numbers: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Library versus examples

This repository deliberately contains **two different things**, and mixing
them up would misread the evidence.

The installed **PyTorch library** is the lift algebra, the memory, the layer,
the replay solvers, and the data generators. The PyTorch scripts
`examples/mqar.py` and `examples/mad.py` are deliberately small usage and
training harnesses around it: they regenerate data, train one configuration
per invocation, and print accuracy.

The **NumPy surrogate experiments** under `examples/capacity/` are a separate
track. They are small, fixed-seed checks of selected rank, interference, and
replay arguments — abstract systems, not trained architectures and not extra
library layers. They never instantiate a ΘetaMem layer, and their numbers
support the reasoning in the paper rather than the benchmark tables.

## Scope

v0.2 of the library is the additive-and-corrected lane: the learned and
parameter-free (`shifted_hadamard`) product lifts, causal sums, optional value
centering, a strict-prefix second-state correction, an experimental causal
triangular `multi_pass`, a sequential delta reference, and global offline
replay solvers. The shipped `multi_pass` has no **global** least-squares
convergence claim; `replay_fit` must retain or reconstruct all records. No
position frontend beyond RoPE — the learned Fourier position features and the
deletion replays of the 2026-08 snapshot are laboratory code — no temporal
decay banks, no write gating (an open direction the paper motivates), no
streaming decode path, and the `fla` backend delegates flat lifts to the
optional `flash-linear-attention` package and fails closed otherwise. The
chunked executor is portable PyTorch and compiles end to end.

## Paper and citation

The versioned technical paper lives in [`paper/`](paper/README.md) with its
render pipeline and version record. It uses the collective byline **The
ThetaMem Project**, allowing later versions to credit substantive research
collaborators without an anonymous author entry.

```bibtex
@article{thetamem2026,
  title   = {ThetaMem: Signed Multiplicative Lifts for Fixed-State Sequence Memory},
  author  = {{The ThetaMem Project}},
  year    = {2026},
  note    = {Versioned technical paper, Public Preview v0.2},
}
```

## Research collaboration

ΘetaMem is looking for collaborators. The project welcomes focused research
collaboration on replication and multi-seed studies, benchmarking the
corrected and centered memories, write-gating mechanisms, fused kernels for
the factorized tensor state, streaming/decode paths, and
language-model-scale evaluation. Reproducibility reports and design
discussions may be opened as issues. Contact
[hi@aim.do](mailto:hi@aim.do) before beginning a substantial code
contribution; the patent and source-available licensing context requires a
contributor agreement before code can be merged. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

ΘetaMem is source-available under the **PolyForm Small Business License
1.0.0**: free use for individuals and organizations with fewer than 100
employees and contractors and under USD 1,000,000 (2019 dollars) in prior-year
revenue; other organizations need a separate commercial license. The copyright
holder and software licensor is **Ultimamind SRL, Belgium**. Licensing
correspondence: [hi@aim.do](mailto:hi@aim.do).

ΘetaMem is patent pending: the mechanisms implemented here are the subject of
U.S. Provisional Patent Application No. **64/132,046**, *Signed
Multiplicatively Lifted Sequence Memories*. That notice makes no
representation about the scope of any claim and grants no license beyond the
one in [LICENSE](LICENSE); see [PATENTS.md](PATENTS.md).

See [LICENSE](LICENSE) for the license text, [LICENSING.md](LICENSING.md) for
a plain-language guide to the boundaries, and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party
attributions. The paper materials in `paper/` are all rights reserved and are
not covered by the software license.
