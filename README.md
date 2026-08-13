# ΘetaMem

**A PyTorch token mixer with a signed, multiplicative key: linear time, a
bounded state, and no erasure.**

*In search of the most efficient state: signed tensor-product keys for
linear-time sequence models.*

## Abstract

> Fixed-state token mixers process a sequence in linear time by writing every
> token into a state of constant size. What such a state can do is set by
> several properties at once: its **capacity** (how many records fit), its
> **noise** (the part of a read that comes from every record except the one
> asked for), and its **retention** (how long a record survives later writes).
> ΘetaMem is a memory design that improves these properties together, in
> search of the most efficient state: the most usable records per state float,
> per trained parameter, and per unit of compute.
>
> The paper is organized around four hypotheses about how to spend a fixed
> state, and it reports what each one currently rests on. **H1**: learned
> **outer products** enlarge the physical state while keeping the surrounding
> projections narrow. **H2**: **signed** decorrelated factors address that
> state better than positive geometries at a comparable key budget, because
> nonnegative read weights cannot cancel and their cross-talk accumulates
> coherently. **H3**: Gram-overlap error can be corrected **without erasing**,
> and one correction is not the natural stopping point. **H4**: signed
> cancellation is conditional on **centered values**, so centering is part of
> the mechanism rather than an option.
>
> The shipped library implements two-factor Hadamard and outer lifts,
> subtractive value centering, a strict-prefix second pass, a causal
> repeated-pass approximation, delta writes, and separate non-causal replay
> solvers. The evidence is mixed in kind, and stated as such.
>
> H1 and H2 have trained support. On multi-query associative recall, at a
> state matched to the baseline's 8,192 floats, the elementwise memory reaches
> **0.668** on a 4× length-extrapolation slice against Gated DeltaNet-2's
> **0.567**, with no erasure and no correction; given a wider physical key the
> same baseline reaches **0.814**, and only the tensor state passes it, at
> **0.976** for 32× the state floats. Against an exact grouped
> positive-semidefinite (PSD) self-product, trained at equal key width and
> equal parameters, a signed product state holds a 4,096-token slice at
> **0.742** where the PSD arm falls to **0.024**. H2 also has an idealized
> random-key surrogate that isolates geometry alone, where the signed code
> carries **2.61×** less interference power at nearly equal state. H3 has
> surrogate support and one nearly neutral trained cell (**0.662** against
> **0.668**). H4 is implemented and tested but not yet benchmarked.
>
> The losses are part of the result. On MAD fuzzy recall our constant-state
> variant reaches **0.181** against the baseline's **0.323** and its wider-key
> control's **0.596**; only the tensor state passes both, at **0.714**. All
> trained results are single-seed synthetic studies. The evidence motivates
> outer products, signed geometry, and multi-step correction; it does not
> establish a worst-case or language-model capacity law.

— quoted verbatim from the
[paper](paper/ThetaMem-Signed-Multiplicative-Lifts.pdf); every number above is
tabulated, with what it is matched on and what it is not, in
[the snapshot and its caveats](#current-research-snapshot).

## What it is

ΘetaMem projects each key through two independently learned matrices and
multiplies them. The product is quadratic in the key, so unrelated records
separate faster, and it is **signed**, so their cross-talk cancels in
expectation instead of piling up with one sign. Taking the **outer** product
instead of the elementwise one turns those two branches into two axes of a
tensor state — a much larger memory bought without widening a single
projection around it.

**Public Preview v0.1 · 2026-08-12.** The implementation and reasoning are
public; the benchmark study is in progress — multi-seed replication,
benchmarking the corrected and centered memories, write gating, fused
tensor-state kernels, and language-model-scale validation are open.

[Paper: *ΘetaMem: Signed Multiplicative Lifts for Fixed-State Sequence Memory* (PDF)](paper/ThetaMem-Signed-Multiplicative-Lifts.pdf) ·
[Paper source and versions](paper/README.md) ·
[API reference](docs/API.md) ·
[Algorithms](docs/ALGORITHMS.md) ·
[Reproduce experiments](docs/EXPERIMENTS.md) ·
[Capacity simulations](examples/capacity/README.md) ·
[Release notes](CHANGELOG.md)

## News

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
DeltaNet-2) write residuals. With full replay, residual iterations can solve
a least-squares problem through the key Gram matrix; a single causal delta
pass is not that solver, and corrective writes still share the available
feature directions. Positive-diagonal self-product memories take another
route: grouped self-outers make the query/key score a nonnegative sum of
squared similarities and give the additive read a positive mass mode.

ΘetaMem answers with four hypotheses, and the
[paper](paper/ThetaMem-Signed-Multiplicative-Lifts.pdf) reports what each one
currently rests on:

- **H1 — allocation.** Outer products enlarge the state without widening the
  key projection. *Trained support; its functional-rank ceiling is measured
  and stated.*
- **H2 — geometry.** Signed decorrelated factors address a bounded state
  better than positive geometries at a comparable key budget, because
  nonnegative read weights cannot cancel. *Supported by a random-key surrogate
  and by a trained, parameter-matched comparison.*
- **H3 — correction.** Gram overlap can be corrected without erasing, and one
  correction is not the natural stopping point. *Direction supported; the
  quantity is not yet benchmarked.*
- **H4 — hygiene.** H2 is conditional on centered values. *Implemented and
  unit-tested; not yet benchmarked.*

Separate maps from the same source are not statistically independent, and a
larger physical outer state need not have equally large functional rank; the
signed advantage requires decorrelated factors and centered unrelated values,
and is not guaranteed for an arbitrary trained model.

The two core **product mechanisms** use the same learned branch projections:

| state | idea | what it buys |
|---|---|---|
| `"outer"` | every coordinate of one branch pairs with every coordinate of the other | a much larger tensor state and a product similarity |
| `"hadamard"` | corresponding branch coordinates are multiplied | a product similarity without growing the recurrent state |

The preset builds both branches from one projected key. The lift algebra also
supports disjoint key parts. Those choices allocate the same kind of tensor
array but do not have the same functional rank: separate matrices are not the
same thing as independent information. The
[`rank_geometry.py`](examples/capacity/rank_geometry.py) surrogate isolates
that distinction; neither construction promises that training will occupy its
algebraic upper bound.

Beyond the presets, a small **lift algebra** composes branches, the raw key
and its parts, elementwise products, direct sums, and outer products into
custom state layouts, with validation rules that keep every instance
executable by the same scan machinery.

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

### How to read the snapshot

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

## Install and first forward pass

```bash
pip install torch numpy
pip install -e .
```

Start here. Every argument below has a default, and the defaults are the
recorded configuration: the Hadamard lift, a pure additive `sum`, no value
centering, and the chunked executor.

```python
import torch
import thetamem

layer = thetamem.ThetaMemLayer(128, heads=4, key_dim=32, value_dim=64)
y = layer(torch.randn(2, 1024, 128))   # [batch, time, d_model]

layer.state_size    # 9,728 streaming floats per layer
```

Change one thing at a time. `state="outer"` is the tensor memory of the tables
above — it is the strongest arm we measured and it costs 27× the streaming
state (263,680 floats per layer), so it is a deliberate choice rather than a
default:

```python
layer = thetamem.ThetaMemLayer(
    128, heads=4, key_dim=32, value_dim=64,
    state="outer",              # "hadamard" | "concat" | "outer"
    update="second_pass",       # "sum" | "second_pass" | "multi_pass" | "delta"
    feature_norm="center",      # final whole-feature coordinate centering
    value_ops=("conv", "silu", "center"),  # center V after its frontend
    value_center="exact_mean",  # "none" | "running_mean" | "exact_mean"
    backend="chunked",          # "naive" | "chunked" | "fla"
    chunk=256,
)
```

The memory alone (no projections, no gates) is `thetamem.ThetaMemory`. The
three presets are shorthands for lift expressions — `state="outer"` is
`outer(branch(32), branch(32))` — and the algebra builds the rest:

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
minimal harness — no benchmark framework is required. MQAR, the three state
kinds:

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

v0.1 of the library is the additive-and-corrected lane: causal sums, optional
value centering, a strict-prefix second-state correction, an experimental
causal triangular `multi_pass`, a sequential delta reference, and global
offline replay solvers. The shipped `multi_pass` has no **global**
least-squares convergence claim; `replay_fit` must retain or reconstruct all
records. No temporal decay banks, no write gating (an open
direction the paper motivates), no streaming decode path, and the `fla`
backend delegates flat lifts to the optional `flash-linear-attention` package
and fails closed otherwise. The chunked executor is portable PyTorch and
compiles end to end.

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
  note    = {Versioned technical paper, Public Preview v0.1},
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
