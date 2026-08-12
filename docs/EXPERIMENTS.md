# Reproducing the experiments

This page records the full protocols behind the numbers in the README and
the paper, and maps them to the shipped harness. The recorded numbers come
from the original benchmark harnesses (zoology for MQAR, the MAD suite's
task definitions vendored into the same harness) on one H100, single seed;
the examples here reproduce the **protocol**, not the bit pattern.

Two evidence levels are kept separate below. MQAR and MAD are trained-model
measurements of shipped layer variants. The capacity programs are deterministic
NumPy theory surrogates: they isolate algebraic rank, a grouped PSD/self-product
geometry, and offline replay solvers. They are not additional library
architectures, trained comparisons, or validation of the library's causal
`multi_pass` update.

## Common shell

Two pre-RMSNorm residual blocks at width 128, each followed by a bias-free
`SwiGLU(128 -> 256 -> 128)`: block 1 mixes tokens with a gated short
convolution (BaseConv-style), block 2 is the memory under test. Tied token
embedding/head. All memories: 4 heads, projected key width 32, value width
64. The Gated DeltaNet-2 baseline ran in the same shell with its own
convolutions and gates; `key 48` denotes its wider-physical-key control.

## MQAR

- Vocabulary 8,192; training mixture `(len 64, 100k examples, 4 pairs)`,
  `(128, 20k, 8)`, `(256, 20k, 16)`, `(256, 20k, 32)`, `(256, 20k, 64)`.
- Evaluation: `(256, 1k, 64)` — hardest in-distribution — and
  `(1024, 1k, 256)` — 4x length extrapolation.
- AdamW, weight decay 0.1, fused on CUDA; per-epoch cosine decay, no warmup;
  32 epochs (22,624 steps at batch 256); seed 123.
- Learning-rate grid {1e-3, 3.16e-3, 1e-2}; report the best cell.

```bash
python examples/mqar.py --state hadamard --lr 3.16e-3   # matched-state arm
python examples/mqar.py --state concat   --lr 1e-3
python examples/mqar.py --state outer    --lr 1e-3
python examples/mqar.py --state hadamard --update second_pass --lr 1e-3
python examples/mqar.py --state hadamard --update multi_pass --passes 3 --lr 1e-3
```

Recorded reference (best over the grid, accuracy):

| memory | core state | key params | overall | len 256 | len 1,024 |
|---|---:|---:|---:|---:|---:|
| raw linear key (no lift) | 8,192 | 16,384 | 0.876 | 0.995 | 0.300 |
| Gated DeltaNet-2 | 8,192 | 16,384 | 0.930 | 0.998 | 0.567 |
| Gated DeltaNet-2, physical key 48 | 12,288 | 24,576 | 0.973 | 1.000 | 0.814 |
| ΘetaMem hadamard | 8,192 | 24,576 | 0.952 | 1.000 | 0.668 |
| ΘetaMem hadamard + second_pass | 16,384 | 24,576 | — | — | 0.662 |
| ΘetaMem concat | 16,384 | 24,576 | 0.968 | 1.000 | 0.781 |
| ΘetaMem outer | 262,144 | 24,576 | 0.997 | 1.000 | 0.976 |

Core state counts the memory tensors only, matching the accounting of the
recorded runs. `ThetaMemLayer.state_size` reports the **inclusive** streaming
figure — core state plus the short-convolution caches (1,536 floats for the
theta shell) plus the ones channel when value centering is on — so it reads
9,728 / 17,920 / 263,680 for the three presets against the 8,192 / 16,384 /
262,144 core figures tabulated here.
Key params count the trained maps that produce the key features per memory
layer: the token-to-key projection (128 x 128 = 16,384) plus the lift branch
weights (2 x 4 heads x 32 x 32 = 8,192, shared with the query side); short
convolutions and the query/value/output paths are excluded because they are
equal across arms. States below 8,192 floats have not been measured yet.

The `overall` column is the accuracy pooled over all seven evaluation slices
of the recorded sweep, not a mean of the two slices shown here.

Archived controls from the same recorded sweeps: positive relu²-threshold
features 0.806 / 0.974 / 0.194 (state 8,192) and Gated DeltaNet-2 with two
memory layers 0.881 / 0.998 / 0.261 (16,384). The relu²-threshold arm comes
from an earlier generation of the codebase and cannot be built with the
published API; the paper argues the elementwise-positive family out in its
Section 2.3 rather than resting on this number.

None of the baseline or archived-control rows is built by the code in this
repository. They were run in the project's benchmark harness under the
protocol above; this tree ships the ΘetaMem layers and the data generators,
so those rows can be re-run under the same protocol but not reproduced from
this tree alone.

### Signed versus PSD self-product, trained at equal key width

A separate recorded run matches on the opposite axis from the tables above:
equal key width and **exactly equal trained parameters** (1,388,288 per arm;
125,440 mixer parameters), with the state left free to differ. Both lifts are
parameter-free functions of the same 32-wide projected key, so the arms differ
in geometry alone. Four heads, `value_dim=64`, no RoPE, bit-identical training
tensors and common initial tensors, 22,624 steps at `lr=1e-3`, seed 123, bf16,
chunked backend at `chunk=256`; each checkpoint then evaluated frozen at four
lengths.

| memory | feature width | core state | 256 / 64 | 1,024 / 256 | 2,048 / 512 | 4,096 / 1,024 |
|---|---:|---:|---:|---:|---:|---:|
| grouped PSD self-product + positive mass | 136 | 35,360 | 1.000000 | 0.983781 | 0.484236 | 0.024242 |
| signed outer of the two key halves, raw read | 256 | 65,536 | 1.000000 | 0.999961 | 0.986990 | 0.742194 |
| signed outer, half 0 L1 / half 1 L2 | 256 | 65,536 | 1.000000 | 0.999945 | 0.981914 | 0.699979 |

The signed arm is expressible with the published API as
`outer(key_part(0, 2), key_part(1, 2))`. The comparator is our own independent
implementation of the grouped Sigma2 equations published by KATA (Ghriss and
Chakraborty, *Kernelized Linear Attention: Breaking the Capacity Wall with
Symmetric Cones*, arXiv:2607.17419) — no upstream source or kernel code was
used — and it is **not shipped in this release**, so this row cannot be
rebuilt from this tree. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

Caveats specific to this run: it is parameter-matched and **not**
state-matched — the signed outer carries 1.853× the comparator's core floats;
it is one learning-rate cell at one seed, not best-over-grid; evaluation
fillers are drawn from the full vocabulary and may collide with a stored key,
so the long slices mix retention with collision distractors, and because every
arm consumed byte-identical tensors the comparison stays controlled but the
absolute values at 4,096 are not a collision-free retention measurement.
Isolated 20-step smoke medians were 21.829 / 17.623 / 15.152 ms per step; the
production run shared one H100 across three workers, so its timings are
excluded from any latency comparison.

## MAD

- Tasks at their standard settings: **fuzzy in-context recall** (vocabulary
  16, length 128, key/value motifs up to 3 tokens, multi-query) and
  **selective copying** (vocabulary 16, length 256, 16 tokens to copy).
- 6,400 training / 1,280 test examples; batch 128; 200 epochs = 10,000
  steps; AdamW, weight decay 0.0; per-epoch cosine decay; seed 123.
- Learning-rate grid {5e-4, 1e-3, 3.16e-3}; report the best cell.

```bash
python examples/mad.py --task fuzzy     --state outer    --lr 3.16e-3
python examples/mad.py --task fuzzy     --state concat   --lr 1e-3
python examples/mad.py --task fuzzy     --state hadamard --lr 3.16e-3
python examples/mad.py --task fuzzy     --state hadamard --update second_pass --lr 3.16e-3
python examples/mad.py --task fuzzy     --state hadamard --update multi_pass --passes 3 --lr 3.16e-3
python examples/mad.py --task selective --state hadamard --lr 1e-3
python examples/mad.py --task selective --state concat   --lr 3.16e-3
python examples/mad.py --task selective --state outer    --lr 3.16e-3
```

Recorded reference (best over the grid, accuracy):

| memory | core state | key params | fuzzy | selective |
|---|---:|---:|---:|---:|
| Gated DeltaNet-2 | 8,192 | 16,384 | 0.323 | 0.869 |
| Gated DeltaNet-2, physical key 48 | 12,288 | 24,576 | 0.596 | 0.914 |
| ΘetaMem hadamard | 8,192 | 24,576 | 0.181 | 0.988 |
| ΘetaMem hadamard + second_pass | 16,384 | 24,576 | 0.264 | 0.982 |
| ΘetaMem concat | 16,384 | 24,576 | 0.398 | 0.983 |
| ΘetaMem outer | 262,144 | 24,576 | 0.714 | 0.982 |

The remaining MAD recall tasks (exact in-context recall, noisy recall,
memorization) saturate for every memory above and are omitted.

## Capacity simulations

The theory checks live in the
[capacity simulation guide](../examples/capacity/README.md) and run with only
NumPy:

```bash
python examples/capacity/rank_geometry.py
python examples/capacity/psd_self_product_capacity.py
python examples/capacity/offline_replay_solvers.py
```

- [`rank_geometry.py`](../examples/capacity/rank_geometry.py) distinguishes
  allocated cells from functional rank. Its fixed-seed 16x16 check observes
  ranks `36 / 256 / 256` for same-source width-8, disjoint split, and
  independent factors. At the production scale used in the argument,
  `(A x) ⊗ (B x)` with `d=32` and width-32 factors has
  `R_phys=1024` but `R_func<=32*33/2=528`; a width-64 source split 32+32 can
  generically use all 1024 coordinates.
- [`psd_self_product_capacity.py`](../examples/capacity/psd_self_product_capacity.py)
  implements a grouped PSD/self-product Gram, not a generic elementwise-positive
  proxy. The construction matches the exact grouped PSD geometry attributed in
  Section 5.3 of the paper. Its signed comparator uses
  independent 10x14 factors. With four
  heads and value width 64, complete memory states are 35,840 versus 35,360
  floats; including the shared 1,536-float frontend cache gives 37,376 versus
  36,896 (`signed/PSD=1.0130`). This matches polynomial degree and state,
  not raw key width or learned projection parameters.
- [`offline_replay_solvers.py`](../examples/capacity/offline_replay_solvers.py)
  compares full-residual Richardson, model-tuned heavy-ball, and conjugate
  gradient. Every iteration rereads all records, so it exercises the
  offline/prefill replay problem, not the shipped causal correction.

Fixed seed `20260812`, grouped PSD/self-product output (interference power and relative
MSE; lower is better):

| records | interference power signed / PSD | additive signed / PSD | delta sweep 1 signed / PSD | delta sweep 2 signed / PSD | delta sweep 3 signed / PSD |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.2211 / 0.5775 | 0.1720 / 0.3170 | 0.1250 / 0.2145 | 0.0174 / 0.0521 | 0.0034 / 0.0158 |
| 64 | 0.4500 / 1.1758 | 0.2998 / 0.4601 | 0.2640 / 0.3590 | 0.0848 / 0.1662 | 0.0371 / 0.0957 |
| 128 | 0.9083 / 2.3726 | 0.4739 / 0.6268 | 0.5440 / 0.6443 | 0.3480 / 0.4687 | 0.2689 / 0.3950 |
| 256 | 1.8230 / 4.7659 | 0.6411 / 0.7627 | 0.9821 / 1.0583 | 0.9112 / 0.9852 | 0.8990 / 0.9726 |

These are conditional average-case results. Values are iid centered Gaussian,
the signed factors are independent, and the additive metric permits one
per-trial target-dependent oracle rescale (computed from the values being
reconstructed, so it is a diagnostic and not an attainable error). The
interference-power ratio needs no fitted quantity and is the assumption-light
number here. Negative coefficients alone give no worst-case guarantee: noise power
still depends on squared weights, separate learned maps of one source need not
be independent, and correlated factors or sign-aligned values can erase the
advantage. At light and moderate load, the second sweep removes a large part
of the remaining error; sweeps two and three replay the records. The size of
this benefit is load-dependent, not a universal two-pass guarantee.

The replay solver makes the implementation boundary visible. At load 0.50
its first-draw `lambda_max=2.902`; relative MSE is:

| global replay pass | Richardson 1.0 | Richardson 0.5 | heavy-ball | CG |
|---:|---:|---:|---:|---:|
| 1 | 0.4999 | 0.3621 | 0.4999 | 0.3208 |
| 3 | 1.161 | 0.1485 | 0.1258 | 0.0630 |
| 5 | 7.553 | 0.0820 | 0.0365 | 0.0150 |
| 10 | 2,361 | 0.0274 | 0.00285 | 0.000445 |

Heavy-ball uses the random-design schedule `eta=1`, `beta=T/state`; it is
not a universal safe setting. Unit Richardson diverges because
`lambda_max>2`, while damped Richardson, tuned heavy-ball, and CG approach
the global fitted solution. The practical result is that three to five
replays already reduce error substantially. These curves validate that
global replay problem only; causal `multi_pass` instead approximates a
lower-triangular prefix system.

For a lifted width `R`, per-head value width `V`, and sequence length `T`, the
exact carried-state work is `Θ(T R V)` and state is `Θ(R V)`. Dense attention
is `Θ(T^2 (d_k + V))`; the fixed-state memory is cheaper only when
`R V < T (d_k + V)`, roughly `R ≲ T` for comparable widths. Outer
factorization can keep the learned projection small, but does not remove the
full-state read/write cost.

## Caveats that travel with every number

Single seed; synthetic recall tasks only. Arms are state-matched (hadamard)
or capacity-scaled (concat, outer), never parameter-matched — the theta
shell carries ~11% more trainable parameters than the baseline on MQAR and
more on MAD, where vocabularies are tiny. Recorded step times on the MAD
runs (theta 19-24 ms, baseline 29-33 ms) compare a compiled theta stack
against the baseline's eager fused kernels. This harness regenerates data
with its own seeded generators, so reruns reproduce the procedure, not
bitwise-identical datasets. Expect run-to-run spread of a few points on the
hardest slices. Two different PSD/self-product comparisons appear on this
page and must not be conflated: the trained one is the parameter-matched K32
run above, and the idealized NumPy surrogate under "Capacity simulations" is
state-matched with random keys and no training at all.
