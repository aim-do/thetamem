# Reproducing the experiments

This page records the full protocols behind the numbers in the README and
the paper, and maps them to the shipped harness. Two trained snapshots are
recorded. The **2026-08 matched-state suite** (v0.2; one H100 per cell, plus a
small-state league on Kaggle T4 GPUs) ran in the project's laboratory harness;
the **v0.1 snapshot** came from the original benchmark harnesses (zoology for
MQAR, the MAD suite's task definitions vendored into the same harness) on one
H100. Every number is single-seed. The examples here reproduce the
**protocol**, not the bit pattern.

Two evidence levels are kept separate below. MQAR and MAD are trained-model
measurements. The capacity programs are deterministic NumPy theory
surrogates: they isolate algebraic rank, a grouped PSD/self-product geometry,
and offline replay solvers. They are not additional library architectures,
trained comparisons, or validation of the library's causal `multi_pass`
update.

## Matched-state suite (2026-08)

The v0.2 headline. Every memory is matched on **core state floats** (8,192 per
memory layer) and key width; the question is what the key geometry alone buys.

- **Shell.** The common shell below (width 128, gated short-convolution block,
  then the memory block, tied head). Every memory: 4 heads, key width 32, value
  width 64.
- **ΘetaMem frontend.** Raw Q/K: bias-free projection -> causal depthwise
  convolution (width 4) -> lift, with no activation, position, or
  normalization. V: projection -> convolution -> SiLU. Output: per-head
  RMSNorm (epsilon 1e-8) times a low-rank SiLU gate. Writes are additive
  (`update="sum"`) unless a row says otherwise.
- **Baseline.** Gated DeltaNet-2 in the same shell with its own convolutions,
  gates, and output normalization, running the official
  flash-linear-attention Triton operators at pinned revision `595682b8`.
- **MQAR.** The v0.1 training mixture (vocabulary 8,192, lengths 64-256,
  4-64 pairs), batch 256, 32 epochs = 22,624 steps, AdamW with weight decay
  0.1, per-epoch cosine decay. Scored on three long slices,
  `thetamem.data.mqar.LONG_EVAL_SLICES` = 1,024 tokens / 256 pairs, 2,048 /
  512, and 4,096 / 512 (1,000 examples each, evaluation seed 1123), and
  reported as their **unweighted mean**. The last slice doubles length at fixed
  load. Fillers are drawn from the full vocabulary and can collide with a
  stored key or value.
- **MAD.** Fuzzy in-context recall and selective copying at the v0.1
  settings below (6,400 / 1,280 examples, batch 128, 10,000 steps, weight
  decay 0).
- **Grid and numerics.** Learning rates {3.16e-4, 1e-3, 3.16e-3}, one run
  each at seed 123; the best cell per task is reported, with no automatic
  grid-edge extension. bf16 autocast with an FP32 loss. ΘetaMem compiled end
  to end with Inductor behind a fail-closed eager-parity gate; the baseline
  ran its fused kernels eagerly. One H100 80GB per cell.

```bash
python examples/mqar.py --state shifted_hadamard --qk-ops conv --position none --eval long --lr 1e-3
python examples/mad.py --task fuzzy     --state shifted_hadamard --qk-ops conv --position none --lr 1e-3
python examples/mad.py --task selective --state shifted_hadamard --qk-ops conv --position none --lr 3.16e-3
```

With these flags the example shell has exactly the recorded parameter counts
(1,388,288 on MQAR, 341,760 on MAD), and its memory layer,
`ThetaMemLayer(128, heads=4, key_dim=32, value_dim=64,
state="shifted_hadamard", qk_ops=("conv",), position="none", eps=1e-8)`, is
numerically identical to the recorded arm's (same parameter set and
initialization from the same seed; identical outputs and gradients in a CPU
check). The linear-key control is the same call with `lift=thetamem.key()`,
verified the same way.
The example's default `eps=1e-6` differs from the recorded output-norm
epsilon of 1e-8; pass a custom shell for an exact match.

Recorded results, best over the grid (score = mean of the three long slices):

| memory (8,192 core state floats) | params (MQAR / MAD) | MQAR score | 1,024 / 256 | 2,048 / 512 | 4,096 / 512 | fuzzy | selective |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gated DeltaNet-2 | 1,453,764 / 407,236 | 0.1948 | 0.4000 | 0.0974 | 0.0871 | 0.2715 | 0.9187 |
| ΘetaMem linear key (`lift=key()`) | 1,388,288 / 341,760 | 0.1404 | 0.2793 | 0.0743 | 0.0675 | 0.3844 | 0.1853 |
| ΘetaMem linear key + 1 deletion replay † | 1,388,292 / 341,764 | 0.0517 | 0.1269 | 0.0157 | 0.0125 | 0.4453 | 0.5567 |
| ΘetaMem linear key + 2 deletion replays † | 1,388,296 / 341,768 | 0.0112 | 0.0329 | 0.0004 | 0.0004 | 0.3191 | 0.7936 |
| **ΘetaMem `shifted_hadamard`** | 1,388,288 / 341,760 | **0.4337** | **0.7365** | **0.2919** | **0.2727** | 0.2347 | 0.1858 |
| ΘetaMem `shifted_hadamard` + 1 deletion replay † | 1,388,292 / 341,764 | 0.3768 | 0.7070 | 0.2307 | 0.1927 | 0.1898 | 0.4557 |
| ΘetaMem `shifted_hadamard` + 2 deletion replays † | 1,388,296 / 341,768 | 0.1131 | 0.2486 | 0.0513 | 0.0393 | 0.2231 | 0.7746 |

Per-cell grid (learning rate: score), in the order 3.16e-4 / 1e-3 / 3.16e-3:

| memory | MQAR | fuzzy | selective |
|---|---|---|---|
| Gated DeltaNet-2 | 0.1559 / **0.1948** / 0.1372 | 0.2116 / **0.2715** / 0.1654 | 0.8389 / **0.9187** / 0.8728 |
| ΘetaMem linear key | 0.1167 / **0.1404** / 0.0337 | **0.3844** / 0.3097 / 0.3286 | 0.1784 / 0.1839 / **0.1853** |
| ΘetaMem `shifted_hadamard` | 0.3507 / **0.4337** / 0.1962 | 0.1697 / **0.2347** / 0.2139 | 0.1787 / 0.1819 / **0.1858** |

Follow-up runs under the same protocol, one suite each (the rows are
directly comparable with the table above; † marks laboratory arms that this
repository does not build):

| memory (8,192 core state floats) | params (MQAR) | MQAR score | 1,024 / 256 | 2,048 / 512 | 4,096 / 512 | fuzzy | selective |
|---|---:|---:|---:|---:|---:|---:|---:|
| ΘetaMem `shifted_hadamard`, no Q/K convolution | 1,387,264 | 0.3600 | 0.6437 | 0.2322 | 0.2041 | 0.3048 | 0.1952 |
| ΘetaMem `shifted_hadamard` + learned causal delta, post-lift L2 † | 1,388,292 | 0.4511 | 0.7642 | 0.3018 | 0.2874 | 0.1793 | 0.7616 |
| ΘetaMem `shifted_hadamard` + learned Fourier positions, additive † | 1,398,536 | 0.2492 | 0.3315 | 0.2196 | 0.1965 | 0.3443 | 0.9954 |
| ΘetaMem `shifted_hadamard` + learned Fourier positions, learned delta † | 1,398,540 | 0.2210 | 0.2784 | 0.2090 | 0.1757 | 0.3382 | 0.9951 |
| same, without Q/K convolution, additive † | 1,397,512 | 0.0656 | 0.1302 | 0.0379 | 0.0288 | 0.3759 | 0.9919 |

Arm definitions:

- **Deletion replay** (laboratory variant of `multi_pass`): the written values
  are iterated as `w <- w - rho_r * L w`, where `L` is the strict
  lower-triangular lifted-key Gram, with learned per-head `rho_r`
  initialized at 0.01; the final iterate is written into the single additive
  state, so the state does not grow.
- **Learned causal delta**: the sequential delta rule with one learned
  sigmoid strength per head (initialized at 1/33) on the L2-normalized lifted
  key, chunk 32.
- **Learned Fourier positions**: 16 learned-frequency Fourier features of the
  position, projected by a zero-initialized 16-to-128 map and added to the
  Q/K input only; then projection -> convolution -> SiLU -> smooth L2
  (epsilon 1e-6) -> identity-initialized per-head 32 x 32 maps -> the
  shifted-split lift.

Readings: the parameter-free quadratic address is the whole MQAR gain
(0.140 -> 0.434 in one shell, state, and parameter count) and passes the
baseline 2.2x on the mean and 3.1x at 4,096 tokens without any erasure. An
additive state has no clock, so selective copying stays near 0.19 until a
position signal or a correction supplies order; deletion replays buy order
with recall, and the learned Fourier frontend buys it while staying above the
baseline on all three tasks. The learned-delta arm has the best recall in the
suite but the weakest fuzzy recall; with one seed it is not ranked against the
additive arm.

Several optima sit at a grid edge: every ΘetaMem selective-copying optimum
(3.16e-3), the MQAR optima of three of the four replay arms and of both
learned-delta arms (3.16e-4), and several fuzzy-recall optima. The headline
MQAR cells of Gated DeltaNet-2 and ΘetaMem `shifted_hadamard` both peak at
the central 1e-3.

Median step times on MQAR were 12.3 ms (ΘetaMem `shifted_hadamard`, compiled)
and 48.0 ms (Gated DeltaNet-2, eager fused kernels); the comparison crosses
executors and is not a kernel benchmark.

## Small-state league (Kaggle T4, 2026-08)

The same question at a 16x smaller state, on free Kaggle T4 GPUs.

- **Shell.** The same two-block shell at width 64; every memory has 2 heads,
  key width 16, value width 16, and **512 core state floats**. FP32
  activations (TF32 matmuls allowed), Inductor-compiled training behind an
  eager-parity gate.
- **MQAR.** Vocabulary 2,048; training lengths/pairs (64, 4), (128, 8),
  (256, 16), (256, 32); 4,000 steps at batch 64, per-step cosine decay,
  AdamW weight decay 0.1; scored as the unweighted mean over five slices from
  64 tokens / 16 pairs to 1,024 / 256. Learning rates {5e-4, 1e-3, 2e-3, 5e-3,
  1e-2}.
- **MAD.** Fuzzy recall and selective copying as above, 10,000 steps at batch
  128, weight decay 0, learning rates {3.16e-4, 1e-3, 3.16e-3}.
- **Baseline.** A pure-PyTorch chunked implementation of the Gated DeltaNet-2
  recurrence (channel-wise decay, erase, and write); FLA kernels do not run
  on T4.
- Seed 123, evaluation seed 1123, best cell per task.

| memory (512 core state floats) | params (MQAR / MAD) | MQAR | fuzzy | selective |
|---|---:|---:|---:|---:|
| Gated DeltaNet-2 (PyTorch reference) | 200,786 / 70,738 | 0.1415 | 0.1477 | 0.8276 |
| ΘetaMem shifted-split, additive, raw Q/K | 194,880 / 64,832 | 0.3051 | 0.3178 | 0.1994 |
| ΘetaMem shifted-split, additive, learned Fourier positions † | 196,936 / 66,888 | 0.3435 | 0.3056 | 0.9899 |
| ΘetaMem shifted-split, learned delta, learned Fourier positions † | 196,938 / 66,890 | 0.3575 | 0.3344 | 0.9896 |
| ΘetaMem shifted-split, additive, RoPE † | 195,904 / 65,856 | 0.1923 | 0.1892 | 0.9464 |

The raw arm's Q/K path is the bare projection, with no convolution. The two
learned-Fourier rows and the RoPE row share one frontend — projection ->
SiLU -> smooth L2 -> identity-initialized per-head 16 x 16 maps -> the
shifted-split lift, again without Q/K convolution — and differ only in the
position signal: learned Fourier features added to the Q/K input, or rotary
position applied before the maps. Both restore copying; the rotation keeps
0.192 of recall where the learned features keep 0.344.

An earlier MQAR-only screen on the same small shell and MQAR protocol
agrees: Gated DeltaNet-2 0.197; the shifted-split product 0.294-0.315 over two
runs; and the parameter-free degree-four split product
`outer(hadamard(p0, p1), hadamard(p2, p3))` of the key quarters
`p_i = key_part(i, 4)` 0.27-0.33 over eight repeats of one configuration,
which also shows the run-to-run spread of this compiled small-GPU setup.

## Common shell

Two pre-RMSNorm residual blocks at width 128, each followed by a bias-free
`SwiGLU(128 -> 256 -> 128)`: block 1 mixes tokens with a gated short
convolution (BaseConv-style), block 2 is the memory under test. Tied token
embedding/head. All memories: 4 heads, projected key width 32, value width
64. The Gated DeltaNet-2 baseline ran in the same shell with its own
convolutions and gates; `key 48` denotes its wider-physical-key control.

## MQAR (v0.1 snapshot)

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

## MAD (v0.1 snapshot)

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

Single seed; synthetic recall tasks only. In the 2026-08 suite every arm is
state-matched and the ΘetaMem shells carry **fewer** trainable parameters than
the baseline (4.5% fewer on MQAR, 16% fewer on MAD). In the v0.1 snapshot arms
are state-matched (hadamard) or capacity-scaled (concat, outer), never
parameter-matched — the theta shell carries ~11% more trainable parameters
than the baseline on MQAR and more on MAD, where vocabularies are tiny.
Recorded step times (2026-08 MQAR: theta 12.3 ms, baseline 48.0 ms median;
v0.1 MAD: theta 19-24 ms, baseline 29-33 ms) compare a compiled theta stack
against the baseline's eager fused kernels. This harness regenerates data
with its own seeded generators, so reruns reproduce the procedure, not
bitwise-identical datasets. Expect run-to-run spread of a few points on the
hardest slices. Two different PSD/self-product comparisons appear on this
page and must not be conflated: the trained one is the parameter-matched K32
run above, and the idealized NumPy surrogate under "Capacity simulations" is
state-matched with random keys and no training at all.
