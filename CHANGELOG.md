# Changelog

## Unreleased

- Added `feature_norm="center"` as final whole-feature coordinate centering.
  Flat lifts are centered after their complete expression. Outer lifts use an
  exact lazy rank-one correction in chunked `sum`, `second_pass`, and
  `multi_pass`, preserving factorized chunk-local scores and the existing
  tensor `state_shape` and `state_size`; `delta` and `fla` retain their
  existing flat-feature materialization.
- Clarified the existing ordered
  `value_ops=(..., "center")` frontend option: it centers each per-token value
  vector across coordinates after preceding transformations. This remains
  separate from `value_center="running_mean"|"exact_mean"`, which centers each
  value channel across causal records inside the memory algorithm.

## 0.1.0 — 2026-08-12

**First public release.** The additive lane of the signed multiplicative-lift
memory with value centering and iterated corrections, complete enough to drop
into another stack and to rerun the two benchmark protocols.

- **Lift algebra** (`branch`, `key`, `key_part`, `normalize`, `hadamard`,
  `outer`, `concat`, `state_lift`): declarative feature constructions over
  the projected key, validated so that every accepted lift is either flat or
  factorized. The three canonical state kinds — Hadamard (`F x V`, matched to
  the baseline's state), concatenated (`(Dk + F) x V`), and outer
  (`F x F x V`, factorized reads) — are presets over the same algebra.
- **`ThetaMemory`**: no-mass reads over the lifted key; updates `sum`
  (additive), `second_pass` (strict-prefix residual correction carried in a
  second state; learned per-head blend, initialized 0.9 toward the base
  read), `multi_pass` (`passes` causal triangular Richardson iterates with
  learned per-pass strengths initialized at 0.1; one pass exactly matches the
  initial 90/10 second-pass blend), and `delta` (sequential delta rule with raw per-head
  strength `beta = sigmoid(logit)`, no `1/||Phi||^2` division).
- **Offline replay fitting** (`replay_fit`, `replay_read`): matrix-free
  Richardson, heavy-ball, per-value CGNR/CGLS, and cyclic delta sweeps over a
  completed record set. Outer factors remain separate and no token-token Gram
  is formed; every iteration still replays the context and touches dense
  tensor-state workspaces.
- **Value centering** (`value_center`): `"running_mean"` (causal running-mean
  subtraction with add-back at read) and `"exact_mean"` (exact retroactive
  centering through the signed key mass — a ones channel through the same
  scan; subtractive, no denominator). Centered reads commute with constant
  value shifts for every update.
- **Executors**: `naive` (masked quadratic oracle), `chunked` (batched score
  tiles for flat lifts, sequential chunk walk with factorized scores for
  outer lifts, exclusive shifted-prefix carries), and explicit `fla`
  delegation for flat lifts (fails closed without the optional dependency).
- **`ThetaMemLayer`**: the recorded shell — bias-free Q/K/V with ordered
  per-stream operation pipelines (causal depthwise conv, SiLU, optional RoPE,
  normalizations), linear or direct-Hadamard Q/K wiring, per-head RMSNorm
  scaled by a low-rank SiLU gate, Xavier gain `2**-2.5`.
- **Data generators** (`thetamem.data`): MQAR with seeded local-RNG fillers
  and the recorded training mixture / evaluation slices; MAD fuzzy
  in-context recall and selective copying with all draws through one seeded
  generator.
- **Examples**: minimal MQAR and MAD harnesses (shared two-block shell, tied
  embedding, AdamW + per-epoch cosine), one configuration per invocation,
  `--smoke` sanity mode; three capacity simulations
  (`examples/capacity/`) behind the paper's accounting — functional-rank
  geometry, grouped PSD/self-product versus an ideal signed control, and offline
  replay-solver convergence.
- **Tests**: 71 unit tests — lift validation, float64 backend parity against
  independent sequential references for all four updates, strict-prefix
  semantics, value-centering shift equivariance and exactness, delta
  chunk/naive agreement, layer causality, data determinism.
- **Docs and paper**: API reference, algorithm notes, experiment protocols
  with the recorded reference numbers; versioned technical paper (Public
  Preview v0.1) with render pipeline and figures.

Known limitations of this snapshot: no temporal decay banks, no write
gating, no streaming decode path; the `fla` backend requires the optional
package and a supported GPU kernel path; recorded benchmark numbers are
single-seed.
