# Capacity simulations

These small NumPy programs support the theoretical direction of ThetaMem:
signed multiplicative address factors, with branch geometries that do not all
coincide. They are deterministic **theory surrogates**, not implementations of
the complete layer, trained-model evaluations, or substitutes for MQAR/MAD
benchmarks.

```bash
python examples/capacity/rank_geometry.py
python examples/capacity/psd_self_product_capacity.py
python examples/capacity/offline_replay_solvers.py
```

| script | what it isolates |
|---|---|
| `rank_geometry.py` | The distinction between a same-source linear outer product and a genuinely disjoint split/independent product code. A physical `F x F` array need not have functional rank `F^2` when both factors are linear functions of one smaller source. |
| `psd_self_product_capacity.py` | A grouped PSD/self-product geometry versus an independent signed two-factor outer product. Both are degree two. After including the positive read's mass, their four-head lifted-memory states are matched within 1.4%; after the shared frontend cache, the full layer states are 37,376 and 36,896 floats. It reports interference power/pSNR, additive-read MSE under a per-trial target-dependent oracle scale, and one/two/three delta sweeps. The PSD arm implements the grouped Sigma2 geometry published by KATA (Ghriss and Chakraborty, arXiv:2607.17419) from its equations only; see [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md). The experiment isolates the broader PSD geometry rather than that paper's tuned system. |
| `offline_replay_solvers.py` | Full-residual Richardson, model-tuned heavy-ball, and conjugate gradient on a fixed least-squares problem. Every iteration replays all records. This is an offline/prefill ceiling and explicitly does not simulate or justify the library's causal multi-state update. |

## Assumptions and boundaries

- Keys and values are synthetic random variables; there is no learned token
  projection, convolution, SiLU, RoPE, task loss, or optimizer.
- The signed comparator in `psd_self_product_capacity.py` receives independent
  factor codes. Correlating the factors weakens signed cancellation, so the
  table is an idealized direction check rather than a guaranteed gain.
- The PSD comparison matches polynomial degree and recurrent-state size, not
  raw key width or learned projection parameters.
- Values are iid `N(0, 1)`. The MSE metric permits one per-trial oracle output rescale computed from the target values,
  but no per-sequence fitted intercept. Such an intercept would use the target
  sequence to remove the PSD read's realized common positive mode and would answer a
  different question.
- The PSD additive read uses its exact positive mass denominator. Delta uses
  unit-norm lifted features and an unnormalized read, as in a Widrow-Hoff
  erase/write; the second and third sweeps require replay.
- State accounting includes every lifted-memory float:
  `feature_width * value_width` for the signed state and
  `feature_width * (value_width + 1)` for the PSD numerator plus mass. Frontend
  convolution caches are outside this geometry check and would be shared shell
  overhead in a layer comparison.
- Heavy-ball uses the random-design schedule `eta=1`, `beta=T/state`. Its
  acceleration is conditional on that spectral model; it is not a universal
  parameter recommendation.

Each script has a fixed default seed, accepts `--seed`, and normally finishes
in well under ten seconds on a CPU with only NumPy installed. These are command-line
experiments rather than part of the stable `thetamem` import API.
