# Examples

Minimal reproduction harnesses for the two benchmark protocols — data
generation via `thetamem.data`, a shared two-block model shell in
[`common.py`](common.py), one training configuration per invocation. The
full protocols, launch grids, and recorded reference numbers are in
[docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md).

```bash
# the 2026-08 matched-state suite: parameter-free quadratic address, raw Q/K
python mqar.py --state shifted_hadamard --qk-ops conv --position none --eval long --lr 1e-3
python mad.py  --task fuzzy --state shifted_hadamard --qk-ops conv --position none --lr 1e-3

# the v0.1 snapshot
python mqar.py --state hadamard --lr 3.16e-3
python mqar.py --state concat   --lr 1e-3
python mqar.py --state outer    --lr 1e-3

python mad.py --task fuzzy     --state outer    --lr 3.16e-3
python mad.py --task selective --state hadamard --lr 1e-3
```

`--qk-ops` and `--position` set the memory layer's Q/K frontend (default:
`conv silu rope l2` with RoPE); `--eval long` scores MQAR on the three long
slices of the matched-state suite and prints their mean. With these flags the
shell has exactly the recorded suite's parameter counts (1,388,288 on MQAR,
341,760 on MAD). `--update second_pass` selects one correction;
`--update multi_pass --passes 3` repeats it; `--update delta` selects the
causal delta reference. `--smoke` runs a tiny CPU sanity pass. Every script prints the exact
`thetamem` API call it builds, so the configuration is the log.

The [`capacity/`](capacity/README.md) directory is a separate theory lab, not
another set of trained or shipped layer variants:

- [`rank_geometry.py`](capacity/rank_geometry.py) separates physical outer
  cells from functional rank (`R_phys=1024`, `R_func<=528` for a same-source
  32-to-32x32 linear outer, versus a generically full 1024-wide 32+32 split).
- [`psd_self_product_capacity.py`](capacity/psd_self_product_capacity.py)
  implements a grouped PSD/self-product Gram and an ideal independent signed
  comparator at nearly equal complete state. Its result is conditional
  average-case evidence, not a worst-case guarantee or a trained benchmark.
- [`offline_replay_solvers.py`](capacity/offline_replay_solvers.py) reruns all
  records for Richardson, model-tuned heavy-ball, and CG. It checks the
  completed-context problem exposed by `replay_fit`; it does not establish a
  convergence curve for the separate causal `multi_pass` update.

All three use fixed default seeds, accept `--seed`, and need NumPy only; exact
outputs and assumptions are in the
[experiment protocol](../docs/EXPERIMENTS.md#capacity-simulations). These are
theory surrogates; the *trained* same-shell comparison against the same PSD
geometry is reported separately in
[docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md#signed-versus-psd-self-product-trained-at-equal-key-width),
and its comparator is not shipped in this release.
