# Third-party notices

This repository bundles **no third-party source code**. Two data-generation
modules reimplement published synthetic task definitions; the reimplemented
code in `src/thetamem/data/` is original to this project and licensed under
the repository license, but the task designs deserve attribution, and users
comparing against the original suites should know the relationship.

| Component | Relationship | Upstream license | Bundled here? |
|---|---|---|---|
| MQAR task definition (`thetamem.data.mqar`) | Reimplementation of the multi-query associative recall synthetic from the zoology benchmark (Arora, Eyuboglu, et al., *Zoology: Measuring and improving recall in efficient language models*, 2023; github.com/HazyResearch/zoology). Adds a seeded local-RNG filler policy. | Apache-2.0 | No (reimplemented) |
| MAD task definitions (`thetamem.data.mad`) | Reimplementation of the fuzzy in-context recall and selective copying tasks from the MAD suite (Poli, Thomas, Massaroli, et al., *Mechanistic Design and Scaling of Hybrid Architectures*, 2024; github.com/athms/mad-lab, MIT, Copyright (c) 2024 Armin W. Thomas). All draws are threaded through one seeded generator. | MIT | No (reimplemented) |
| KATA Sigma2 comparison geometry (`examples/capacity/psd_self_product_capacity.py`, and the trained comparator of `docs/EXPERIMENTS.md`) | Independent implementation of the grouped symmetric positive feature and its normalized numerator/mass read, written from the equations published by Ayoub Ghriss and Sourav Chakraborty, *Kernelized Linear Attention: Breaking the Capacity Wall with Symmetric Cones*, arXiv:2607.17419, 2026. No upstream source or kernel code is copied, and the comparison is ours, not an endorsement or a claim about their tuned implementation. | Upstream repository declared no license in the audited snapshot; only the published equations were reimplemented | No (equations reimplemented; the trained comparator is not shipped) |
| `flash-linear-attention` | Optional runtime dependency of the explicit `fla` backend (`pip install thetamem[fla]`). Never imported unless that backend is requested. | MIT | No (dependency) |
| PyTorch, NumPy | Runtime dependencies. | BSD-style | No (dependencies) |

Benchmark baselines referenced in the paper and docs (Gated DeltaNet-2 and
its reference implementation) were run in a separate benchmarking harness and
are **not** part of this repository; note that the upstream GDN-2 reference
implementation carries a non-commercial source license, which is one reason
no baseline code is bundled or borrowed here.

Nothing in this file relicenses any upstream project, and nothing in the
repository license applies to the upstream projects listed above.
