# Paper versions

| Version | Date | Notes |
|---|---|---|
| Public Preview v0.2 | 2026-10-22 | New evidence and a shorter abstract. The abstract is cut to two paragraphs that state the construction (a signed quadratic Hadamard lift with additive writes) and the result, without the hypothesis-by-hypothesis framing, which remains in Sections 1 and 8. Section 4.3 adds the parameter-free shifted-split product $[a \odot b;\ a \odot \mathrm{roll}(b,1)]$; Section 7.5 reports the 2026-08 matched-state suite on H100 (8,192 state floats: MQAR long-slice score 0.434 against Gated DeltaNet-2's 0.195 and a linear key's 0.140; the additive memory's selective-copying failure; deletion replays; learned Fourier positions passing the baseline on all three tasks); Section 7.6 reports the 512-float league on Kaggle T4; the limitations become Section 7.7; Sections 8 and 9 add the corresponding verdict and the open position problem. The v0.1 data and claims are unchanged. |
| Public Preview v0.1.1 | 2026-08-13 | Abstract revised: the opening now states the motivation — linear time is why the family matters, a bounded state forgets, and the practical aim is keeping the latest record per key readable while it waits for its query — and names the three levers (state size, key nonlinearity, repeatable correction). H1 adds the sharpness of the product read; H3 states its practical retention target. No data, algorithm, or numerical claim changed. |
| Public Preview v0.1 | 2026-08-12 | First public preview. The state is framed as a joint capacity/noise/retention budget; physical outer width is separated from functional rank; signed cancellation is stated as a conditional average-case mechanism; the grouped PSD/self-product comparison and replay-solver curves are deterministic theory surrogates; the library ships the strict-prefix second pass, repeated causal triangular corrections, and separate offline replay solvers; MQAR and MAD evidence remains a single-seed study in progress. |

## Update policy

Substantive changes to data, algorithms, claims, authorship, or conclusions
advance the paper preview version and add a dated entry here. Typographical
fixes that change no claim may be applied within a version.
