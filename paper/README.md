# Paper materials

This directory contains the versioned technical paper
*ΘetaMem: Signed Multiplicative Lifts for Fixed-State Sequence Memory*.

- [`ThetaMem-Signed-Multiplicative-Lifts.md`](ThetaMem-Signed-Multiplicative-Lifts.md) — the source of record.
- [`ThetaMem-Signed-Multiplicative-Lifts.pdf`](ThetaMem-Signed-Multiplicative-Lifts.pdf) — the rendered release form.
- [`VERSIONS.md`](VERSIONS.md) — the dated version record and update policy.
- [`render_paper.py`](render_paper.py) — the renderer (Markdown to A4 PDF via a
  local Chromium browser; dependencies in
  [`requirements-render.txt`](requirements-render.txt)). No other Python
  belongs in this directory.

## Byline policy

The collective byline is **The ThetaMem Project**. It is a project byline,
not an anonymous-submission designation. Future versions may name
collaborators who opt in and make substantive research contributions.

## Citation

```bibtex
@article{thetamem2026,
  title   = {ThetaMem: Signed Multiplicative Lifts for Fixed-State Sequence Memory},
  author  = {{The ThetaMem Project}},
  year    = {2026},
  note    = {Versioned technical paper, Public Preview v0.1},
}
```

## Rebuilding the PDF

```bash
pip install -r requirements-render.txt
python render_paper.py
```

The renderer needs a local Chrome, Chromium, or Edge; pass `--chrome PATH` if
it is not on the standard paths.

## Copyright

The paper materials in this directory are **not** covered by the repository's
software license. Copyright 2026 Ultimamind SRL, Belgium. All rights
reserved. See [`../LICENSING.md`](../LICENSING.md) for the boundary.
