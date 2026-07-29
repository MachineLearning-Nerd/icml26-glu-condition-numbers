# Reproducing why GLU improves conditioning

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-w0JhOFWPJl-the-devil-is-in-the-condition-numbers-why-is-glu-better-than-non-glu-structu/blob/main/notebooks/glu_condition_numbers.py)

This reproduction tests the claim that GLU’s gate improves optimization by
contracting the NTK spectrum. The paper predicts condition-number scaling of
`O(n/d²)` for GLU versus `O(n/d)` for non-GLU; the reproduced finite slopes are
`−1.69` versus `−0.96`. In real full-width ViT and GPT-2-small backbones, GLU
has a lower empirical FFN-NTK condition number in `24/24` paired comparisons.

Assessment: Claims 1–4 and 6 are `VERIFIED` with scoped limitations. Claim 5
is `BLOCKED`: a full-CIFAR reduced Mixer showed equivalent matched-loss
generalization gaps but no optimization acceleration, and its
dimension/depth/horizon deviations prevent a valid paper-level falsification.
The previous live judge score remains `8/12`; `10/12` is the best-supported
forecast, not an earned score.

Read the [illustrated technical report](reports/reproduction/report.md) or
open the [self-contained tutorial notebook](notebooks/glu_condition_numbers.py).
The report includes the exact paper and observed numbers, implementation,
controls, substitutions, raw evidence, and remaining risk.

Compute agreement: no GPU; short single-core checks locally, and HF
`cpu-upgrade` for all uncertain or multi-core CPU work. The long full-CIFAR run
used eight CPUs for `7,845.4 s`; the final cumulative suite used eight CPUs for
`90.82 s`.

## Experiment log

Every experiment used the same exact command:
`uv sync --frozen --no-dev && uv run python -m reproduction.run_suite`.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Public report, notebook, and release surface | Not run as an experiment (publication surface) | Presentation only | none |
| [locked historical baseline](https://github.com/MachineLearning-Nerd/icml26-repro-w0JhOFWPJl-the-devil-is-in-the-condition-numbers-why-is-glu-better-than-non-glu-structu/tree/orx/locked-historical-regression-baseline) | Freeze judged Claims 1–3 and old no-crossing result | `uv sync --frozen --no-dev && uv run python -m reproduction.run_suite` | Claims 1–3 retained; old Claim 4 assumption violation exposed | HF `cpu-upgrade`, 26 s |
| [exact loss crossing](https://github.com/MachineLearning-Nerd/icml26-repro-w0JhOFWPJl-the-devil-is-in-the-condition-numbers-why-is-glu-better-than-non-glu-structu/tree/orx/exact-corollary-4-2-loss-crossing) | Enforce every Corollary 4.2 assumption | `uv sync --frozen --no-dev && uv run python -m reproduction.run_suite` | `5/5` crossings; Claim 4 `VERIFIED` | HF `cpu-upgrade`, 37 s |
| [ViT and GPT-2 NTKs](https://github.com/MachineLearning-Nerd/icml26-repro-w0JhOFWPJl-the-devil-is-in-the-condition-numbers-why-is-glu-better-than-non-glu-structu/tree/orx/real-vit-and-gpt-2-ffn-ntks) | Measure parameter-matched real-architecture FFN NTKs | `uv sync --frozen --no-dev && uv run python -m reproduction.run_suite` | `24/24` GLU wins; Claim 6 `VERIFIED` scoped | HF `cpu-upgrade`, 90.69 s |
| [full-CIFAR Mixer](https://github.com/MachineLearning-Nerd/icml26-repro-w0JhOFWPJl-the-devil-is-in-the-condition-numbers-why-is-glu-better-than-non-glu-structu/tree/orx/full-cifar-10-mixer-generalization-gap) | Direct matched-loss gap and optimization comparison | `uv sync --frozen --no-dev && uv run python -m reproduction.run_suite` | Gap equivalent; acceleration absent; Claim 5 `BLOCKED` | HF `cpu-upgrade`, 7,845.40 s |
| [Claim 5 audit](https://github.com/MachineLearning-Nerd/icml26-repro-w0JhOFWPJl-the-devil-is-in-the-condition-numbers-why-is-glu-better-than-non-glu-structu/tree/orx/claim-5-independent-audit-and-falsification-rout) | Independent equivalence route and mandatory falsification route | `uv sync --frozen --no-dev && uv run python -m reproduction.run_suite` | Invalid counterexample rejected; final Claim 5 `BLOCKED` | HF `cpu-upgrade`, 90.82 s |

## Reproduce

```bash
uv sync --frozen --no-dev
uv run python -m reproduction.run_suite
```

The suite exits nonzero if an accepted claim check, provenance check, or
negative control fails. A successful cumulative exit reproduces Claim 5’s
honest `BLOCKED` classification; it does not relabel that claim as verified.
