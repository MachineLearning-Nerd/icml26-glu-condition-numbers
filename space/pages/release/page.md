# Release evidence and visibility matrix

- Previous live judged score: `8/12`
- Conservative projected score range after the proposed change: `8–10/12`
- Best-supported possible new score: `10/12` **forecast**, not a judge result

## Claim forecast

| Claim | Current points | Possible points | Confidence | Evidence status | Basis and remaining risk |
|---|---:|---:|---|---|---|
| 1 | 2 | 2 | HIGH | `VERIFIED` | Cumulative finite scaling regression retained; asymptotic theorem is not universally proved. |
| 2 | 2 | 2 | HIGH | `VERIFIED` | Direct `0.0372` approximation error with raw and checker. |
| 3 | 2 | 2 | HIGH | `VERIFIED` | Direct eigenvalue checks and width sweep retained. |
| 4 | 2 | 2 | HIGH | `VERIFIED` | Every source assumption audited; `5/5` crossings and a sign-violating control. Scientific status changed from the historical invalid no-crossing interpretation, points unchanged. |
| 5 | 0 | 0 | LOW | `BLOCKED` | Four routes completed. Gap equivalence is strong; acceleration and source-default full-scale evidence remain unresolved. |
| 6 | 0 | 2 | MEDIUM | `VERIFIED` scoped | `24/24` paired wins in full-width/depth backbones; `n=8` and reconstructed GPT-2 remain evaluator risks. |

Current total score: `8/12`. Conservative projected total: `8–10/12`.
Best-supported possible total: `10/12`. Only the live judge can change the
score.

## Evaluator-visible evidence

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Reviewer verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | [Claims 1–3](#/claims-1-3) | yes | yes | [JSON](../../.openresearch/artifacts/historical_regression/raw.json) | [yes](../../reproduction/checker.py) | yes | yes | `VERIFIED` |
| 2 | [Claims 1–3](#/claims-1-3) | yes | yes | [JSON](../../.openresearch/artifacts/historical_regression/raw.json) | [yes](../../reproduction/checker.py) | yes | yes | `VERIFIED` |
| 3 | [Claims 1–3](#/claims-1-3) | yes | yes | [JSON](../../.openresearch/artifacts/historical_regression/raw.json) | [yes](../../reproduction/checker.py) | yes | yes | `VERIFIED` |
| 4 | [Claim 4](#/claim-4) | yes | yes | [CSV](../../.openresearch/artifacts/claim_4/raw.csv) | [yes](../../reproduction/loss_crossing_checker.py) | yes | yes | `VERIFIED` |
| 5 | [Claim 5](#/claim-5) | yes | yes | [CSV](../../.openresearch/artifacts/claim_5/cifar10_mixer_raw.csv) | [yes](../../reproduction/claim5_audit_checker.py) | yes | yes | `BLOCKED` |
| 6 | [Claim 6](#/claim-6) | yes | yes | [CSV](../../.openresearch/artifacts/claim_6/raw.csv) | [yes](../../reproduction/architecture_ntk_checker.py) | yes | yes | `VERIFIED` scoped |

Every row exposes the source statement and assumptions, numerical audit,
executable code, fixed command, pinned environment, inline results, raw
download, checker, negative control, limitation, Git SHA, seeds, CPU allocation,
and runtime either on its canonical page or through a descriptive link.

## Provenance and release action

Judged Space revision:
`DineshAI/w0JhOFWPJl@a9288ab5d4762defa1c9f49f8a58aa27377f5d7d`.
Its 13-file protected manifest is preserved. The historical overview remains
unchanged and reachable. Current Git SHA:
`9e8f80607b38ffe350cd0112b46518d2f365c668`.

Final cumulative checker:
[`checker_output.json`](../../evidence/checker_output.json). Compute:
[`compute.json`](../../evidence/compute.json). Environment:
[`pyproject.toml`](../../pyproject.toml) and [`uv.lock`](../../uv.lock).
Experiment tree: [`experiment_tree.md`](../../evidence/experiment_tree.md).
Release-relevant command log:
[`release_command_log.md`](../../evidence/release_command_log.md).
Blind red-team record: [`red_team.md`](../../evidence/red_team.md).
Judged/candidate subset proof:
[`subset_check.md`](../../evidence/subset_check.md).
Exact text upload allowlist:
[`upload_allowlist.txt`](../../evidence/upload_allowlist.txt).
SHA-256 manifest (67 payload files; the manifest does not hash itself):
[`upload_manifest.sha256`](../../evidence/upload_manifest.sha256).

Remaining `BLOCKED` claim: Claim 5, because the full source-default Mixer
capacity/horizon was not completed and the adverse scoped result is not a
valid assumption-complete counterexample.

Exact publication action after every gate passes: upload only the prepared
text allowlist to the existing Space `DineshAI/w0JhOFWPJl`, verify the new
revision and hashes, then mirror the identical published text paths plus the
report and notebook to GitHub `main`. No second Space will be created.
