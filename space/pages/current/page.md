# Current verification: GLU conditioning, claim by claim

![Real-architecture GLU/non-GLU condition-number ratios](../../reports/reproduction/images/architecture_ratios.svg)

The strongest new evidence is direct: parameter-matched gated FFNs had lower
empirical NTK condition numbers in all `24/24` paired ViT and GPT-2
comparisons. Five claims have accepted evidence. Claim 5 remains honestly
`BLOCKED`.

Previous live judged score: `8/12`. Conservative projected range after this
release: `8–10/12`. Best-supported possible score: `10/12` **forecast**, not a
judge result.

## Six outcomes

| Claim | Exact paper statement tested | Observed evidence | Status |
|---|---|---|---|
| 1 | Theorem 3.1: \(\kappa(\widetilde K)=O(n/d^2)\), versus \(O(n/d)\) | slopes `−1.69` vs `−0.96` | `VERIFIED` |
| 2 | Section 3: \(\widetilde K\approx K\odot XX^\top/d\) | relative error `0.0372` | `VERIFIED` |
| 3 | Theorem 3.1: extra-\(d\) \(\lambda_{\max}\) contraction, shared \(O(m)\) \(\lambda_{\min}\) | GLU \(\lambda_{\max}\) lower at every `d`; \(\lambda_{\min}\) `27.6→102.4` as width quadruples | `VERIFIED` |
| 4 | Corollary 4.2: early/late expected losses cross under all assumptions | `5/5` targets crossed at steps `1812–1943` | `VERIFIED` |
| 5 | Section 5/Figure 7: optimization acceleration, not a materially smaller gap | gap equivalent; downscaled run did not accelerate; exact falsification unavailable | `BLOCKED` |
| 6 | Figure 3: lower ViT and GPT-2 condition numbers | `24/24` paired GLU wins | `VERIFIED` scoped |

## Exact executable

Current verifier:
[`reproduction/run_suite.py`](../../reproduction/run_suite.py). Independent
checkers:
[`checker.py`](../../reproduction/checker.py),
[`loss_crossing_checker.py`](../../reproduction/loss_crossing_checker.py),
[`claim5_audit_checker.py`](../../reproduction/claim5_audit_checker.py), and
[`architecture_ntk_checker.py`](../../reproduction/architecture_ntk_checker.py).

```bash
uv sync --frozen --no-dev && uv run python -m reproduction.run_suite
```

Environment: repository-level [`pyproject.toml`](../../pyproject.toml) and
[`uv.lock`](../../uv.lock), Python 3.12, exactly one `.venv`. Every verifier
exits nonzero when its accepted evidence, provenance, or negative control
fails. A successful cumulative exit reproduces Claim 5’s `BLOCKED`
classification; it does not call that claim verified.

## Raw evidence

- [Structured six-claim results](../../evidence/campaign_results.json)
- [Independent checker output and controls](../../evidence/checker_output.json)
- [Compute allocations and runtimes](../../evidence/compute.json)
- [Paper source and exact anchors](../../evidence/source_audit.md)
- [Exact commands](../../evidence/commands.md)
- [Illustrated implementation-led report](../../reports/reproduction/report.md)

The final cumulative run is
`16f0b1b5-a0e8-40a8-b15c-b6352de9f2da` at Git SHA
`9e8f80607b38ffe350cd0112b46518d2f365c668`: `EVAL_STATUS=PASS`,
runtime `90.821 s`, estimated/allocated eight HF `cpu-upgrade` cores. No GPU
was used.

## Historical safety

The old [Historical rejected baseline](#/overview) is preserved unchanged. It
is not the current verifier: its loss-crossing experiment reported
`crossing=False` at `n=40`, outside the source assumption `n>=300`, and it
deferred real ViT/GPT-2 NTKs. The current pages and code above supersede it.
