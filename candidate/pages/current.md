# Current audit: frozen historical regression baseline

The current executable is
[`reproduction/run_suite.py`](../../reproduction/run_suite.py), with an
independent fixture checker in
[`reproduction/checker.py`](../../reproduction/checker.py). It is invoked by
the fixed command:

```bash
uv sync --frozen --no-dev && uv run python -m reproduction.run_suite
```

## Evidence now visible

| Claim | Current page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Reviewer verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | This page | yes | yes | [JSON](../../.openresearch/artifacts/historical_regression/raw.json) | yes | yes | finite regression of credited values | retained |
| 2 | This page | yes | `0.037` | [JSON](../../.openresearch/artifacts/historical_regression/raw.json) | yes | yes | Eq. 5 approximation at `n=d=40,m=1000` | retained |
| 3 | This page | yes | `27.6,52.8,102.4` | [JSON](../../.openresearch/artifacts/historical_regression/raw.json) | yes | yes | finite width/eigenvalue regression | retained |
| 4 | [Exact contract](../../.openresearch/artifacts/claim_4/claim_contract.json) | yes | pending formal run | pending | yes | yes | Proposition 4.1 and every Corollary 4.2 assumption | pending |
| 5 | — | no | no | no | no | no | not in baseline | BLOCKED pending child |
| 6 | [Exact contract](../../.openresearch/artifacts/claim_6/claim_contract.json) | yes | pending formal run | pending | yes | yes | ViT and GPT-2 FFN NTKs, three activation pairs | pending |

The raw file records the exact numbers visible in the judged Space revision
`a9288ab5d4762defa1c9f49f8a58aa27377f5d7d`. The independent checker is
`reproduction/checker.py`; its negative control corrupts the Eq. 5 relative
error to `0.50` and must be rejected.

## Source and assumptions

The source HTML was retrieved at `2026-07-29T07:48:03Z` and has SHA-256
`d391f59973c02be99395f55c770d31c8728c5bdc4373244b59c4597dad246e4f`.
The exact anchors and quantifiers are in
[source_audit.md](../../.openresearch/artifacts/historical_regression/source_audit.md).

Corollary 4.2 requires `n>=300`; the historical run used `n=40`. Therefore the
old page is preserved as historical evidence but superseded as the default
verification. The current child evaluates the exact expected-loss expression
at `n=300,d=20`, uses one common learning rate, searches a fixed multi-horizon
grid, and audits the quadratic-form assumption for every seed.
