# Claim 4: exact-assumption loss crossing

![First crossing step by seed](../../reports/reproduction/images/loss_crossing.svg)

## Exact claim

Proposition 4.1 and Corollary 4.2 state an early/late expected-loss ordering
that produces a crossing under iid Gaussian inputs, `d>=5`, `n>=300`,
`d+1<n`, nonnegative \(Y^\top(K-\widetilde K)Y\), small early
`eta*k`, and sufficiently large late integer `k`.

The current contract uses `n=300`, `d=20`, width scaling `5000`, one common
step size, fixed multi-horizon search through `10,000,000`, and five
independently seeded ReLU-teacher targets.

## Raw result inline

| Seed | Quadratic form | First crossing step | Δ before | Δ after |
|---:|---:|---:|---:|---:|
| 101 | `230,086,597` | `1834` | `−3.553` | `+4.234` |
| 102 | `227,914,228` | `1853` | `−2.154` | `+4.795` |
| 103 | `295,328,788` | `1943` | `−4.370` | `+1.245` |
| 104 | `131,906,079` | `1812` | `−4.442` | `+3.583` |
| 105 | `132,998,539` | `1939` | `−5.695` | `+0.768` |

Status: `VERIFIED`.
Aggregate result: `5/5` independently seeded targets crossed while satisfying
every audited source assumption.

The minimum-eigenvector negative control has
\(Y^\top(K-\widetilde K)Y=-10{,}534\); the independent checker rejects it
for violating the stated sign assumption.

## Code, raw data, and limitation

Code:
[`loss_crossing.py`](../../reproduction/loss_crossing.py) and
[`loss_crossing_checker.py`](../../reproduction/loss_crossing_checker.py).
Raw CSV: [five crossings](../../.openresearch/artifacts/claim_4/raw.csv).
Contract: [`claim_contract.json`](../../.openresearch/artifacts/claim_4/claim_contract.json).
Checker output:
[`checker_output.json`](../../.openresearch/artifacts/claim_4/checker_output.json).
Runtime: [`runtime.json`](../../.openresearch/artifacts/claim_4/runtime.json).

This is an expected-kernel numerical reconstruction, not a proof certificate.
The historical `n=40, crossing=False` page is preserved as
[Historical rejected baseline](#/overview) and is not a valid current
falsification because `n>=300` was false.
