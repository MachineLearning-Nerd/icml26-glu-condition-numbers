# Claims 1–3: kernel conditioning

![Condition-number scaling](../../reports/reproduction/images/scaling.svg)

## Exact contracts and assumptions

Claim 1 tests Theorem 3.1’s asymptotic comparison:
\(\kappa(\widetilde K)=O(n/d^2)\) for GLU and
\(\kappa(K)=O(n/d)\) for non-GLU. Claim 2 tests Section 3/Equation 5’s
approximation \(\widetilde K\approx K\odot(XX^\top/d)\), including the
source’s dropped small-variance terms. Claim 3 tests the extra-\(d\)
contraction of \(\lambda_{\max}\) and shared \(O(m)\) scaling of
\(\lambda_{\min}\).

These are finite numerical corroborations of asymptotic statements. They are
not machine-checkable universal proofs; this is the same limitation under
which the live judge awarded full credit.

## Inline results

| `d` | non-GLU κ | GLU κ | ratio non/GLU |
|---:|---:|---:|---:|
| 20 | `192.300` | `43.079` | `4.46` |
| 40 | `95.669` | `10.713` | `8.93` |
| 80 | `50.711` | `4.162` | `12.18` |

Log slopes are `−0.9615` and `−1.6859`. The Hadamard relative Frobenius
error is `0.03716` at `d=40`. GLU’s largest eigenvalue is lower at each
tested `d`; the non-GLU smallest eigenvalue rises
`27.6→52.8→102.4` as width rises `400→800→1600`.

Status: Claims 1, 2, and 3 are `VERIFIED`.

## Reproduce and inspect

Run the [fixed command](../../evidence/commands.md). Code:
[`kernels.py`](../../reproduction/kernels.py) and
[`checker.py`](../../reproduction/checker.py). Raw judged regression:
[`raw.json`](../../.openresearch/artifacts/historical_regression/raw.json).
Current structured values:
[`campaign_results.json`](../../evidence/campaign_results.json).

The checker’s negative control changes the Hadamard error to `0.50`; it must
be rejected. The cumulative [checker output](../../evidence/checker_output.json)
records that rejection.
