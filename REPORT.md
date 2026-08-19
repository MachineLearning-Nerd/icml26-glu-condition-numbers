# Audit report

## Executive result

Claims 1–4 and 6 have reproducible scoped evidence. Claim 5’s matched-loss
component is supported, but its optimization conclusion remains blocked because
the CPU-feasible Mixer run does not use the paper’s source-default capacity or
100-epoch horizon. The only score claimed is the historical external result
`8/12`.

Overall status:

`PARTIAL_C1_C2_C3_C4_C6_VERIFIED_C5_BLOCKED_HISTORICAL_SCORE_8_OF_12_NO_CURRENT_SCORE`

## Claim matrix

| Claim | Result | Primary route | Main boundary |
| --- | --- | --- | --- |
| C1 | `VERIFIED_SCOPED` | Arc-cosine kernel slopes | Finite corroboration of an asymptotic statement. |
| C2 | `VERIFIED_SCOPED` | Hadamard approximation | Approximation drops small variance terms. |
| C3 | `VERIFIED_SCOPED` | Extreme eigenvalue trends | Tested finite widths only. |
| C4 | `VERIFIED_SCOPED` | Exact Corollary 4.2 loss crossing | Only under every stated Gaussian, dimension, sample, quadratic-form, and step assumption. |
| C5 | `BLOCKED_PROTOCOL` | Full-CIFAR matched-loss and four-route audit | Width/depth/epoch mismatch prevents a paper-level optimization verdict. |
| C6 | `VERIFIED_SCOPED` | ViT/GPT-2 real-architecture NTKs | `n=8` inputs and independently reconstructed GPT-2 path. |

## Quantitative evidence

- C1 slopes: GLU `−1.69`, non-GLU `−0.96`.
- C2 relative Frobenius error: `0.0372`.
- C3: GLU has lower largest eigenvalue at every tested dimension; nonzero
  smallest eigenvalues rise `27.6→102.4` as width quadruples.
- C4: five of five targets cross at steps `1812–1943`; the negative control has
  quadratic form `−10,534`.
- C5: matched-loss gap difference `−0.000777`, 95% interval
  `[−0.009303,+0.007749]`; source-default protocol remains unavailable.
- C6: GLU wins `24/24` paired comparisons; geometric-mean ratios are recorded
  in `evidence/campaign_results.json`.

## Score and publication boundary

- Historical live score: `8/12`
- Current score claim: `false`
- Publication allowed: `false`
- Official author endorsement: `false` / not claimed

Open [`CLAIM_EVIDENCE.md`](CLAIM_EVIDENCE.md) for production paths and
controls, and [`SOURCE_AUDIT.md`](SOURCE_AUDIT.md) for source/version scope.
