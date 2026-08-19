# Audit status

**State:** Claims 1–4 and 6 have scoped reproducible evidence; Claim 5 remains
blocked by a material source-protocol mismatch.

- Paper: [The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?](https://arxiv.org/abs/2605.20749)
- Authors: Xingyu Lyu, Qianqian Xu, Zhiyong Yang, Peisong Wen, and Qingming Huang
- ICML submission: `w0JhOFWPJl`
- Repository: [MachineLearning-Nerd/icml26-glu-condition-numbers](https://github.com/MachineLearning-Nerd/icml26-glu-condition-numbers)
- Overall status: `PARTIAL_C1_C2_C3_C4_C6_VERIFIED_C5_BLOCKED_HISTORICAL_SCORE_8_OF_12_NO_CURRENT_SCORE`
- C1–C3: `VERIFIED_SCOPED` by finite kernel scaling, Hadamard, and eigenvalue audits
- C4: `VERIFIED_SCOPED` under the exact Gaussian, dimension, sample-size, quadratic-form, and step-size assumptions
- C5: `BLOCKED_PROTOCOL` because the CPU run uses width `64`, depth `2`, and 15 epochs instead of the source-default width `256`, depth `4`, and 100 epochs
- C6: `VERIFIED_SCOPED` with `24/24` paired real-architecture NTK wins, using `n=8` inputs and an independently reconstructed GPT-2 route
- Historical external score: `8/12`
- Current score claim: `false`
- Publication allowed: `false`
- Official author endorsement: `false` / not claimed
- Commit identity: all reachable history uses `MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>`
- Recovery bundle SHA-256: `50630d4b59cb128b7c500a0ad9c069c80098d8cb0904df3468725315384c9f49`

The inherited `10/12` estimate is a forecast only. The current audit does not
claim a new judge score or a source-scale Claim 5 reproduction.
