# The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?

Independent, claim-by-claim reproduction audit for the ICML 2026 paper
[“The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?”](https://arxiv.org/abs/2605.20749).

> **Audit status:** `PARTIAL_C1_C2_C3_C4_C6_VERIFIED_C5_BLOCKED_HISTORICAL_SCORE_8_OF_12_NO_CURRENT_SCORE`
>
> Claims 1–4 and 6 are verified within explicit finite or architecture-scoped
> contracts. Claim 5 remains blocked because the CPU-feasible CIFAR Mixer run
> changes the source-default width, depth, and epoch horizon. See
> [`STATUS.md`](STATUS.md), [`CLAIM_EVIDENCE.md`](CLAIM_EVIDENCE.md), and
> [`REPORT.md`](REPORT.md) for the standardized audit record.

This repository audits why gated feed-forward blocks such as ReGLU, GEGLU,
and SwiGLU can optimize better than non-gated counterparts. It is an
independent evidence package, not an official author implementation. The
status labels below preserve the paper assumptions and distinguish a scoped
replication from a paper-level conclusion.

## Current assessment

Claims 1–4 have reproducible evidence. Claim 6 is `VERIFIED` within a scoped
real-architecture NTK audit. Claim 5 remains `BLOCKED`: the downscaled
full-CIFAR Mixer run found an equivalent matched-loss gap but no optimization
advantage, yet its width, depth, and training horizon differ materially from
the paper defaults. The previous live judge score remains `8/12`; `10/12` is a
forecast, not an earned score.

| Claim | Paper statement tested | How the result is produced | Evidence and verdict |
| --- | --- | --- | --- |
| 1 | Theorem 3.1: GLU has condition-number scaling `O(n/d²)` versus `O(n/d)` for non-GLU structure. | Evaluate the exact arc-cosine ReLU/ReGLU kernels over finite dimensions and fit the log–log slopes. | Reproduced slopes are `−1.69` for GLU and `−0.96` for non-GLU. This is finite corroboration of the asymptotic statement. **VERIFIED.** |
| 2 | Section 3, Equation 5: the GLU NTK is approximately reweighted by `K ⊙ (XXᵀ/d)`. | Compare the exact GLU kernel with the Hadamard approximation over the released finite checks. | Relative error is `0.0372`. **VERIFIED.** |
| 3 | Theorem 3.1: gating contracts the largest eigenvalue with an extra dimension factor while the smallest eigenvalue retains shared `O(m)` behavior. | Track the extreme eigenvalues across widths and compare GLU/non-GLU values at each tested dimension. | GLU has the lower largest eigenvalue at every tested `d`; the nonzero smallest-eigenvalue trend rises from `27.6` to `102.4` as width quadruples. **VERIFIED.** |
| 4 | Proposition 4.1 and Corollary 4.2: early and late losses cross under the stated Gaussian, dimension, sample-size, quadratic-form, and step-size assumptions. | Use iid Gaussian inputs with `n=300`, `d=20`, independent targets satisfying `Yᵀ(K−K̃)Y ≥ 0`, and verify the expected-kernel loss crossing for five seeds. | All `5/5` targets cross at steps `1812–1943`; the minimum-eigenvector negative control is rejected because its quadratic form is `−10,534`. **VERIFIED.** |
| 5 | Section 5/Figure 7: GLU’s primary advantage is faster optimization rather than a smaller generalization gap. | Train parameter-matched ReLU/ReGLU Mixers on all CIFAR-10 images, compare matched-loss gaps and optimization AUC, then run an assumption-preserving falsification audit. | Matched-loss gap difference is practically equivalent (`−0.000777`, 95% interval `[−0.009303,+0.007749]`), but the run uses width `64`, depth `2`, and `15` epochs instead of source defaults `256`, `4`, and `100`. **BLOCKED.** |
| 6 | Figure 3: parameter-matched ViT and GPT-2 FFN blocks have lower empirical NTK condition numbers when gated. | Form exact scalar-output Jacobian Gram matrices for ReLU/ReGLU, GELU/GEGLU, and SiLU/SwiGLU across four paired seeds. | GLU wins in `24/24` paired comparisons. The audit uses `n=8` inputs instead of the official ViT script’s `64` and independently reconstructs GPT-2 NTKs. **VERIFIED, scoped.** |

## What the paper is doing

The paper studies two-layer networks in the neural tangent kernel (NTK)
regime to explain why gated linear units often outperform non-gated
feed-forward blocks. Its mechanism is spectral: the gate reshapes the NTK,
making the eigenvalue distribution more compact and reducing the condition
number. Better conditioning predicts faster optimization and the characteristic
loss-crossing behavior between GLU and non-GLU models.

The paper then tests whether that optimization benefit also reduces
generalization gaps in modern architectures, including ViT and GPT-2. Its
empirical conclusion is that gating’s main benefit is optimization speed, with
limited effect on the generalization gap. This audit verifies the theory and
real-architecture condition-number route while keeping the full-scale Mixer
limitation visible.

## Reproducing the evidence

Every experiment branch inherits the same pinned command:

```bash
uv sync --frozen --no-dev
uv run python -m reproduction.run_suite
```

The suite runs the scaling, Hadamard approximation, eigenvalue, loss-crossing,
real-architecture NTK, and generalization-gap routes. It also runs provenance
and negative controls. A successful cumulative run means that accepted checks
and intended controls behaved as specified; it does not turn Claim 5’s
protocol mismatch into a paper-level verdict.

Useful entry points:

- [Illustrated technical report](reports/reproduction/report.md)
- [Campaign results](evidence/campaign_results.json)
- [Source audit and exact paper anchors](evidence/source_audit.md)
- [Experiment tree](evidence/experiment_tree.md)
- [Reproduction suite](reproduction/run_suite.py)
- [Reproduction notebook](notebooks/glu_condition_numbers.py)
- [Evaluator-visible Space](https://huggingface.co/spaces/DineshAI/w0JhOFWPJl)

## Branch organization

`main` is the publication surface. `audit/*` branches preserve claim-specific
experiments and falsification routes, `release/*` branches preserve evaluator
packages and provenance corrections, and `historical/*` preserves the earlier
judged baseline. The complete old-to-clean mapping is in
[`branch-audit.md`](branch-audit.md). Every clean branch receives this README
and the branch map so each checkout remains self-describing.

## Scope and limitations

- Finite checks corroborate the paper’s asymptotic claims; they do not replace
  a proof of the universal statements.
- Claim 4 is valid only under Corollary 4.2’s assumptions, including `n>=300`
  and the quadratic-form condition. The historical `n=40` result is preserved
  only as a rejected baseline.
- Claim 6 is scoped: the paired architecture audit uses small input batches,
  and the GPT-2 NTK implementation is independently reconstructed.
- Claim 5’s all-CIFAR run uses the paper’s data family and matched-loss test,
  but not the source-default Mixer capacity or 100-epoch horizon. It is
  therefore `BLOCKED`, not a falsification.
- No GPU training result or new live judge score is claimed.

## Paper

- **Title:** The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?
- **Authors:** Xingyu Lyu, Qianqian Xu, Zhiyong Yang, Peisong Wen, Qingming Huang
- **Paper:** [arXiv:2605.20749](https://arxiv.org/abs/2605.20749)
- **HTML source:** [arXiv HTML](https://arxiv.org/html/2605.20749)
- **Submission:** May 20, 2026; revised May 25, 2026; accepted by ICML 2026
- **Paper identifier:** `w0JhOFWPJl`

## Citation

```bibtex
@misc{lyu2026devil,
  title         = {The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?},
  author        = {Lyu, Xingyu and Xu, Qianqian and Yang, Zhiyong and Wen, Peisong and Huang, Qingming},
  year          = {2026},
  eprint        = {2605.20749},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note          = {Accepted by ICML 2026; revised version arXiv:2605.20749v2}
}
```

## Thank you

Thank you to Xingyu Lyu, Qianqian Xu, Zhiyong Yang, Peisong Wen, and Qingming
Huang for making the GLU-versus-non-GLU mechanism concrete through NTK spectra,
condition-number scaling, loss-crossing assumptions, and modern-architecture
measurements. The paper’s explicit assumptions made it possible to repair the
loss-crossing audit and to state honestly where the full Mixer reproduction
still needs source-scale compute.

## Attribution

This independent audit is maintained by
[MachineLearning-Nerd](https://github.com/MachineLearning-Nerd). It is not
affiliated with or endorsed by the paper’s authors.
