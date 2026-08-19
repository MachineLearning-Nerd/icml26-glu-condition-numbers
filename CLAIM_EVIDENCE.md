# Claim-to-evidence ledger

This repository audits six source-anchored claims from *The Devil is in the
Condition Numbers: Why is GLU Better than non-GLU Structure?* The production
path and evidence boundary for each claim are explicit below.

| Claim | Paper anchor | How the result is produced | Evidence and control | Scope and status |
| --- | --- | --- | --- | --- |
| C1 — condition-number scaling | Theorem 3.1 | Evaluate exact arc-cosine ReLU/ReGLU kernels at `d=20,40,80` and fit log–log slopes. | `evidence/campaign_results.json`, `reproduction/kernels.py`, and `reproduction/checker.py`; historical Hadamard corruption is rejected by the cumulative checker. | Finite corroboration of the asymptotic `O(n/d²)` versus `O(n/d)` statement. **VERIFIED_SCOPED** |
| C2 — Hadamard NTK approximation | Section 3, Equation 5 | Compare the exact GLU kernel with `K ⊙ (XXᵀ/d)` at the released finite check. | `evidence/campaign_results.json` and `reproduction/checker.py`; the corrupted relative-error fixture is rejected. | Approximation at the tested finite dimension; dropped variance terms remain a limitation. **VERIFIED_SCOPED** |
| C3 — extreme-eigenvalue behavior | Theorem 3.1 | Track largest and smallest eigenvalues across widths and compare GLU/non-GLU values. | `evidence/campaign_results.json`, `reproduction/kernels.py`, and checker output. | Finite-width trend, not a universal asymptotic proof. **VERIFIED_SCOPED** |
| C4 — loss crossing | Proposition 4.1 and Corollary 4.2 | Generate iid Gaussian inputs at `n=300,d=20`, enforce `Yᵀ(K−K̃)Y≥0`, search the common-step loss grid, and test five seeds. | `evidence/campaign_results.json`, `reproduction/loss_crossing.py`, and `.openresearch/artifacts/claim_4`; minimum-eigenvector control has quadratic form `−10,534` and is rejected. | Verified for the exact displayed assumptions; the historical `n=40` check is preserved only as rejected baseline. **VERIFIED_SCOPED** |
| C5 — optimization versus generalization gap | Section 5 and Figure 7 | Run full-CIFAR matched-loss and optimization routes, then run an independent equivalence/falsification dossier. | `evidence/campaign_results.json`, `.openresearch/artifacts/claim_5`, and `reproduction/claim5_audit.py`; the current route has four completed audits and a source-default mismatch. | Gap component is supported, but Claim 5 is blocked because width/depth/epochs differ from the paper. **BLOCKED_PROTOCOL** |
| C6 — real-architecture NTK conditioning | Figure 3 | Form scalar-output Jacobian Gram matrices for parameter-matched ViT and GPT-2 FFNs across three activation pairs and four paired seeds. | `evidence/campaign_results.json`, `.openresearch/artifacts/claim_6`, and `reproduction/architecture_ntk.py`; identical-ratio control is rejected. | `24/24` wins under `n=8` inputs; GPT-2 is independently reconstructed and the official ViT route uses `n=64`. **VERIFIED_SCOPED** |

## Evidence ladder

1. The arXiv HTML source is pinned by SHA-256
   `d391f59973c02be99395f55c770d31c8728c5bdc4373244b59c4597dad246e4f`.
2. Each current route has code, a machine-readable result, a checker, and an
   invalid or assumption-violating control.
3. Finite kernel evidence corroborates asymptotic claims; it does not replace
   their proofs.
4. Claim 5’s matched-loss result is not promoted to a paper-level optimization
   conclusion until the source-default Mixer capacity and 100-epoch horizon are
   reproduced.

The complete path inventory is in [`EVIDENCE_MANIFEST.json`](EVIDENCE_MANIFEST.json).
