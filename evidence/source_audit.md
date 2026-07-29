# Paper source and exact anchors

The HTML source was retrieved from
`https://ar5iv.labs.arxiv.org/html/2605.20749` with an explicit browser
User-Agent at `2026-07-29T07:48:03Z`. SHA-256:
`d391f59973c02be99395f55c770d31c8728c5bdc4373244b59c4597dad246e4f`.

- Claim 1: Theorem 3.1, asymptotic condition numbers
  `O(n/d^2)` versus `O(n/d)`.
- Claim 2: Section 3, Equation 5, the approximate Hadamard reweighting.
- Claim 3: Section 3 and Theorem 3.1, largest- and smallest-eigenvalue scaling.
- Claim 4: Proposition 4.1 and Corollary 4.2. Inputs are iid Gaussian;
  `d>=5`, `n>=300`, `d+1<n`, `Y^T(K-K_tilde)Y>=0`, small early
  `eta*k`, and sufficiently large late integer `k`.
- Claim 5: Section 5 and Figure 7. The source says joint
  `(training loss, generalization gap)` distributions overlap and reports
  energy-permutation `p>=0.05`; Figure 7(a) is MLP-Mixer/CIFAR-10.
  Appendix D specifies SGD at learning rate `0.005` without extra
  regularization. “Limited” has no source effect-size threshold.
- Claim 6: Figure 3, empirical condition numbers for parameter-matched ViT and
  GPT-2 GLU/non-GLU structures.

A finite experiment does not prove a universally quantified asymptotic
theorem. Claims 1–3 are retained because the live judge already credited their
finite, reproducible corroboration; their limitation remains explicit.
