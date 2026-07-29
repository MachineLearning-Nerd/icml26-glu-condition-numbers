# Source audit

- Paper: arXiv `2605.20749`.
- Retrieved URL: `https://ar5iv.labs.arxiv.org/html/2605.20749`.
- Retrieval UTC: `2026-07-29T07:48:03Z`.
- SHA-256: `d391f59973c02be99395f55c770d31c8728c5bdc4373244b59c4597dad246e4f`.
- Theorem 3.1 anchor: HTML `#S3.Thmtheorem1`.
- Equation 5 anchor: HTML `#S3.E5`.
- Proposition 4.1 anchor: HTML `#S4.Thmtheorem1`.
- Corollary 4.2 anchor: HTML `#S4.Thmtheorem2`.

Theorem 3.1 assumes two-layer ReLU/ReGLU models under LeCun initialization,
i.i.d. standard-Gaussian inputs, and `d+1<n`, and concerns limiting spectral
distributions. Corollary 4.2 assumes Gaussian inputs; its early-stage statement
also requires `d>=5`, `n>=300`, and
`Y^T(K-K_tilde)Y>=0`. Its late-stage statement is for sufficiently large `k`.

The judged `n=d=40` protocol does not satisfy either `d+1<n` or `n>=300`.

