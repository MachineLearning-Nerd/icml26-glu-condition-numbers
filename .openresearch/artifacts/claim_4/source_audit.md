# Claim 4 source audit

Proposition 4.1 (`#S4.Thmtheorem1`) gives the expected kernel-regime loss

`Tr[(I-eta K)^(2k)K] + Y^T(I-eta K)^(2k)Y`.

Corollary 4.2 (`#S4.Thmtheorem2`) states that for two-layer ReLU/ReGLU
models with Gaussian inputs:

1. Early, when `eta*k` is small, ReLU has smaller expected loss provided
   `Y^T(K-K_tilde)Y>=0`, `d>=5`, and `n>=300`.
2. Sufficiently late, ReGLU has smaller expected loss.

The test uses one common stable `eta`, unlike the historical rejected
baseline, which independently normalized each kernel and used `n=40`.

