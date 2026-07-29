# Claim 4 method

For five predeclared Gaussian input and teacher seeds, reconstruct the expected
ReLU and ReGLU NTKs from Equations 3 and 4. Diagonalize each kernel once and
evaluate Proposition 4.1 directly. Search a fixed, formula-independent horizon:
all steps `0..256` plus 800 logarithmically spaced integer steps through
`10,000,000`. Binary-search only after the first negative-to-positive bracket is
found.

The checker accepts only independently generated targets satisfying all source
assumptions. The negative control selects the minimum-eigenvalue direction of
`K-K_tilde`, which must be rejected specifically because its quadratic form is
negative.

