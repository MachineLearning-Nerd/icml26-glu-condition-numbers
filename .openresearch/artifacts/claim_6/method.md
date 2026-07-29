# Claim 6 method

For each architecture, activation pair, and four predeclared seeds, compute
per-example gradients of a scalar model output with respect to every FFN
parameter. The exact empirical NTK is `J J^T`; its condition number is the ratio
of the largest to smallest eigenvalue.

Gated hidden widths are reduced using the official parameter-matching rule, and
the checker requires paired FFN parameter counts within `0.5%` (integer hidden
widths prevent exact equality in the official ViT configuration). All
attention and normalization parameters remain in the forward architecture but
are held fixed so the measurement isolates the FFN structure changed by the
claim.

The negative control compares a ReLU ViT measurement with itself. Its ratio is
exactly one and must not be accepted as a GLU reduction.
