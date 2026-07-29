# Claim 6 evaluation

The formal CPU run prints every seed-level eigenvalue and condition number,
paired log-ratio confidence intervals, parameter-count audits, checker output,
negative-control output, CPU allocation, and runtime.

The architecture is full-depth and full-width, but the Jacobian sample count is
reduced to eight for CPU feasibility. This is a material deviation from the
official ViT script's `N=64` and is disclosed rather than called full-scale.

