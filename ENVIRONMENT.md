# Environment and reproduction boundary

## Locked command

Run the cumulative suite from `main` with:

```bash
uv sync --frozen --no-dev && uv run python -m reproduction.run_suite
```

The project requires Python `3.12` and pins dependencies in [`pyproject.toml`](pyproject.toml)
and [`uv.lock`](uv.lock). The suite is CPU-only for the reported evidence; the
long CIFAR route used an eight-CPU `cpu-upgrade` allocation and no GPU.

## What the suite checks

- finite GLU/non-GLU kernel slopes, Hadamard approximation, and eigenvalues;
- exact-assumption Gaussian loss crossing at `n=300,d=20` with five seeds;
- full-CIFAR matched-loss and optimization diagnostics plus Claim 5 controls;
- real ViT and GPT-2 FFN Jacobian Gram condition numbers;
- historical fixture integrity and an invalid control for each consequential
  route.

## Runtime boundary

The final cumulative verifier took about `90.82 s`; the full-CIFAR run took
`7,845.40 s`. Claim 5’s run used width `64`, depth `2`, and 15 epochs instead
of source defaults width `256`, depth `4`, and 100 epochs. That mismatch is why
the result is `BLOCKED_PROTOCOL`, not a paper-level verification or falsification.
