# Evaluation

- Claims 1-3: historical numerical evidence retained for cumulative regression.
- Claim 4: the historical `crossing=false` output is retained, but its
  assumption audit fails because `n=40<300`. It is not the current verifier.
- Claims 5-6: absent from this frozen baseline by design.
- Verifier: `uv sync --frozen --no-dev && uv run python -m reproduction.run_suite`.
- Expected compute: one CPU core and under five minutes after environment setup.
- Environment setup is uncertain on a fresh worker, so the first formal run is
  routed to Hugging Face `cpu-upgrade`.

