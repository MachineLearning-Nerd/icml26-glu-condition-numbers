# Claim 5 evaluation

Run with the project-wide fixed command:

`uv sync --frozen --no-dev && uv run python -m reproduction.run_suite`

The run prints raw per-seed/per-epoch losses, the full structured evidence,
independent checker output, negative-control result, CPU allocation, Git SHA,
and runtime. The checker exits nonzero unless Claim 5 is VERIFIED or validly
FALSIFIED and every cumulative regression passes.

This page is a committed protocol, not a result. Raw output and the final
verdict are populated only from the OpenResearch run log.
