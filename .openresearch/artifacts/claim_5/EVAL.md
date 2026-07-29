# Claim 5 evaluation

Run with the project-wide fixed command:

`uv sync --frozen --no-dev && uv run python -m reproduction.run_suite`

The run prints raw per-seed/per-epoch losses, the full structured evidence,
independent checker output, negative-control result, CPU allocation, Git SHA,
and runtime. The checker exits nonzero unless Claim 5 is VERIFIED or validly
FALSIFIED and every cumulative regression passes.

This page is a committed protocol, not a result. Raw output and the final
verdict are populated only from the OpenResearch run log.

Attempt 1 run `729a1f23-04e2-49c3-a1c9-165f04aa8741` was cancelled after
24m48s: the default-size model did not complete its first full-dataset
evaluation. This is an explicit resource-design failure, not claim evidence.

Attempt 2's initial launch `59f54833-e236-48e4-aa0d-65000d36f98d` was
cancelled before any `GAP_PROGRESS` line after the same silent torchvision
download stall. The subsequent downloader-only repair retains the full
dataset and verifies its authoritative SHA-256 before training.
