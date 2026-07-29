# Claim 5 evaluation

Run with the project-wide fixed command:

`uv sync --frozen --no-dev && uv run python -m reproduction.run_suite`

The answered run printed raw per-seed/per-epoch losses, the full structured
evidence, independent checker output, negative-control result, CPU allocation,
Git SHA, and runtime. Its deliberate nonzero exit recorded Claim 5 as BLOCKED.

This page is a committed protocol, not a result. Raw output and the final
verdict are populated only from the OpenResearch run log.

Attempt 1 run `729a1f23-04e2-49c3-a1c9-165f04aa8741` was cancelled after
25m24s: the default-size model did not complete its first full-dataset
evaluation. This is an explicit resource-design failure, not claim evidence.

Attempt 2's initial launch `59f54833-e236-48e4-aa0d-65000d36f98d` was
cancelled before any `GAP_PROGRESS` line after the same silent torchvision
download stall. The subsequent downloader-only repair retains the full
dataset and verifies its authoritative SHA-256 before training.

The repaired answered run is
`822e0d9a-9b4a-4f90-9f40-32413a073ef4` at Git SHA
`e0fe32a3bd26a0732b42afbf39eedc112263c7c9`. It used HF `cpu-upgrade`,
requested eight cores, received an eight-CPU cgroup quota, and ran for
7,845.397 seconds (Claim 5: 7,752.818 seconds). The complete 96-row trajectory
is in `cifar10_mixer_raw.csv`.

Observed ReGLU-minus-ReLU matched-loss seed means were `-0.0000490`,
`-0.0045145`, and `+0.0022332` CE; mean `-0.0007768`, 95% t interval
`[-0.0093029,+0.0077493]`. The generalization-gap component is practically
equivalent. ReGLU/ReLU loss-AUC ratios were `1.04828`, `1.01452`, and
`1.00526`; optimization acceleration was supported in zero of three seeds.

Attempts 3 and 4 are executable in `reproduction/claim5_audit.py`, independently
checked by `reproduction/claim5_audit_checker.py`. The audit checker exits
nonzero if raw data are altered, equivalence fails, or the downscaled adverse
observation is incorrectly promoted to falsification. The expected final
scientific result is exactly `BLOCKED`; a successful audit exit means that
blocked classification and its controls were reproduced, not that Claim 5
passed.
