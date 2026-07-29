# Release command log

This log records the reproducibility- and release-relevant commands. Routine
read-only `sed`, `rg`, `git diff`, and log-window inspection commands do not
alter scientific behavior and are not represented as experiments.

## Startup and source audit

```bash
orx skill
orx skill orx-experiment-tree
orx skill orx-evidence
orx skill orx-git
orx skill orx-compute
orx projects --json
orx project view 95c6d7fc-9739-4f03-9bc3-90ee31fa0020
orx runs 95c6d7fc-9739-4f03-9bc3-90ee31fa0020
git rev-parse HEAD
git status --short
df -h .
```

The paper HTML was fetched from
`https://ar5iv.labs.arxiv.org/html/2605.20749` with an explicit
`OpenResearch-Reproduction/1.0` User-Agent and SHA-256 checked. The exact
judged Space was downloaded with:

```bash
hf download DineshAI/w0JhOFWPJl --repo-type space --revision a9288ab5d4762defa1c9f49f8a58aa27377f5d7d --local-dir <fresh-empty-dir> --max-workers 4
```

## Fixed scientific command

Every formal node ran:

```bash
uv sync --frozen --no-dev && uv run python -m reproduction.run_suite
```

Every multi-core or uncertain run was launched through the project’s default
HF backend, with this form:

```bash
orx exp run <experiment-id> --flavor cpu-upgrade --image ghcr.io/astral-sh/uv:python3.12-bookworm-slim --timeout <declared-bound>
orx exp wait <experiment-id> --timeout 480
orx logs <run-id> --bytes 200000
```

Behavioral variants were committed in code; no experimental knob varied in
the command or environment.

## Presentation validation

```bash
marimo check --fix notebooks/glu_condition_numbers.py
marimo check --strict notebooks/glu_condition_numbers.py
xmllint --noout reports/reproduction/images/*.svg
```

## Publication commands

After the manifests, subset check, secret scan, cumulative release run, and
blind traversal pass, the prepared text-only staging directory is uploaded in
one additive commit:

```bash
hf upload DineshAI/w0JhOFWPJl <text-only-staging-dir> . --repo-type space --commit-message "Publish claim-by-claim GLU reproduction evidence"
hf download DineshAI/w0JhOFWPJl --repo-type space --revision <published-revision> --local-dir <fresh-empty-dir> --max-workers 4
git push origin main
git ls-remote origin refs/heads/main
```

No token, credential value, generated wrapper, or secret is recorded.
