# Exact commands and revisions

Every experiment node inherited this command unchanged:

```bash
uv sync --frozen --no-dev && uv run python -m reproduction.run_suite
```

The environment is Python 3.12 with the repository-level `pyproject.toml`,
`uv.lock`, and one `.venv` created by `uv`. The final cumulative run used Git
SHA `9e8f80607b38ffe350cd0112b46518d2f365c668`.

Formal runs were launched through:

```bash
orx exp run <experiment-id> --flavor cpu-upgrade --image ghcr.io/astral-sh/uv:python3.12-bookworm-slim --timeout <bound>
```

The placeholder is orchestration metadata only. Scientific behavior never
varied through the command or environment; variants were committed in code.
