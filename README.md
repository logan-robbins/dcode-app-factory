# dcode-app-factory

Contract-first implementation of an AI software product factory with three deterministic loops:

- **Product Loop**: converts a markdown spec into a structured hierarchy (pillar/epic/story/task).
- **Project Loop**: validates dependency DAG and dispatches one task at a time.
- **Engineering Loop**: executes Propose → Challenge → Adjudicate and only ships on PASS.

## State-of-the-art context engineering updates

- Per-agent configuration files are defined for each stage under `agent_configs/`.
- Every agent role has explicit context-window limits and an allowed-context policy.
- Engineering debate uses a task-scoped `ContextPack` with explicit allow/deny file patterns.
- Task execution is dependency-aware and halts deterministically on failed adjudication.

## Requirements

1. Python 3.12+
2. `uv`

## Setup

```bash
uv sync --all-groups --frozen
```

If you are initializing locally without an existing lock environment, use:

```bash
uv sync --all-groups
```

## Run

```bash
uv run python scripts/factory_main.py
```

Or with an explicit spec path:

```bash
uv run python scripts/factory_main.py --spec-file SPEC.md
```

## Test

```bash
uv run pytest -q
```

## Codex Cloud Environment

Use these settings in Codex cloud environments:

1. Base image: `codex-universal`
2. Environment variables:
   - Required: `CODEX_ENV_PYTHON_VERSION=3.12`
   - Optional: `PYTHONUNBUFFERED=1`
3. Secrets:
   - Required: none for this repository as currently implemented
   - Add only if your fork adds private package registries or external APIs
4. Setup script: contents of `setup_script.md`
5. Maintenance script:

```bash
#!/usr/bin/env bash
set -euxo pipefail
uv sync --all-groups --frozen
```

## Layout

- `dcode_app_factory/models.py`: contracts, spec hierarchy, context pack, agent config models
- `dcode_app_factory/loops.py`: product/project/engineering loop orchestration
- `dcode_app_factory/debate.py`: 3-agent debate protocol and trace artifact
- `dcode_app_factory/registry.py`: append-only in-memory code index with contract fingerprints
- `agent_configs/*/*.json`: agent runtime configs for Product, Project, Engineering stages
- `scripts/factory_main.py`: CLI entrypoint
- `tests/`: pytest suite
- `AGENTS.md`: repository instructions for coding agents
- `setup_script.md`: copy-paste setup script for Codex cloud UI
- `.gitignore`: ignore rules for local environments, caches, and test artifacts
