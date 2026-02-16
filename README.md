# dcode_app_factory

`dcode_app_factory` is a deterministic skeleton of the AI Software Product
Factory described in `SPEC.md`.

It currently demonstrates:

1. Product loop: creates a structured spec placeholder.
2. Project loop: flattens tasks and dispatches them sequentially.
3. Engineering loop: runs a propose/challenge/adjudicate flow and registers module contracts.

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

Run with auto-loaded `SPEC.md` (if present):

```bash
uv run python scripts/factory_main.py
```

Run with an explicit spec path:

```bash
uv run python scripts/factory_main.py --spec-file SPEC.md
```

The script prints:

1. `project_success=<true|false>`
2. Registered micro-module contracts from the in-memory Code Index

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

## Repository Layout

- `dcode_app_factory/`: package modules for models, loops, debate, registry, and utilities
- `scripts/factory_main.py`: CLI entrypoint
- `tests/`: pytest suite
- `SPEC.md`: detailed target specification
- `tasks.md`: living task list
- `AGENTS.md`: repository instructions for coding agents
- `setup_script.md`: copy-paste setup script for Codex cloud UI
- `.gitignore`: ignore rules for local environments, caches, and test artifacts

## Notes

1. The implementation is intentionally minimal and does not yet implement the full architecture in `SPEC.md`.
2. The Code Index is currently in-memory and not persisted.
3. Debate behavior is deterministic and simplified.
