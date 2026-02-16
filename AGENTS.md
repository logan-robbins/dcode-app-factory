# Repository Agent Guide

These instructions are for coding agents working in this repository.

## Environment

1. Use Python 3.12.
2. Use `uv` for all dependency and execution workflows.
3. Do not use `pip` or manually managed virtual environments.

## Standard Commands

1. Sync dependencies: `uv sync --all-groups --frozen`
2. Run tests: `uv run pytest -q`
3. Run the app: `uv run python scripts/factory_main.py`
4. Run the app with explicit spec input: `uv run python scripts/factory_main.py --spec-file SPEC.md`

## Working Rules

1. Prefer extending existing modules in `dcode_app_factory/` over creating parallel implementations.
2. Fail fast with clear errors when required inputs are missing.
3. Keep docs accurate: if command paths or behaviors change, update `README.md` and `tasks.md` in the same change.
4. Preserve deterministic behavior in loops and tests.
