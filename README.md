# dcode-app-factory

Contract-first implementation of an AI software product factory with three deterministic loops:

- **Product Loop**: converts a markdown spec into a structured hierarchy (pillar/epic/story/task).
- **Project Loop**: validates dependency DAG and dispatches one task at a time.
- **Engineering Loop**: executes Propose → Challenge → Adjudicate and only ships on PASS.

## Data flow

```
Raw spec (str) → ProductLoop.run() → StructuredSpec
StructuredSpec + CodeIndex → ProjectLoop.run() → for each task (topological order):
  EngineeringLoop.run() → Debate(propose→challenge→adjudicate, retries) → CodeIndex.register(contract)
```

## State-of-the-art context engineering updates

- Per-agent configuration files are bundled in the package under `src/dcode_app_factory/agent_configs/`.
- Every agent role has explicit context-window limits and an allowed-context policy.
- Engineering debate uses a task-scoped `ContextPack` with explicit allow/deny file patterns.
- Engineering loop records `ShipEvidence` on PASS and `EscalationArtifact` on repeated FAIL.
- Task execution is dependency-aware, and downstream tasks are marked `blocked` after a failed dependency.
- Runtime settings can be configured via environment variables for section fan-out, context budgets, and default spec path.

Supported environment variables:

- `FACTORY_MAX_PRODUCT_SECTIONS` (default `8`) — max number of `##` sections converted into tasks.
- `FACTORY_CONTEXT_BUDGET_FLOOR` (default `2000`) — minimum context budget per invocation.
- `FACTORY_CONTEXT_BUDGET_CAP` (default `16000`) — maximum context budget per invocation.
- `FACTORY_DEFAULT_SPEC_PATH` (default `SPEC.md`) — fallback spec path used when `--spec-file` is not provided.

### Default LLM routing (configurable)

Agent configs provide a `model_tier` (`frontier`, `efficient`, `economy`) and runtime resolves these tiers to concrete model IDs.

Current defaults:

- `frontier` → `openai:gpt-5.2`
- `efficient` → `openai:gpt-5.2-mini`
- `economy` → `openai:gpt-5.2-nano`

Override with environment variables:

- `FACTORY_MODEL_FRONTIER`
- `FACTORY_MODEL_EFFICIENT`
- `FACTORY_MODEL_ECONOMY`
- `FACTORY_MODEL_ROLE_OVERRIDES_JSON` (JSON object with optional keys of `stage.role` or `role`, e.g. `{"engineering_loop.proposer":"anthropic:claude-sonnet-4.5"}`)

Agent config JSON schema (`src/dcode_app_factory/agent_configs/<stage>/<role>.json`):

```json
{
  "stage": "product_loop|project_loop|engineering_loop",
  "role": "string",
  "model_tier": "string",
  "temperature": 0.0,
  "max_context_tokens": 48000,
  "context_policy": "strict_pack|context_pack_backend|...",
  "allowed_context_sections": ["list", "of", "strings"]
}
```

## Requirements

1. Python 3.12+
2. `uv`
3. Runtime dependency: `rfc8785>=0.1.4` (installed via `uv sync`)

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

- `src/dcode_app_factory/`: Python package (src layout)
- `src/dcode_app_factory/models.py`: contracts, spec hierarchy, context pack, agent config models
- `src/dcode_app_factory/loops.py`: product/project/engineering loop orchestration
- `src/dcode_app_factory/debate.py`: 3-agent debate protocol and trace artifact
- `src/dcode_app_factory/registry.py`: append-only in-memory code index with contract fingerprints
- `src/dcode_app_factory/utils.py`: slugify, context pack builder, agent config loader, DAG validation
- `src/dcode_app_factory/agent_configs/*/*.json`: agent runtime configs for Product, Project, Engineering stages
- `agent_configs/release_loop/*.json`: release-stage config artifacts merged from the remote branch (not yet wired into runtime loops)
- `scripts/factory_main.py`: CLI entrypoint
- `tests/`: pytest suite
- `AGENTS.md`: repository instructions for coding agents
- `setup_script.md`: copy-paste setup script for Codex cloud UI
- `.gitignore`: ignore rules for local environments, caches, and test artifacts

## Type hierarchy

- **StructuredSpec** → pillars → epics → stories → tasks
- **Task**: `task_id`, `depends_on`, `io_contract_sketch`, `status`, `contract` (set after EngineeringLoop)
- **MicroModuleContract**: `module_id`, `name`, `fingerprint`, inputs/outputs/error_surfaces
- **ContextPack**: `task_id`, `objective`, `interfaces`, `allowed_files`, `denied_files`
- **AgentConfig**: `stage`, `role`, `max_context_tokens`, `context_policy`, `allowed_context_sections`

## Extension points

- **New agent role**: add `src/dcode_app_factory/agent_configs/<stage>/<role>.json`; loops load all `*.json` in the stage dir.
- **New stage**: add a config dir and pass it to the loop constructor via `config_dir`.
- **Debate roles**: `_proposer`, `_challenger`, `_arbiter` in `EngineeringLoop`; each receives `(prompt, ContextPack)` and returns a string.

## Fail-fast behavior

- `validate_task_dependency_dag(spec)` raises on cycles or unknown dependencies.
- `IOContractSketch.validate_complete()` raises if any field contains placeholders (`tbd`, `todo`, `n/a`, etc.).
- `EngineeringLoop(..., max_retries=<n>)` raises if `max_retries < 0`.
- `ProductLoop.run()` raises with explicit validation details when generated specs contain structural errors.
- `load_raw_spec()` raises if the requested spec file is missing.

## Implementation state

This is a skeleton. The debate uses placeholder implementations (no LLM calls). Agent configs are loaded but not yet used to invoke models. `build_context_pack()` in `utils.py` constructs `ContextPack`; `allowed_files`/`denied_files` are set but not enforced at runtime.

Contract fingerprinting now uses RFC 8785 canonical JSON serialization before SHA-256 hashing, matching the spec requirement for deterministic canonicalization.

## Conventions

- Task IDs: `TASK-{index:03d}-{slug}`
- Slugs: `slugify_name()` → lowercase, hyphens, no consecutive hyphens
- CodeIndex keys: `slugify_name(contract.name)`

## Quick reference

| Task | Where |
|------|-------|
| Add agent role | `src/dcode_app_factory/agent_configs/<stage>/<role>.json` |
| Change debate logic | `EngineeringLoop._proposer`, `_challenger`, `_arbiter` in `loops.py` |
| Change context pack | `build_context_pack()` in `utils.py` |
| Change spec parsing | `ProductLoop._extract_sections`, `_task_from_section` in `loops.py` |
| Add/modify contract fields | `models.py` |
