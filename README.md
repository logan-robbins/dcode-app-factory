# dcode-app-factory

## LLM Operator Guide

This README is for LLM operators. It defines how to start, run, and operate the platform as an agentic coding system.

Design assumption: the platform is always agent-runtime driven. There is no non-agent mode.

## System Design

```mermaid
flowchart TD
    INPUT[Work Request]

    subgraph OUTER[Factory Orchestrator Graph]
        PL[Product Agent Loop\nresearcher -> structurer -> validator]
        AG[Approval Gate\nAPPROVE | REJECT | AMEND]
        PJ[Project Agent Loop\ndependency_manager -> dispatcher -> state_auditor]
        EN[Engineering Agent Loop\nmicro_planner + shipper]
        DB[Debate Agent Swarm\nproposer <-> challenger -> arbiter]
        RL[Release Agent Loop\ngatekeeper -> release_manager]
    end

    OUTPUT[Run Result\nproject_success\nrelease_result\nrelease_details]

    INPUT --> PL --> AG
    AG -->|APPROVE| PJ
    AG -->|REJECT or AMEND| PL
    PJ --> EN --> DB --> PJ
    PJ --> RL --> OUTPUT
```

## Prerequisites

1. Python 3.12+
2. `uv`
3. `OPENAI_API_KEY`

## Setup

```bash
uv sync --all-groups --frozen
```

## Start and Run

Default run:

```bash
uv run python scripts/factory_main.py
```

Run with inline request text:

```bash
uv run python scripts/factory_main.py \
  --request-text "Build a minimal API endpoint for health checks" \
  --request-kind TASK \
  --approval-action APPROVE \
  --log-level INFO
```

Run with explicit request file:

```bash
uv run python scripts/factory_main.py \
  --request-file <REQUEST_FILE> \
  --request-kind FULL_APP \
  --approval-action APPROVE
```

Run incremental work against an existing codebase:

```bash
uv run python scripts/factory_main.py \
  --request-text "Add OAuth login with role-based route guards" \
  --request-kind FEATURE \
  --target-codebase-root <TARGET_REPO_ROOT> \
  --approval-action APPROVE
```

## CLI Contract

| Argument | Purpose | Values |
|---|---|---|
| `--request-text` | Inline work request | string |
| `--request-file` | File-based work request | path |
| `--spec-file` | Backward-compatible alias for `--request-file` | path |
| `--request-kind` | Request routing intent | `AUTO`, `FULL_APP`, `FEATURE`, `BUGFIX`, `REFACTOR`, `TASK` |
| `--target-codebase-root` | Existing repo root for incremental work | path |
| `--approval-action` | Approval gate action | `APPROVE`, `REJECT`, `AMEND` |
| `--approval-feedback` | Feedback for `REJECT` or `AMEND` | string |
| `--log-level` | Logging verbosity | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

Input rules:

- `--request-text` is mutually exclusive with `--request-file` and `--spec-file`.
- `--request-file` and `--spec-file` are mutually exclusive.
- If no request argument is provided, the runtime loads the default request path and fails fast if missing.

## Request-Kind Routing Guide

| Kind | Use When |
|---|---|
| `FULL_APP` | Building a full product scope from scratch |
| `FEATURE` | Adding a new capability to an existing product |
| `BUGFIX` | Correcting incorrect behavior |
| `REFACTOR` | Improving structure without intended behavior change |
| `TASK` | Delivering a focused implementation task |
| `AUTO` | Let the Product Agent Loop infer intent |

## Output Contract

Successful and failed runs always emit machine-readable summary lines to stdout:

- `project_success=<True|False>`
- `release_result=<PASS|FAIL>` when release executes
- `release_details:` followed by JSON payload

Process exit code:

- `0` when `project_success=True`
- `1` when request loading fails, execution fails, or `project_success=False`

## Environment Variables

Core runtime settings:

- `FACTORY_DEFAULT_REQUEST_PATH`
- `FACTORY_DEFAULT_SPEC_PATH` (legacy alias)
- `FACTORY_STATE_STORE_ROOT`
- `FACTORY_PROJECT_ID`
- `FACTORY_MAX_PRODUCT_SECTIONS`
- `FACTORY_CONTEXT_BUDGET_FLOOR`
- `FACTORY_CONTEXT_BUDGET_CAP`
- `FACTORY_RECURSION_LIMIT`
- `FACTORY_CHECKPOINT_DB`
- `FACTORY_CLASS_CONTRACT_POLICY`

Agent runtime routing:

- `FACTORY_MODEL_FRONTIER`
- `FACTORY_MODEL_EFFICIENT`
- `FACTORY_MODEL_ECONOMY`
- `FACTORY_EMBEDDING_MODEL`

Search tooling:

- `BRIGHTDATA_API_KEY`
- `BRIGHTDATA_SERP_ZONE`
- `BRIGHTDATA_SERP_COUNTRY`

## Operator Procedure for LLMs

1. Normalize user intent into one request statement.
2. Select `--request-kind` based on scope and change type.
3. Run once with `--approval-action APPROVE` unless human intervention is explicitly required.
4. Parse `project_success`, `release_result`, and `release_details`.
5. If run fails, rerun with `--log-level DEBUG` and an amended request.
6. Repeat until release gates pass or a human decision is required.

## Verification

```bash
uv run pytest -q
```
