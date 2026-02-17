# dcode-app-factory

LangGraph + deepagents implementation of the AI software product factory defined in `SPEC.md`, including:

- Product Loop with structured `ProductSpec` validation and emission
- Project Loop `StateGraph` dispatch cycle (`init_state_machine -> dispatch -> engineering -> update_state_machine`)
- Engineering Loop `StateGraph` with `micro_plan` + per-module iteration and nested debate subgraph
- Debate subgraph routing (`propose -> challenge -> route -> revise/adjudicate -> ship/halt`) using `Command(goto=...)` and parent propagation support
- Release Loop integration gates (`dependency`, `fingerprint`, `deprecation`, `code_index`)
- Filesystem state-store with universal artifact envelopes, context-pack persistence, exceptions, escalations, debates, modules, and release manifests
- Chroma-backed append-only Code Index with OpenAI semantic embeddings by default, metadata filters, lifecycle transitions, and model-change reindex support
- Backend enforcement wrappers for immutability, opaque implementation access control, and context-pack permissions
- SQLite checkpointing for outer/project graph execution

## Requirements

1. Python 3.12+
2. `uv`
3. `OPENAI_API_KEY` set (or present in `.env`)

## Setup

```bash
uv sync --all-groups --frozen
```

## Run

Default run (uses `SPEC.md` unless overridden):

```bash
uv run python scripts/factory_main.py
```

Run with explicit spec and non-interactive approval action:

```bash
uv run python scripts/factory_main.py \
  --spec-file /tmp/spec.md \
  --approval-action APPROVE \
  --log-level INFO
```

Run with a project-scoped delivery repository root:

```bash
FACTORY_PROJECT_ID=ACME-TRADER-SITE uv run python scripts/factory_main.py --spec-file /tmp/spec.md
```

## Official E2E prompt test

```bash
cat > /tmp/stock_trader_prompt_spec.md <<'EOF2'
# Product
## build a website for stock traders that comapares APIs from major providers and ranks them
EOF2
uv run python scripts/factory_main.py --spec-file /tmp/stock_trader_prompt_spec.md --log-level INFO
```

Expected output includes:

- `project_success=True`
- `release_result=PASS|FAIL`
- `release_details` JSON

## Tests

```bash
uv run pytest -q
```

## Environment Variables

Runtime settings:

- `FACTORY_DEFAULT_SPEC_PATH` (default: `SPEC.md`)
- `FACTORY_STATE_STORE_ROOT` (default: `state_store`)
- `FACTORY_PROJECT_ID` (default: `PROJECT-001`; used as project-scoped repository folder name)
- `FACTORY_MAX_PRODUCT_SECTIONS` (default: `8`)
- `FACTORY_CONTEXT_BUDGET_FLOOR` (default: `2000`)
- `FACTORY_CONTEXT_BUDGET_CAP` (default: `16000`)
- `FACTORY_RECURSION_LIMIT` (default: `1000`)
- `FACTORY_CHECKPOINT_DB` (default: `state_store/checkpoints/langgraph.sqlite`)

Model routing:

- `FACTORY_MODEL_FRONTIER` (default: `gpt-4o`)
- `FACTORY_MODEL_EFFICIENT` (default: `gpt-4o-mini`)
- `FACTORY_MODEL_ECONOMY` (default: `gpt-4o-mini`)
- `FACTORY_EMBEDDING_MODEL` (default: `text-embedding-3-large`; test-only deterministic override: `deterministic-hash-384`)
- `FACTORY_DEBATE_USE_LLM` (default: `true`; real model-backed debate path)

Search tooling:

- `TAVILY_API_KEY` (optional; enables `web_search` tool)
- `SERPAPI_API_KEY` (optional; fallback for `web_search`)

## State Store Layout

Primary directories under `state_store/projects/{project_id}/`:

- `product/` (`spec.json`, `spec.md`)
- `project/` (`state_machine.json`)
- `tasks/` (`{task_id}.md`)
- `artifacts/{artifact_id}/` (`envelope.json`, payload files)
- `modules/{module_id}/{version}/` (`contract.json`, `examples.md`, `ship.json`, sealed markers)
- `debates/{artifact_id}/` (`proposal.json`, `challenge.json`, `adjudication.json`)
- `context_packs/{cp_id}.json`
- `exceptions/{exception_id}.json`
- `escalations/{escalation_id}.json`
- `code_index/` (Chroma persistence)
- `release/{release_id}.json`
- `checkpoints/*.sqlite`

## Debugging

Inspect project state:

```bash
cat state_store/projects/PROJECT-001/project/state_machine.json
```

Inspect escalations:

```bash
ls -1 state_store/projects/PROJECT-001/escalations
```

Inspect debate artifacts:

```bash
find state_store/projects/PROJECT-001/debates -maxdepth 2 -type f
```

Inspect release manifests:

```bash
ls -1 state_store/projects/PROJECT-001/release
```

## Notes

- Product Loop uses deepagents (`create_deep_agent`) with integrated tools.
- Product + Engineering flows enforce reuse-first policy and execute `search_code_index` before create-new decisions.
- Engineering debate runs model-backed by default with structured output via `function_calling` + `strict=true` to avoid current `json_schema` serializer warnings in the LangChain/OpenAI response path; set `FACTORY_DEBATE_USE_LLM=false` only for deterministic local testing.
- Engineering contract dependencies are resolved from actual shipped module refs (not hardcoded `@1.0.0`), and release manifests include transitive dependency closure before running release gates.
- Opaque access-denied errors use the required fixed format defined in SPEC §12.5.
- Code Index status transitions are one-way (`CURRENT -> DEPRECATED|SUPERSEDED`).
