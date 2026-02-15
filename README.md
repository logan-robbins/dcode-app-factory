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

## Layout

- `dcode_app_factory/models.py`: contracts, spec hierarchy, context pack, agent config models.
- `dcode_app_factory/loops.py`: product/project/engineering loop orchestration.
- `dcode_app_factory/debate.py`: 3-agent debate protocol and trace artifact.
- `dcode_app_factory/registry.py`: append-only in-memory code index with contract fingerprints.
- `agent_configs/*/*.json`: agent runtime configs for Product, Project, Engineering stages.

## Run tests

```bash
PYTHONPATH=$(pwd) pytest -q
```
