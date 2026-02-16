# SPEC Implementation Gap Analysis

## Scope and approach

This audit compares the requirements in `SPEC.md` against the current repository implementation documented in `README.md` and implemented in `src/dcode_app_factory/` + `scripts/` + `tests/`.

## Executive summary

- The repository currently implements a **deterministic skeleton** of the three-loop flow (Product → Project → Engineering), basic dependency DAG validation, basic debate flow, task blocking on failure, RFC 8785 contract canonicalization, and in-memory append-only contract registration.
- The majority of the deep spec requirements (state store filesystem schema, artifact envelopes, immutable + opaque backends, interface-change exception protocol, reuse-search governance, release loop, LangGraph/deepagents integration, embedded vector store, and checkpoint/recovery guarantees) are **not implemented**.
- Net assessment: **partially aligned prototype**, not yet compliant with the full SPEC.

## Coverage matrix

### 1) Product Loop

| Spec requirement | Current state | Gap |
|---|---|---|
| Convert raw spec into structured hierarchy (pillar/epic/story/task) | Implemented in `ProductLoop.run()` with one synthetic pillar/epic/story and tasks from markdown `##` headings. | **Partial**: hierarchy exists, but content is synthetic and does not preserve rich spec semantics. |
| Enforce complete `io_contract_sketch` dimensions | Implemented (`IOContractSketch.validate_complete`). | **Mostly implemented** for non-empty/non-placeholder checks. |
| Generate globally unique task IDs with collision resolution and max-length limits | IDs generated as `TASK-{index}-{slug}` only. No collision suffix pass or max-length enforcement. | **Missing** full ID-generation rules. |
| Persist hierarchical project folder + per-task files | No state-store folder generation exists. | **Missing**. |

### 2) Project Loop / state machine

| Spec requirement | Current state | Gap |
|---|---|---|
| Dependency DAG validation + cycle/unknown dependency fail-fast | Implemented in `validate_task_dependency_dag()`. | Implemented baseline. |
| Deterministic dispatch by dependency order | Implemented via sorted Kahn topological traversal. | Implemented baseline. |
| Full state machine JSON (`state_machine/state.json`) with declaration_order and legal transitions | Not implemented; state is in-memory task objects only. | **Missing**. |
| Strict transition rules (`PENDING/IN_PROGRESS/SHIPPED/HALTED/BLOCKED` semantics) | Minimal status model exists but lacks full transition validator and transition audit writes. | **Missing** full rules. |
| External state-store persistence and resumability | No persisted state machine. | **Missing**. |

### 3) Engineering Loop and debate

| Spec requirement | Current state | Gap |
|---|---|---|
| Mandatory Propose → Challenge → Adjudicate | Implemented via `Debate.run()`. | Implemented minimal flow. |
| Debate retries bounded to exactly 2 revisions and final verdict categories (`APPROVE`, `APPROVE_WITH_AMENDMENTS`, `REJECT`) | Uses configurable retries default 2 but arbiter returns only `PASS`/`FAIL`; no amendment verdicts or structured adjudication schema. | **Missing** spec verdict model and state-graph enforcement details. |
| Micro Plan artifact before implementation | No micro-plan artifact/model/node exists. | **Missing**. |
| Structured debate artifacts persisted to state store | Trace object exists in memory only; not persisted in required artifact envelope layout. | **Missing**. |
| Escalation artifact on terminal failure with full schema fields | Minimal `EscalationArtifact` dataclass exists with limited fields. | **Partial/Missing** full schema + persistence + resume integration. |

### 4) Contracts, artifacts, and evidence

| Spec requirement | Current state | Gap |
|---|---|---|
| Canonicalized contract fingerprinting (RFC 8785 + SHA-256) | Implemented (`canonical.py` + `MicroModuleContract.fingerprint`). | Implemented baseline. |
| Full `MicroModuleContract` canonical schema from spec | Contract exists but omits many spec fields/typed dimensions and governance metadata. | **Partial**. |
| Universal Artifact Envelope metadata for all artifacts | No artifact envelope model or write-path enforcement. | **Missing**. |
| Ship evidence schema with required test evidence links | Minimal `ShipEvidence` exists (tests_run/logs only). | **Partial/Missing** full schema requirements. |
| Immutable shipped deliverables | Not implemented; no immutable backend. | **Missing**. |

### 5) Context isolation and opaque implementation

| Spec requirement | Current state | Gap |
|---|---|---|
| Fresh context per role invocation with enforceable allowed/denied access | `ContextPack` is generated but enforcement is not at backend/tool layer. | **Missing** enforceable isolation. |
| Opaque implementation rule enforced after ship (closed-source behavior) | No filesystem/backend gate to block implementation reads. | **Missing**. |
| Required blocked-access error format | No blocked-access runtime path exists. | **Missing**. |
| Role-specific context sections enforced | Config includes allowed sections; no strict runtime enforcement beyond passing lists into `ContextPack`. | **Partial**. |

### 6) Interface change exception protocol

| Spec requirement | Current state | Gap |
|---|---|---|
| Detect and raise interface-change exceptions with strict schema | No exception artifact/model/workflow implemented. | **Missing**. |
| Pause dependent tasks and route exception resolution | Not implemented in dispatcher/state machine. | **Missing**. |
| Resume flow after contract update + requeue | Not implemented. | **Missing**. |

### 7) Code Index and reuse governance

| Spec requirement | Current state | Gap |
|---|---|---|
| Append-only Code Index with one-way status transitions (`SHIPPED`→`DEPRECATED/SUPERSEDED`) | In-memory append-only registration exists, but no status model or one-way transition controls. | **Partial/Missing**. |
| Embedded vector store + semantic search | Not implemented (no ChromaDB/equivalent integration). | **Missing**. |
| Reuse-first policy with mandatory `ReuseSearchReport` before `CREATE_NEW` | No reuse search/report flow exists. | **Missing**. |
| Deprecation metadata and replacement pointers | Not implemented. | **Missing**. |
| Reindex operation on embedding-model changes | Not implemented. | **Missing**. |

### 8) Release stage and lifecycle governance

| Spec requirement | Current state | Gap |
|---|---|---|
| Release loop (`release_manager`, `gatekeeper`) integrated in runtime | JSON configs exist under `agent_configs/release_loop/` but runtime loops do not load/use them. | **Missing integration**. |
| Release-time dependency checks on deprecated modules | Not implemented. | **Missing**. |
| Compatibility/deprecation enforcement workflow | Not implemented. | **Missing**. |

### 9) Platform architecture and resilience requirements

| Spec requirement | Current state | Gap |
|---|---|---|
| LangGraph + deepagents orchestration | Not implemented; loops are plain Python classes. | **Missing**. |
| FilesystemBackend/CompositeBackend/ContextPackBackend/OpaqueEnforcementBackend/ImmutableArtifactBackend | Not implemented. | **Missing**. |
| Checkpointing and crash-and-resume integration tests | Not implemented. | **Missing**. |
| Recursion-limit and state-key leakage mitigations | Not implemented. | **Missing**. |

## Feature inventory: implemented vs absent

### Implemented (notable)

1. Deterministic three-loop skeleton orchestration in memory.
2. Product-loop section extraction and task creation from markdown headings.
3. IO contract sketch completeness checks.
4. DAG validation for task dependencies.
5. Deterministic topological dispatch.
6. Debate protocol with proposer/challenger/arbiter hooks.
7. Retry behavior with escalation object on repeated failure.
8. Context pack object construction with config-driven section list and budget bounds.
9. RFC 8785 canonical JSON contract hashing.
10. In-memory append-only code index insertion with uniqueness by slug.
11. Basic CLI with explicit/default spec loading.
12. Basic test coverage for the above baseline behaviors.

### Missing or materially incomplete (spec-critical)

1. Full state-store folder materialization and artifact persistence.
2. Full state machine schema + legal transition enforcement + declaration order persistence.
3. Artifact envelope for all writes.
4. Micro-plan generation and enforcement.
5. Structured debate/adjudication schema and PASS class granularity.
6. Backend-level context enforcement and opaque-implementation read blocking.
7. Interface-change exception lifecycle.
8. Reuse-search governance and report validation.
9. Embedded vector Code Index + semantic retrieval.
10. Module status lifecycle (`SHIPPED/DEPRECATED/SUPERSEDED`) and one-way transitions.
11. Release stage implementation and deprecation checks.
12. LangGraph/deepagents implementation, checkpointing, recursion/state-key mitigations.
13. Crash recovery tests.

## Prioritized gap list (recommended implementation order)

1. **State + storage foundations:** implement filesystem state store, universal artifact envelope, and state machine persistence/transition validator.
2. **Context and safety controls:** implement backend-level context enforcement + opaque enforcement + immutable artifact backend.
3. **Engineering-loop fidelity:** add micro-plan artifact, structured debate artifacts, adjudication taxonomy, and escalation persistence.
4. **Interface governance:** implement interface-change exception detection, routing, and resume.
5. **Code Index maturity:** add vector-backed append-only index, lifecycle states, reuse-search reports, and reindex support.
6. **Release governance:** wire release loop and deprecation/compatibility checks.
7. **Resilience/compliance:** add LangGraph checkpoints, crash-recovery tests, recursion/state-key mitigations.


## Step-by-step progress log (this review pass)

- [x] Step 1 — Read `README.md` fully and confirm current documented implementation scope. **COMPLETE**
- [x] Step 2 — Read `SPEC.md` in full and enumerate requirement areas by section. **COMPLETE**
- [x] Step 3 — Inspect runtime code paths (`loops.py`, `models.py`, `utils.py`, `registry.py`, `debate.py`, `scripts/factory_main.py`) against SPEC requirements. **COMPLETE**
- [x] Step 4 — Inspect tests to validate what behavior is currently covered vs. untested. **COMPLETE**
- [x] Step 5 — Build a requirement-by-requirement coverage matrix and classify each item as Implemented / Partial / Missing. **COMPLETE**
- [x] Step 6 — Produce a consolidated implemented-vs-missing inventory for execution planning. **COMPLETE**
- [x] Step 7 — Prioritize gaps into a practical implementation order for follow-on work. **COMPLETE**
