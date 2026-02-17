# Development Tasks

This document outlines the high‑level tasks required to build a skeleton
implementation of the **AI software product factory** described in the
specification. Items are numbered and should be updated as work
progresses. Completed items should be annotated with a status mark.

1. **Project initialization** – Create a `pyproject.toml` with the
   appropriate metadata and dependencies and initialise the virtual
   environment using `uv`. ✅
2. **Directory structure** – Create a Python package under
   `src/dcode_app_factory` (src layout) and submodules for loops, data
   models, and infrastructure. ✅
3. **Data modelling** – Define data classes (using the standard
   library `dataclasses`) to represent the structured specification
   (pillars, epics, stories, tasks) and micro‑module contracts
   (inputs, outputs, errors). ✅
4. **Micro‑module registry** – Implement a registry that records
   available micro modules and their metadata in a Code Index. ✅
5. **Loop implementations** – Implement skeleton classes for the
   Product Loop, Project Loop, and Engineering Loop. Each loop should
   expose a `run()` method that logs or returns its progress. ✅
6. **Debate mechanism** – Implement a minimal debate mechanism that
   simulates proposal, challenge, and adjudication phases between
   placeholder agents. The mechanism should return PASS/FAIL results
   without invoking any external models. ✅
7. **CLI entry point** – Provide a `factory_main.py` script under
   `scripts` that instantiates and runs the loops in sequence. ✅
8. **Unit tests** – Write pytest suites under the `tests` directory
   covering data modelling, registry operations, loop sequencing, and
   the debate mechanism. ✅
9. **Documentation update** – Update `README.md` to reflect the
   current system, installation instructions, and usage. ✅

## Codex Cloud Readiness (2026-02-16)

1. **Dependency management hardening** – Add `pytest` as a managed `uv`
   dev dependency so `uv run pytest` works in clean environments. ✅
2. **CLI entrypoint restoration** – Add `scripts/factory_main.py` so the
   documented launch command exists and runs end-to-end. ✅
3. **Repository agent guidance** – Add root-level `AGENTS.md` with
   explicit `uv` workflows and verification commands. ✅
4. **Cloud setup handoff** – Add `setup_script.md` with copy-paste setup
   commands for Codex cloud environment bootstrap. ✅
5. **Documentation refresh** – Rewrite `README.md` to match current
   commands, environment settings, and Codex cloud setup. ✅

## Repository Hygiene (2026-02-16)

1. **Git ignore baseline** – Add a project-appropriate `.gitignore` for
   Python, `uv`, test/coverage outputs, caches, editor files, and local
   env files. ✅

## 2026 API and configurability hardening (2026-02-16)

1. **Canonical contract fingerprinting** – Adopt RFC 8785 canonical JSON
   serialization before SHA-256 hashing for contract fingerprints. ✅
2. **Runtime settings centralization** – Add environment-driven runtime
   settings for section fan-out, context budget floor/cap, and default
   spec path behavior. ✅
3. **End-to-end validation expansion** – Add robust E2E tests covering
   CLI execution, configurable default spec selection, and runtime config
   validation failures. ✅
4. **Docs synchronization** – Update `README.md` with new settings and
   dependency details. ✅
5. **Default LLM configurability** – Add tier-based default model
   routing with env overrides and tests for role-level selection. ✅

## Merge `codex/update-code-to-2026-standards` (2026-02-16)

1. **Discover** – Inspect local/remote branch state and identify merge
   conflict files. ✅
2. **Plan** – Track conflict-resolution progress in this task list and
   resolve each conflict with one canonical implementation. ✅
3. **Implement** – Merge `origin/codex/update-code-to-2026-standards`
   into `main` and resolve all conflicts without introducing parallel
   paths. ✅
4. **Verify** – Run `uv run pytest -q` and confirm no regressions. ✅
5. **Update** – Refresh `README.md` if behavior/commands changed and
   finalize task status annotations. ✅

## SPEC gap implementation kickoff (2026-02-16)

1. **State store foundation** – Added filesystem-backed project state store with `state_machine/state.json` persistence. ✅
2. **Transition guardrails** – Added explicit legal task transition validator for project state-machine status changes. ✅
3. **Artifact envelope baseline** – Added minimal universal artifact envelope and persisted ship-evidence artifacts under `state_store/artifacts/`. ✅
4. **Validation coverage** – Added tests for state persistence, artifact output, and illegal transition rejection. ✅

## End-to-end Teams bot prompt validation (2026-02-16)

1. **Discover** – Audit CLI, loops, model routing, and tests for existing end-to-end behavior. ✅
2. **Plan** – Track the run/fix/verify cycle for the official Teams-bot prompt test. ✅
3. **Implement** – Execute the app with the official prompt input and fix bugs encountered (`state_store/` git-noise). ✅
4. **Verify** – Re-ran the prompt command and `uv run pytest -q`; both pass. ✅
5. **Update** – Synced `README.md` run/debug guidance and finalized task statuses. ✅

## IMPLEMENT.md full execution (2026-02-16)

1. **Phase 1 architecture migration** — LangGraph/deepagents orchestration, backend layering, full state-store schema. ✅
2. **Phase 2 Product Loop fidelity** — ProductSpec schema, tooling, task-id/slug rules, approval gate. ✅
3. **Phase 3 Project Loop state machine** — full model, transitions, deterministic dispatch, blocked cascade + re-eval. ✅
4. **Phase 4 Engineering + debate fidelity** — micro-plan, structured debate artifacts, retry/amendment, per-role context packs, module iteration, escalation schema. ✅
5. **Phase 5 contracts/artifacts/evidence** — full contract/evidence schemas, universal envelope, immutability. ✅
6. **Phase 6 context/opaque enforcement** — ContextPackBackend, OpaqueEnforcementBackend, blocked-access errors. ✅
7. **Phase 7 code index governance** — Chroma-backed index, reuse search/report, lifecycle status, reindex. ✅
8. **Phase 8 interface-change exceptions** — ICE schema, raise/detect/persist, resolution routing hooks. ✅
9. **Phase 9 release governance** — release loop integration, deprecation enforcement gates. ✅
10. **Phase 10 resilience/compliance** — sqlite checkpointing, crash-recovery and isolation tests, recursion/state-key mitigations. ✅

## Production real-debate default and stock-trader E2E validation (2026-02-17)

1. **Discover** — Re-read `IMPLEMENT.md` fully and verify code/docs paths tied to `FACTORY_DEBATE_USE_LLM`. ✅
2. **Plan** — Add a numbered execution checklist for real-by-default debate and live prompt validation. ✅
3. **Implement** — Flip runtime default to model-backed debate and align README/test expectations. ✅
4. **Verify** — Run `uv run pytest -q` and real E2E CLI run using stock-trader prompt with `nohup` + 15s polling; fixed DebateGraph halt propagation retry-loop bug uncovered during live run. ✅
5. **Update** — Finalize this task section and ensure README reflects the validated production path. ✅

## Project-scoped delivery root (2026-02-17)

1. **Discover** — Trace `project_id` usage and all filesystem write paths across loops/state store/tools. ✅
2. **Plan** — Route all generated artifacts into a project-id-scoped repository folder under state store. ✅
3. **Implement** — Add project root scoping helper, wire loops/orchestrator/tools, and preserve compatibility for direct state-store tests. ✅
4. **Verify** — Update/extend tests for project-root namespacing and run `uv run pytest -q`. ✅
5. **Update** — Refresh README state-store layout and env variable docs for `FACTORY_PROJECT_ID`. ✅

## Production hardening: vectorization, micro decomposition, prompt/index policy (2026-02-17)

1. **Discover** — Audit vector embedding implementation, micro-plan decomposition granularity, and prompt/index policy enforcement points. ✅
2. **Plan** — Define one canonical production path: OpenAI semantic embeddings by default, atomic module decomposition, and explicit reuse-first code-index policy in prompts. ✅
3. **Implement** — Upgrade embedding stack, wire runtime embedding model selection, improve micro-plan decomposition/dependency inference, and strengthen agent prompts/context to require code-index evidence. ✅
4. **Verify** — Run `uv run pytest -q` plus official stock-trader E2E prompt with `nohup` and 15s polling. ✅
5. **Update** — Sync `README.md` to reflect embedding defaults, reuse policy behavior, and validation commands. ✅

## Structured-output serializer warning elimination (2026-02-17)

1. **Discover** — Reproduce and trace `PydanticSerializationUnexpectedValue` during live `DebateGraph` structured-output calls. ✅
2. **Plan** — Define one canonical, warning-free structured-output path and typed normalization boundary in `llm.py`. ✅
3. **Implement** — Add structured-output adapter/normalizer and switch debate proposer/challenger/arbiter calls to explicit `function_calling + strict`. ✅
4. **Verify** — Run `uv run pytest -q` and a live model-backed debate invocation with warnings-as-errors checks. ✅
5. **Update** — Refresh `README.md` notes to reflect the production structured-output method and warning rationale. ✅

## Release gate consistency for mixed-version reuse (2026-02-17)

1. **Discover** — Trace why stock-trader E2E release failed (`dependency_check`/`fingerprint_check`) with mixed `module_ref` versions. ✅
2. **Plan** — Enforce dependency-ref resolution from actual module refs and expand release set by transitive dependency closure. ✅
3. **Implement** — Patch EngineeringLoop contract dependency wiring + reuse compatibility checks; patch ReleaseLoop init to include dependency closure. ✅
4. **Verify** — Run `uv run pytest -q` and rerun stock-trader E2E prompt with 15s polling; validated both first-run and mixed-version rerun release outcomes. ✅
5. **Update** — Sync README with release-closure behavior and finalize statuses. ✅
