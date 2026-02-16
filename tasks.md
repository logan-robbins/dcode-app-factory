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
