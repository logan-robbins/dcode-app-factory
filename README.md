# dcode_app_factory

This repository contains a **skeleton implementation** of an AI software
product factory as outlined in the provided specification. The intent
of this project is not to fully realise the ambitious design but to
demonstrate the core concepts in a minimal, deterministic form.

## Architecture Overview

The factory is organised into three orchestrated loops, each
responsible for a distinct phase of software production:

1. **Product Loop** – Converts a high‑level specification into a
   structured specification comprising pillars, epics, stories and
   tasks. In this skeleton the Product Loop simply generates a single
   placeholder pillar containing one epic, one story and one task
   without performing any research【972992892572183†L90-L107】.
2. **Project Loop** – Decomposes the structured specification into a
   flat list of tasks and dispatches them to the Engineering Loop
   sequentially. Execution halts on the first failure【972992892572183†L90-L116】.
3. **Engineering Loop** – Implements a single task by performing a
   mock “debate” consisting of propose, challenge and adjudicate
   phases. Success marks the task as completed and registers its
   contract in the Code Index; failure marks the task as failed and
   stops the project.

Supporting these loops are several small components:

* **Data models** – Plain Python dataclasses represent the structured
  specification and micro‑module contracts. No external validation
  library is required.
* **Code Index** – An in‑memory registry of micro‑module contracts
  keyed by a slugified version of the module name. It provides
  registration and lookup functionality.
* **Debate mechanism** – A lightweight engine that executes a
  deterministic proposal/challenge/adjudication sequence.

## Installation

The project avoids external dependencies to remain runnable in an
offline environment. A `pyproject.toml` is provided for completeness
but declares no runtime dependencies. The only requirement is
Python 3.12 or newer. Although the original specification calls for
dependency management via `uv` and a dedicated virtual environment,
network restrictions prevent retrieving packages from external
repositories. Therefore the factory runs against the system Python
environment.

To install development dependencies manually (optional), ensure
pytest is available in your environment. On a system with internet
access this could be achieved with `pip install pytest`, but in the
offline execution environment used here pytest is already provided.

## Running the Factory

The main entry point is `scripts/factory_main.py`. It reads an
optional `SPEC.md` file, runs the Product Loop to produce a
structured specification, then runs the Project Loop and the
Engineering Loop for each task. The registered micro‑modules are
listed at the end.

```bash
# From the repository root, run the factory with the default placeholder spec
PYTHONPATH=$(pwd) python scripts/factory_main.py

# Or provide a path to an existing SPEC.md
PYTHONPATH=$(pwd) python scripts/factory_main.py --spec-file path/to/SPEC.md
```

Running under `uv` is also possible, but only the `--active`
environment flag can be used because the offline environment cannot
create a fully populated virtual environment:

```bash
PYTHONPATH=$(pwd) uv run --active python scripts/factory_main.py
```

## Testing

Unit tests are located in the `tests/` directory and can be executed
with pytest. Because the project is not installed into a virtual
environment, `PYTHONPATH` must be set to include the repository
root:

```bash
PYTHONPATH=$(pwd) pytest -q
```

The tests cover slugification, canonical JSON serialization, registry
behaviour, and the end‑to‑end execution of the loops.

## Limitations & Future Work

This project is intentionally minimal and deterministic. Key
limitations include:

* **No actual research or planning** – The Product Loop does not
  expand or refine the spec using external knowledge sources.
* **No multi‑agent debate** – The Engineering Loop uses a single
  propositional function and trivial challenge/adjudication, rather
  than three distinct agents.
* **No persistence** – The Code Index is in‑memory and is lost
  between runs.
* **No concurrency** – Tasks are executed sequentially; there is no
  parallelism or retry logic.

Despite these simplifications, the skeleton captures the essence of the
factory architecture: structured decomposition of a spec, single‑task
execution with context isolation, and an append‑only registry of
micro‑modules【972992892572183†L90-L116】. Extending the system to meet the full
specification would involve integrating research tools, state machines,
robust agentic debate and persistent storage.
