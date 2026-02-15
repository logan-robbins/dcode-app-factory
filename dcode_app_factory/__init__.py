"""Top‑level package for the AI software product factory.

This package exposes the core components required to build and run a
simplified version of the factory described in the specification. The
implementation focuses on data modelling, micro‑module registration,
loop orchestration, and a mock debate mechanism. Real LLM calls and
external integrations are intentionally avoided to ensure reproducible
behaviour within the harness.

Key modules exposed:

* :mod:`models` – Pydantic data models representing the structured
  specification and micro‑module contracts.
* :mod:`registry` – A simple code index for registering and
  retrieving micro modules.
* :mod:`loops` – Skeleton implementations of the Product, Project,
  and Engineering loops.
* :mod:`debate` – A minimal debate mechanism simulating propose,
  challenge, and adjudication phases.
* :mod:`utils` – Utility functions for slugifying strings and
  producing canonical JSON.

The package version is derived from the project metadata defined in
`pyproject.toml`.
"""

from importlib.metadata import version

from .models import (  # noqa: F401
    StructuredSpec,
    Pillar,
    Epic,
    Story,
    Task,
    MicroModuleContract,
    InputSpec,
    OutputSpec,
    ErrorSpec,
)
from .registry import CodeIndex  # noqa: F401
from .loops import ProductLoop, ProjectLoop, EngineeringLoop  # noqa: F401
from .debate import DebateResult, Debate  # noqa: F401
from .utils import slugify_name, to_canonical_json  # noqa: F401


def get_version() -> str:
    """Return the package version as declared in pyproject.toml."""
    try:
        return version(__name__)
    except Exception:
        # When running from a source checkout without installation,
        # fallback to a default version.
        return "0.0.0"
