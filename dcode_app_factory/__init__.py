from importlib.metadata import version

from .debate import Debate, DebateResult
from .loops import EngineeringLoop, ProductLoop, ProjectLoop
from .models import (
    AgentConfig,
    ContextPack,
    Epic,
    IOContractSketch,
    MicroModuleContract,
    Pillar,
    Story,
    StructuredSpec,
    Task,
    TaskStatus,
)
from .registry import CodeIndex
from .utils import slugify_name, to_canonical_json, validate_task_dependency_dag


def get_version() -> str:
    try:
        return version(__name__)
    except Exception:
        return "0.0.0"


__all__ = [
    "AgentConfig",
    "CodeIndex",
    "ContextPack",
    "Debate",
    "DebateResult",
    "EngineeringLoop",
    "Epic",
    "IOContractSketch",
    "MicroModuleContract",
    "Pillar",
    "ProductLoop",
    "ProjectLoop",
    "Story",
    "StructuredSpec",
    "Task",
    "TaskStatus",
    "slugify_name",
    "to_canonical_json",
    "validate_task_dependency_dag",
]
