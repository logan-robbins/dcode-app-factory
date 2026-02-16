from importlib.metadata import version

from .debate import Debate, DebateResult
from .loops import EngineeringLoop, ProductLoop, ProjectLoop
from .models import (
    AgentConfig,
    ContextPack,
    Epic,
    EscalationArtifact,
    IOContractSketch,
    MicroModuleContract,
    Pillar,
    ShipEvidence,
    Story,
    StructuredSpec,
    Task,
    TaskStatus,
    ValidationIssue,
)
from .registry import CodeIndex
from .utils import get_agent_config_dir, slugify_name, to_canonical_json, validate_task_dependency_dag


def get_version() -> str:
    try:
        return version(__name__)
    except Exception:
        return "0.0.0"


__all__ = [
    "AgentConfig",
    "get_agent_config_dir",
    "CodeIndex",
    "ContextPack",
    "Debate",
    "DebateResult",
    "EngineeringLoop",
    "Epic",
    "EscalationArtifact",
    "IOContractSketch",
    "MicroModuleContract",
    "Pillar",
    "ProductLoop",
    "ProjectLoop",
    "ShipEvidence",
    "Story",
    "StructuredSpec",
    "Task",
    "TaskStatus",
    "ValidationIssue",
    "slugify_name",
    "to_canonical_json",
    "validate_task_dependency_dag",
]
