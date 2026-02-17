from importlib.metadata import version

from .debate import DebateGraph, DebateResult
from .loops import EngineeringLoop, FactoryOrchestrator, ProductLoop, ProjectLoop, ReleaseLoop
from .model_selection import DEFAULT_MODELS_BY_TIER, RuntimeModelSelection, resolve_agent_models
from .models import (
    Adjudication,
    AgentConfig,
    ArtifactEnvelope,
    ArtifactStatus,
    ArtifactType,
    Challenge,
    CodeIndexEntry,
    CodeIndexStatus,
    ContextPack,
    Epic,
    EscalationArtifact,
    HumanResolution,
    InterfaceChangeException,
    IOContractSketch,
    MicroModuleContract,
    MicroPlan,
    Pillar,
    ProductSpec,
    ProjectState,
    Proposal,
    ReuseSearchReport,
    ShipEvidence,
    Story,
    StructuredSpec,
    Task,
    TaskStatus,
    ValidationIssue,
    ValidationReport,
)
from .registry import AppendOnlyCodeIndex, CodeIndex
from .utils import get_agent_config_dir, slugify_name, to_canonical_json, validate_spec, validate_task_dependency_dag


def get_version() -> str:
    try:
        return version(__name__)
    except Exception:
        return "0.0.0"


__all__ = [
    "Adjudication",
    "AgentConfig",
    "AppendOnlyCodeIndex",
    "ArtifactEnvelope",
    "ArtifactStatus",
    "ArtifactType",
    "Challenge",
    "CodeIndex",
    "CodeIndexEntry",
    "CodeIndexStatus",
    "ContextPack",
    "DebateGraph",
    "DebateResult",
    "EngineeringLoop",
    "Epic",
    "EscalationArtifact",
    "FactoryOrchestrator",
    "HumanResolution",
    "IOContractSketch",
    "InterfaceChangeException",
    "MicroModuleContract",
    "MicroPlan",
    "Pillar",
    "ProductLoop",
    "ProductSpec",
    "ProjectLoop",
    "ProjectState",
    "Proposal",
    "ReleaseLoop",
    "ReuseSearchReport",
    "RuntimeModelSelection",
    "ShipEvidence",
    "Story",
    "StructuredSpec",
    "Task",
    "TaskStatus",
    "ValidationIssue",
    "ValidationReport",
    "DEFAULT_MODELS_BY_TIER",
    "get_agent_config_dir",
    "resolve_agent_models",
    "slugify_name",
    "to_canonical_json",
    "validate_spec",
    "validate_task_dependency_dag",
]
