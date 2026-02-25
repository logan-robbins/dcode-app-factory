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
    BoundaryLevel,
    ClassContract,
    Challenge,
    CodeIndexEntry,
    CodeIndexStatus,
    ComponentContract,
    ContractCompatibility,
    ContextPack,
    Epic,
    EscalationArtifact,
    FractalPlan,
    HumanResolution,
    InterfaceChangeException,
    IOContractSketch,
    MicroModuleContract,
    MicroPlan,
    Pillar,
    ProductSpec,
    ProjectState,
    Proposal,
    ServiceContract,
    SystemContract,
    ReuseSearchReport,
    ShipEvidence,
    Story,
    Task,
    TaskStatus,
    ValidationIssue,
    ValidationReport,
)
from .registry import CodeIndex
from .utils import get_agent_config_dir, slugify_name, to_canonical_json, validate_spec, validate_task_dependency_dag


def get_version() -> str:
    try:
        return version(__name__)
    except Exception:
        return "0.0.0"


__all__ = [
    "Adjudication",
    "AgentConfig",
    "ArtifactEnvelope",
    "ArtifactStatus",
    "ArtifactType",
    "BoundaryLevel",
    "ClassContract",
    "Challenge",
    "CodeIndex",
    "CodeIndexEntry",
    "CodeIndexStatus",
    "ComponentContract",
    "ContractCompatibility",
    "ContextPack",
    "DebateGraph",
    "DebateResult",
    "EngineeringLoop",
    "Epic",
    "EscalationArtifact",
    "FactoryOrchestrator",
    "FractalPlan",
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
    "ServiceContract",
    "ShipEvidence",
    "Story",
    "SystemContract",
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
