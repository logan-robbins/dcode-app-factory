from __future__ import annotations

import hashlib
import re
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from .canonical import to_canonical_json


SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
TASK_ID_RE = re.compile(r"^(TSK-\d+|T-[a-z0-9-]+-[a-z0-9-]+-[a-z0-9-]+-\d+)$")

# Exact-match placeholder tokens (case-insensitive after strip + lower)
_PLACEHOLDER_EXACT: frozenset[str] = frozenset({
    "tbd", "todo", "n/a", "na", "none", "placeholder", "stub", "mock",
    "tba", "fixme", "xxx", "...", "pending", "unimplemented",
    "not implemented", "replace me", "change me", "update this",
    "fill in later", "to be determined", "to be defined",
})

# Substring patterns that indicate placeholder content within longer text
_PLACEHOLDER_SUBSTR_RE: re.Pattern[str] = re.compile(
    r"\b(TODO|FIXME|STUB|MOCK|PLACEHOLDER|TBD|HACK|XXX)\b",
    re.IGNORECASE,
)

_MIN_SUBSTANTIVE_LENGTH: int = 10


class Severity(StrEnum):
    ERROR = "ERROR"
    WARNING = "WARNING"


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    SHIPPED = "SHIPPED"
    HALTED = "HALTED"
    BLOCKED = "BLOCKED"
    ABANDONED = "ABANDONED"


class ArtifactType(StrEnum):
    MICRO_PLAN = "MICRO_PLAN"
    CONTRACT = "CONTRACT"
    IMPLEMENTATION = "IMPLEMENTATION"
    PROPOSAL = "PROPOSAL"
    CHALLENGE = "CHALLENGE"
    ADJUDICATION = "ADJUDICATION"
    SHIP_EVIDENCE = "SHIP_EVIDENCE"
    ESCALATION = "ESCALATION"
    CONTEXT_PACK = "CONTEXT_PACK"
    INTERFACE_CHANGE_EXCEPTION = "INTERFACE_CHANGE_EXCEPTION"
    REUSE_SEARCH_REPORT = "REUSE_SEARCH_REPORT"
    PRODUCT_SPEC = "PRODUCT_SPEC"
    EXAMPLES = "EXAMPLES"


ARTIFACT_TYPE_PREFIX: dict[ArtifactType, str] = {
    ArtifactType.MICRO_PLAN: "MP",
    ArtifactType.CONTRACT: "CTR",
    ArtifactType.IMPLEMENTATION: "IMPL",
    ArtifactType.PROPOSAL: "PROP",
    ArtifactType.CHALLENGE: "CHAL",
    ArtifactType.ADJUDICATION: "ADJ",
    ArtifactType.SHIP_EVIDENCE: "SHIP",
    ArtifactType.ESCALATION: "ESC",
    ArtifactType.CONTEXT_PACK: "CP",
    ArtifactType.INTERFACE_CHANGE_EXCEPTION: "ICE",
    ArtifactType.REUSE_SEARCH_REPORT: "RSR",
    ArtifactType.PRODUCT_SPEC: "SPEC",
    ArtifactType.EXAMPLES: "EX",
}


class ArtifactStatus(StrEnum):
    DRAFT = "DRAFT"
    CHALLENGED = "CHALLENGED"
    ADJUDICATED = "ADJUDICATED"
    SHIPPED = "SHIPPED"
    DEPRECATED = "DEPRECATED"


ARTIFACT_STATUS_TRANSITIONS: dict[ArtifactStatus, set[ArtifactStatus]] = {
    ArtifactStatus.DRAFT: {ArtifactStatus.CHALLENGED},
    ArtifactStatus.CHALLENGED: {ArtifactStatus.ADJUDICATED},
    ArtifactStatus.ADJUDICATED: {ArtifactStatus.SHIPPED, ArtifactStatus.DRAFT},
    ArtifactStatus.SHIPPED: {ArtifactStatus.DEPRECATED},
    ArtifactStatus.DEPRECATED: set(),
}


class CodeIndexStatus(StrEnum):
    CURRENT = "CURRENT"
    DEPRECATED = "DEPRECATED"
    SUPERSEDED = "SUPERSEDED"


class DebateVerdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


class AdjudicationDecision(StrEnum):
    APPROVE = "APPROVE"
    APPROVE_WITH_AMENDMENTS = "APPROVE_WITH_AMENDMENTS"
    REJECT = "REJECT"


class ShipDirective(StrEnum):
    SHIP = "SHIP"
    NO_SHIP = "NO_SHIP"


class ReuseDecision(StrEnum):
    REUSE = "REUSE"
    CREATE_NEW = "CREATE_NEW"


class ReuseConclusion(StrEnum):
    REUSE_EXISTING = "REUSE_EXISTING"
    CREATE_NEW = "CREATE_NEW"


class ContextAccessLevel(StrEnum):
    FULL = "FULL"
    CONTRACT_ONLY = "CONTRACT_ONLY"
    SUMMARY_ONLY = "SUMMARY_ONLY"
    METADATA_ONLY = "METADATA_ONLY"


class BoundaryLevel(StrEnum):
    L1_FUNCTIONAL = "L1_FUNCTIONAL"
    L2_SYSTEM = "L2_SYSTEM"
    L3_SERVICE = "L3_SERVICE"
    L4_COMPONENT = "L4_COMPONENT"
    L5_CLASS = "L5_CLASS"


class ContractCompatibility(StrEnum):
    NON_BREAKING = "NON_BREAKING"
    BREAKING_MAJOR = "BREAKING_MAJOR"


class InterfaceChangeType(StrEnum):
    CANNOT_SUPPORT = "CANNOT_SUPPORT"
    AMBIGUOUS = "AMBIGUOUS"
    INCOMPLETE = "INCOMPLETE"
    MISSING_DIMENSION = "MISSING_DIMENSION"


class CompatibilityExpectation(StrEnum):
    BACKWARD_COMPATIBLE = "BACKWARD_COMPATIBLE"
    BREAKING = "BREAKING"
    UNKNOWN = "UNKNOWN"


class Urgency(StrEnum):
    BLOCKING = "BLOCKING"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"


class HumanResolutionAction(StrEnum):
    APPROVE_OVERRIDE = "APPROVE_OVERRIDE"
    AMEND_SPEC = "AMEND_SPEC"
    SPLIT_TASK = "SPLIT_TASK"
    REVISE_PLAN = "REVISE_PLAN"
    PROVIDE_FIX = "PROVIDE_FIX"
    ABANDON_TASK = "ABANDON_TASK"


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Severity
    path: str
    field: str
    message: str


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    role: str
    model_tier: str
    temperature: float
    max_context_tokens: int
    context_policy: str
    allowed_context_sections: list[str]


class ProductRoleReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    summary: str
    warnings: list[str] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)


class DependencyManagerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    rationale: str
    blocking_dependencies: list[str] = Field(default_factory=list)


class DispatchDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_task_id: str | None = None
    rationale: str


class StateAuditDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    rationale: str
    findings: list[str] = Field(default_factory=list)


class MicroPlanReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    rationale: str
    blockers: list[str] = Field(default_factory=list)
    required_revisions: list[str] = Field(default_factory=list)


class ShipperDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ship_directive: ShipDirective
    rationale: str
    required_fixes: list[str] = Field(default_factory=list)


class ReleaseGateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dependency_check: Literal["PASS", "FAIL"]
    contract_completeness_check: Literal["PASS", "FAIL"]
    compatibility_check: Literal["PASS", "FAIL"]
    ownership_check: Literal["PASS", "FAIL"]
    tests_pass: Literal["PASS", "FAIL"]
    notes: list[str] = Field(default_factory=list)

    def gate_map(self) -> dict[str, str]:
        return {
            "dependency_check": self.dependency_check,
            "contract_completeness_check": self.contract_completeness_check,
            "compatibility_check": self.compatibility_check,
            "ownership_check": self.ownership_check,
            "tests_pass": self.tests_pass,
        }


class ReleaseManagerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_result: Literal["PASS", "FAIL"]
    rationale: str
    release_notes: list[str] = Field(default_factory=list)


class IOContractSketch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: str
    outputs: str
    error_surfaces: str
    effects: str
    modes: str

    @model_validator(mode="after")
    def validate_non_empty(self) -> "IOContractSketch":
        """Reject empty, placeholder, or trivially short field values."""
        for key in ("inputs", "outputs", "error_surfaces", "effects", "modes"):
            value = getattr(self, key).strip()
            if not value:
                raise ValueError(f"io_contract_sketch.{key} must be non-empty")
            if value.lower() in _PLACEHOLDER_EXACT:
                raise ValueError(
                    f"io_contract_sketch.{key} cannot be placeholder text (got {value!r})"
                )
            if _PLACEHOLDER_SUBSTR_RE.search(value):
                raise ValueError(
                    f"io_contract_sketch.{key} contains placeholder token "
                    f"(matched {_PLACEHOLDER_SUBSTR_RE.search(value).group()!r})"  # type: ignore[union-attr]
                )
            if len(value) < _MIN_SUBSTANTIVE_LENGTH:
                raise ValueError(
                    f"io_contract_sketch.{key} must be at least {_MIN_SUBSTANTIVE_LENGTH} "
                    f"characters (got {len(value)})"
                )
        return self


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    name: str
    description: str
    subtasks: list[str] = Field(min_length=2)
    acceptance_criteria: list[str] = Field(min_length=2)
    depends_on: list[str]
    io_contract_sketch: IOContractSketch

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        if not TASK_ID_RE.match(value):
            raise ValueError("task_id must match TSK-NNN or T-{pillar}-{epic}-{story}-{seq}")
        return value

    @model_validator(mode="after")
    def validate_required(self) -> "Task":
        if len({s.strip().lower() for s in self.subtasks}) != len(self.subtasks):
            raise ValueError("subtasks must be semantically distinct")
        return self


class Story(BaseModel):
    model_config = ConfigDict(extra="forbid")

    story_id: str = Field(pattern=r"^STR-\d+$")
    name: str
    description: str
    user_facing_behavior: str
    tasks: list[Task] = Field(min_length=1)


class Epic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epic_id: str = Field(pattern=r"^EPC-\d+$")
    name: str
    description: str
    success_criteria: list[str] = Field(min_length=1)
    stories: list[Story] = Field(min_length=1)


class Pillar(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pillar_id: str = Field(pattern=r"^PIL-\d+$")
    name: str
    description: str
    rationale: str
    epics: list[Epic] = Field(min_length=1)


class ProductSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_id: str = Field(pattern=r"^SPEC-\d+$")
    spec_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    title: str
    description: str
    created_at: datetime
    updated_at: datetime
    pillars: list[Pillar] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_task_uniqueness(self) -> "ProductSpec":
        tasks = self.iter_tasks()
        ids = [task.task_id for task in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task_id values must be globally unique")
        by_id = {task.task_id for task in tasks}
        for task in tasks:
            for dep in task.depends_on:
                if dep not in by_id:
                    raise ValueError(f"task {task.task_id} depends_on unknown task {dep}")
        return self

    def iter_tasks(self) -> list[Task]:
        return [
            task
            for pillar in self.pillars
            for epic in pillar.epics
            for story in epic.stories
            for task in story.tasks
        ]

class ProjectTaskState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pillar: str
    epic: str
    story: str
    task: str
    status: TaskStatus
    depends_on: list[str] = Field(default_factory=list)
    service_refs: list[str] = Field(default_factory=list)
    component_refs: list[str] = Field(default_factory=list)
    class_refs: list[str] = Field(default_factory=list)
    module_ref: str | None = None
    module_refs: list[str] = Field(default_factory=list)
    shipped_at: datetime | None = None
    halted_reason: str | None = None
    escalation_ref: str | None = None
    declaration_order: int = Field(ge=0)


PROJECT_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED},
    TaskStatus.IN_PROGRESS: {TaskStatus.SHIPPED, TaskStatus.HALTED},
    TaskStatus.BLOCKED: {TaskStatus.PENDING},
    TaskStatus.HALTED: {TaskStatus.PENDING, TaskStatus.ABANDONED, TaskStatus.SHIPPED},
    TaskStatus.SHIPPED: set(),
    TaskStatus.ABANDONED: set(),
}


class ProjectState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    spec_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    updated_at: datetime
    tasks: dict[str, ProjectTaskState]

    def transition(self, task_id: str, to_status: TaskStatus) -> None:
        task = self.tasks[task_id]
        if to_status not in PROJECT_TRANSITIONS[task.status]:
            raise ValueError(
                f"Illegal task status transition for {task_id}: {task.status.value} -> {to_status.value}"
            )
        task.status = to_status
        self.updated_at = datetime.now(UTC)


class ContextPermission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    access_level: ContextAccessLevel
    level: BoundaryLevel | None = None
    contract_ref: str | None = None

    @model_validator(mode="after")
    def validate_contract_ref_requires_level(self) -> "ContextPermission":
        if self.contract_ref is not None and self.level is None:
            raise ValueError("ContextPermission.contract_ref requires level")
        return self


class ContextPack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cp_id: str = Field(pattern=r"^CP-[0-9a-f]{8}$")
    task_id: str
    role: str
    objective: str
    permissions: list[ContextPermission]
    context_budget_tokens: int = Field(ge=256)
    required_sections: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def matching_permissions_for_path(self, path: str) -> list[ContextPermission]:
        normalized = path.rstrip("/")
        matches: list[ContextPermission] = []
        for permission in self.permissions:
            allowed = permission.path.rstrip("/")
            if normalized == allowed or normalized.startswith(f"{allowed}/"):
                matches.append(permission)
        return matches

    def access_for_path(self, path: str) -> ContextAccessLevel:
        matches = self.matching_permissions_for_path(path)
        if matches:
            matches.sort(key=lambda item: len(item.path), reverse=True)
            return matches[0].access_level
        return ContextAccessLevel.CONTRACT_ONLY

    def matching_permissions_for_ref(
        self,
        *,
        level: BoundaryLevel | None,
        contract_ref: str | None,
        path: str,
    ) -> list[ContextPermission]:
        matched = self.matching_permissions_for_path(path)
        if level is None:
            return [permission for permission in matched if permission.level is None]

        scoped: list[ContextPermission] = []
        for permission in matched:
            if permission.level is not None and permission.level != level:
                continue
            if permission.contract_ref is not None and contract_ref is not None and permission.contract_ref != contract_ref:
                continue
            if permission.contract_ref is not None and contract_ref is None:
                continue
            scoped.append(permission)
        if scoped:
            scoped.sort(key=lambda item: len(item.path), reverse=True)
            return scoped
        return []


class ContractInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    constraints: list[str] = Field(default_factory=list)


class ContractOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    invariants: list[str] = Field(default_factory=list)


class ContractErrorSurface(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    when: str
    surface: str


class EffectType(StrEnum):
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    WRITE = "WRITE"
    CALL = "CALL"
    MUTATION = "MUTATION"


class ContractEffect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: EffectType
    target: str
    description: str


class ContractModes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sync: bool
    async_mode: bool = Field(alias="async")
    notes: str


class DependencyRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: str
    why: str

    @field_validator("ref")
    @classmethod
    def validate_ref(cls, value: str) -> str:
        if "@" not in value:
            raise ValueError("dependency ref must be ModuleRef format MM-...@x.y.z")
        return value


class CompatibilityRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backward_compatible_with: list[str] = Field(default_factory=list)
    breaking_change_policy: str


class RuntimeBudgets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latency_ms_p95: float | None = None
    memory_mb_max: float | None = None


class SystemContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_id: str = Field(pattern=r"^SYS-[a-zA-Z0-9-]+$")
    system_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    name: str
    purpose: str
    parent_task_ref: str
    functional_contract_ref: str
    service_refs: list[str] = Field(min_length=1)
    owner: str = Field(min_length=1)
    compatibility_type: ContractCompatibility = ContractCompatibility.NON_BREAKING
    status: ArtifactStatus = ArtifactStatus.DRAFT
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ServiceContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service_id: str = Field(pattern=r"^SVC-[a-zA-Z0-9-]+$")
    service_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    name: str
    purpose: str
    parent_task_ref: str
    owner: str = Field(min_length=1)
    component_refs: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(min_length=1)
    outputs: list[str] = Field(min_length=1)
    error_surfaces: list[str] = Field(default_factory=list)
    effects: list[str] = Field(default_factory=list)
    modes: list[str] = Field(default_factory=list)
    compatibility_type: ContractCompatibility = ContractCompatibility.NON_BREAKING
    status: ArtifactStatus = ArtifactStatus.DRAFT
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field(return_type=str)
    @property
    def interface_fingerprint(self) -> str:
        """SHA-256 hex digest of the canonical JSON of interface-defining fields."""
        payload = {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error_surfaces": self.error_surfaces,
            "effects": self.effects,
            "modes": self.modes,
        }
        canonical = to_canonical_json(payload)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ClassMethodContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    raises: list[str] = Field(default_factory=list)


class ClassContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: str = Field(pattern=r"^CLS-[a-zA-Z0-9-]+$")
    class_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    name: str
    purpose: str
    parent_task_ref: str
    component_ref: str
    owner: str = Field(min_length=1)
    shared: bool = True
    methods: list[ClassMethodContract] = Field(min_length=1)
    invariants: list[str] = Field(default_factory=list)
    error_surfaces: list[str] = Field(default_factory=list)
    compatibility_type: ContractCompatibility = ContractCompatibility.NON_BREAKING
    status: ArtifactStatus = ArtifactStatus.DRAFT
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field(return_type=str)
    @property
    def interface_fingerprint(self) -> str:
        """SHA-256 hex digest of the canonical JSON of interface-defining fields."""
        payload = {
            "methods": [entry.model_dump(mode="json") for entry in self.methods],
            "invariants": self.invariants,
            "error_surfaces": self.error_surfaces,
        }
        canonical = to_canonical_json(payload)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class MicroModuleContract(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    module_id: str = Field(pattern=r"^MM-[a-zA-Z0-9-]+$")
    module_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    name: str
    purpose: str
    tags: list[str] = Field(default_factory=list)
    examples_ref: str
    status: ArtifactStatus = ArtifactStatus.DRAFT
    created_by: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    supersedes: str | None = None
    deprecated_by: str | None = None
    inputs: list[ContractInput] = Field(min_length=1)
    outputs: list[ContractOutput] = Field(min_length=1)
    error_surfaces: list[ContractErrorSurface] = Field(min_length=1)
    effects: list[ContractEffect] = Field(default_factory=list)
    modes: ContractModes
    error_cases: list[str] = Field(default_factory=list)
    dependencies: list[DependencyRef] = Field(default_factory=list)
    compatibility: CompatibilityRule
    runtime_budgets: RuntimeBudgets = Field(default_factory=RuntimeBudgets)
    level: BoundaryLevel = BoundaryLevel.L4_COMPONENT
    owner: str = "engineering_loop"
    service_ref: str | None = None
    class_contract_refs: list[str] = Field(default_factory=list)
    compatibility_type: ContractCompatibility = ContractCompatibility.NON_BREAKING

    @model_validator(mode="after")
    def validate_level(self) -> "MicroModuleContract":
        if self.level != BoundaryLevel.L4_COMPONENT:
            raise ValueError("MicroModuleContract level must be L4_COMPONENT")
        return self

    @computed_field(return_type=str)
    @property
    def interface_fingerprint(self) -> str:
        """SHA-256 hex digest of the canonical JSON of interface-defining fields."""
        payload = {
            "inputs": [entry.model_dump(mode="json") for entry in self.inputs],
            "outputs": [entry.model_dump(mode="json") for entry in self.outputs],
            "error_surfaces": [entry.model_dump(mode="json") for entry in self.error_surfaces],
            "effects": [entry.model_dump(mode="json") for entry in self.effects],
            "modes": self.modes.model_dump(by_alias=True, mode="json"),
        }
        canonical = to_canonical_json(payload)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ComponentContract(MicroModuleContract):
    """Canonical L4 contract type."""


class FileSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str        # relative to workspace_root
    content: str


class ShipVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: Literal["PASS", "FAIL"]
    interface_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_ref: str


class ShipEnvironment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_revision: str
    dependency_lock_ref: str
    runner_id: str


class ShipEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module_id: str
    module_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    ship_id: str = Field(pattern=r"^SHIP-[0-9a-f]{8}$")
    verified_at: datetime
    verification: ShipVerification
    environment: ShipEnvironment
    test_artifact_refs: list[str] = Field(default_factory=list)
    files_written: list[str] = Field(default_factory=list)
    test_output: str = ""
    test_passed: bool = False
    ship_time: datetime


class CodeIndexIoSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: list[str]
    outputs: list[str]
    error_surfaces: list[str]
    effects: list[str]
    modes: list[str]


class CodeIndexEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module_ref: str
    module_id: str
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    level: BoundaryLevel = BoundaryLevel.L4_COMPONENT
    name: str
    purpose: str
    owner: str = "engineering_loop"
    tags: list[str] = Field(default_factory=list)
    contract_ref: str
    examples_ref: str
    ship_ref: str
    io_summary: CodeIndexIoSummary
    dependencies: list[str] = Field(default_factory=list)
    compatibility_type: ContractCompatibility = ContractCompatibility.NON_BREAKING
    status: CodeIndexStatus = CodeIndexStatus.CURRENT
    superseded_by: str | None = None
    deprecation_reason: str | None = None
    notes: str = ""


class ReuseSearchCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module_ref: str
    why_rejected: str


class ReuseSearchReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    candidates_considered: list[ReuseSearchCandidate] = Field(default_factory=list)
    conclusion: ReuseConclusion
    justification: str

    @model_validator(mode="after")
    def validate_create_new_justification(self) -> "ReuseSearchReport":
        if self.conclusion == ReuseConclusion.CREATE_NEW and not self.justification.strip():
            raise ValueError("CREATE_NEW requires a non-empty justification")
        return self


class MicroIoContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: list[str] = Field(min_length=1)
    outputs: list[str] = Field(min_length=1)
    error_surfaces: list[str] = Field(default_factory=list)
    effects: list[str] = Field(default_factory=list)
    modes: list[str] = Field(default_factory=list)


class MicroPlanModule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module_id: str = Field(pattern=r"^MM-[a-zA-Z0-9-]+$")
    name: str
    purpose: str
    level: BoundaryLevel = BoundaryLevel.L4_COMPONENT
    owner: str = "engineering_loop"
    service_ref: str | None = None
    class_contract_refs: list[str] = Field(default_factory=list)
    io_contract: MicroIoContract
    error_cases: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    reuse_candidate_refs: list[str] = Field(default_factory=list)
    reuse_decision: ReuseDecision
    reuse_justification: str
    reuse_search_report: ReuseSearchReport


class MicroPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    micro_plan_id: str = Field(pattern=r"^MP-[0-9a-f]{8}$")
    parent_task_ref: str
    system_contract_ref: str | None = None
    service_contract_refs: list[str] = Field(default_factory=list)
    class_contract_refs: list[str] = Field(default_factory=list)
    modules: list[MicroPlanModule] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_structure(self) -> "MicroPlan":
        ids = [module.module_id for module in self.modules]
        if len(ids) != len(set(ids)):
            raise ValueError("micro plan module_id values must be unique")

        known = set(ids)
        incoming: dict[str, int] = {module.module_id: 0 for module in self.modules}
        edges: dict[str, list[str]] = {module.module_id: [] for module in self.modules}
        for module in self.modules:
            for dep in module.depends_on:
                if dep not in known:
                    if not dep.startswith("MM-"):
                        raise ValueError(f"invalid depends_on reference: {dep}")
                    continue
                incoming[module.module_id] += 1
                edges[dep].append(module.module_id)

        queue = [module.module_id for module in self.modules if incoming[module.module_id] == 0]
        seen = 0
        while queue:
            current = queue.pop(0)
            seen += 1
            for nxt in edges[current]:
                incoming[nxt] -= 1
                if incoming[nxt] == 0:
                    queue.append(nxt)
        if seen != len(self.modules):
            raise ValueError("micro plan depends_on graph contains a cycle")
        return self


class FractalPlan(MicroPlan):
    """Canonical level-aware plan artifact."""


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(pattern=r"^PROP-[0-9a-f]{8}$")
    target_artifact_id: str
    claim: str = Field(
        min_length=10,
        description="A substantive claim describing what the implementation achieves.",
    )
    deliverable_ref: str
    acceptance_checks: list[str] = Field(
        min_length=1,
        description="List of specific, executable acceptance criteria for the proposal.",
    )

    @field_validator("claim")
    @classmethod
    def validate_claim_not_placeholder(cls, value: str) -> str:
        """Reject placeholder or trivially short claims."""
        stripped = value.strip()
        if stripped.lower() in _PLACEHOLDER_EXACT:
            raise ValueError(f"claim cannot be placeholder text (got {value!r})")
        if _PLACEHOLDER_SUBSTR_RE.search(stripped):
            raise ValueError(
                f"claim contains placeholder token "
                f"(matched {_PLACEHOLDER_SUBSTR_RE.search(stripped).group()!r})"  # type: ignore[union-attr]
            )
        return value

    @field_validator("acceptance_checks")
    @classmethod
    def validate_acceptance_checks_substantive(cls, value: list[str]) -> list[str]:
        """Each acceptance check must be specific and executable (min 10 chars)."""
        for idx, check in enumerate(value):
            stripped = check.strip()
            if not stripped:
                raise ValueError(f"acceptance_checks[{idx}] must be non-empty")
            if len(stripped) < _MIN_SUBSTANTIVE_LENGTH:
                raise ValueError(
                    f"acceptance_checks[{idx}] must be at least {_MIN_SUBSTANTIVE_LENGTH} "
                    f"characters (got {len(stripped)})"
                )
            if stripped.lower() in _PLACEHOLDER_EXACT:
                raise ValueError(
                    f"acceptance_checks[{idx}] cannot be placeholder text (got {check!r})"
                )
        return value


class ChallengeFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invariant: str
    evidence: str
    required_change: str


class RubricAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion: Literal["R1", "R2", "R3", "R4", "R5", "R6"]
    assessment: Literal["MET", "NOT_MET"]
    evidence: str


class Challenge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    challenge_id: str = Field(pattern=r"^CHAL-[0-9a-f]{8}$")
    target_artifact_id: str
    verdict: DebateVerdict
    failures: list[ChallengeFailure] = Field(default_factory=list)
    optional_alternative_ref: str | None = None
    rubric_assessments: list[RubricAssessment] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def validate_rubric(self) -> "Challenge":
        """Enforce rubric completeness and verdict/failure consistency."""
        criteria = {entry.criterion for entry in self.rubric_assessments}
        if criteria != {"R1", "R2", "R3", "R4", "R5", "R6"}:
            raise ValueError("rubric_assessments must include exactly R1..R6")
        has_not_met = any(entry.assessment == "NOT_MET" for entry in self.rubric_assessments)
        if has_not_met and self.verdict != DebateVerdict.FAIL:
            raise ValueError("challenge verdict must be FAIL when any rubric assessment is NOT_MET")
        if self.verdict == DebateVerdict.FAIL and not self.failures:
            raise ValueError(
                "challenge with verdict=FAIL must include at least one failure entry"
            )
        return self


class AmendmentAction(StrEnum):
    MODIFY = "MODIFY"
    ADD = "ADD"
    REMOVE = "REMOVE"


class AdjudicationAmendment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: AmendmentAction
    target: str
    detail: str


class Adjudication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adjudication_id: str = Field(pattern=r"^ADJ-[0-9a-f]{8}$")
    target_artifact_id: str
    decision: AdjudicationDecision
    amendments: list[AdjudicationAmendment] = Field(default_factory=list)
    rationale: str = Field(
        min_length=1,
        description="Non-empty reasoning for the adjudication decision.",
    )
    ship_directive: ShipDirective

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> "Adjudication":
        """Enforce cross-field invariants for adjudication decisions."""
        if not self.rationale.strip():
            raise ValueError("adjudication rationale must be non-empty (not just whitespace)")
        if self.decision == AdjudicationDecision.REJECT and self.ship_directive != ShipDirective.NO_SHIP:
            raise ValueError(
                "adjudication with decision=REJECT must have ship_directive=NO_SHIP "
                f"(got {self.ship_directive.value})"
            )
        if self.decision == AdjudicationDecision.APPROVE_WITH_AMENDMENTS and not self.amendments:
            raise ValueError(
                "adjudication with decision=APPROVE_WITH_AMENDMENTS must include "
                "at least one amendment"
            )
        return self


class DebateTrail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposals: list[str] = Field(default_factory=list)
    challenges: list[str] = Field(default_factory=list)
    adjudications: list[str] = Field(default_factory=list)


class FailedInvariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invariant: str
    evidence: str


class EscalationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    escalation_id: str = Field(pattern=r"^ESC-[0-9a-f]{8}$")
    task_id: str
    task_ref: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    debate_trail: DebateTrail
    failed_invariants: list[FailedInvariant] = Field(default_factory=list)
    state_machine_snapshot: dict[str, Any]
    minimal_decision_required: str
    recommended_resolution: HumanResolutionAction
    resolution_context: str
    failure_root_cause: str
    retries_exhausted: int = Field(ge=0)
    context_pack_refs: list[str] = Field(default_factory=list)
    debate_artifact_refs: list[str] = Field(default_factory=list)


class RaisedBy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_ref: str
    role: str
    run_id: str


class ProposedContractDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    added_inputs: list[dict[str, Any]] = Field(default_factory=list)
    removed_inputs: list[dict[str, Any]] = Field(default_factory=list)
    modified_inputs: list[dict[str, Any]] = Field(default_factory=list)
    added_outputs: list[dict[str, Any]] = Field(default_factory=list)
    removed_outputs: list[dict[str, Any]] = Field(default_factory=list)
    modified_outputs: list[dict[str, Any]] = Field(default_factory=list)
    added_error_surfaces: list[dict[str, Any]] = Field(default_factory=list)
    added_effects: list[dict[str, Any]] = Field(default_factory=list)
    mode_changes: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_has_change(self) -> "ProposedContractDelta":
        values = self.model_dump()
        has_change = any(bool(value) for value in values.values())
        if not has_change:
            raise ValueError("proposed_contract_delta must include at least one change")
        return self


class InterfaceChangeException(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exception_id: str = Field(pattern=r"^ICE-[0-9a-f]{8}$")
    type: InterfaceChangeType
    raised_by: RaisedBy
    target_module: str
    reason: str
    evidence: list[str] = Field(min_length=1)
    proposed_contract_delta: ProposedContractDelta
    compatibility_expectation: CompatibilityExpectation
    urgency: Urgency


class ActorRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    run_id: str


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: str
    purpose: str


class ArtifactEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(pattern=r"^(MP|CTR|IMPL|PROP|CHAL|ADJ|SHIP|ESC|CP|ICE|RSR|SPEC|EX)-[0-9a-f]{8}$")
    artifact_type: ArtifactType
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    status: ArtifactStatus
    created_at: datetime
    created_by: ActorRef
    context_pack_ref: str
    inputs: list[ArtifactRef] = Field(default_factory=list)
    outputs: list[ArtifactRef] = Field(default_factory=list)
    notes: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        artifact_type: ArtifactType,
        created_by: ActorRef,
        context_pack_ref: str,
        payload: dict[str, Any],
        *,
        version: str = "1.0.0",
        status: ArtifactStatus = ArtifactStatus.DRAFT,
        inputs: list[ArtifactRef] | None = None,
        outputs: list[ArtifactRef] | None = None,
        notes: str = "",
    ) -> "ArtifactEnvelope":
        if not SEMVER_RE.match(version):
            raise ValueError(f"invalid semver version: {version}")
        prefix = ARTIFACT_TYPE_PREFIX[artifact_type]
        artifact_id = f"{prefix}-{uuid.uuid4().hex[:8]}"
        return cls(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            version=version,
            status=status,
            created_at=datetime.now(UTC),
            created_by=created_by,
            context_pack_ref=context_pack_ref,
            inputs=inputs or [],
            outputs=outputs or [],
            notes=notes,
            payload=payload,
        )


class SplitTaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    acceptance_criteria: list[str] = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)


class RevisedPlanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module_id: str
    description: str
    depends_on: list[str] = Field(default_factory=list)


class HumanResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: HumanResolutionAction
    rationale: str | None = None
    amended_acceptance_criteria: list[str] = Field(default_factory=list)
    amendment_rationale: str | None = None
    new_tasks: list[SplitTaskInput] = Field(default_factory=list)
    revised_micro_plan: list[RevisedPlanInput] = Field(default_factory=list)
    fix_description: str | None = None
    fix_artifacts: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_action_payload(self) -> "HumanResolution":
        action = self.action
        if action == HumanResolutionAction.APPROVE_OVERRIDE and not (self.rationale and self.rationale.strip()):
            raise ValueError("APPROVE_OVERRIDE requires rationale")
        if action == HumanResolutionAction.AMEND_SPEC:
            if not self.amended_acceptance_criteria:
                raise ValueError("AMEND_SPEC requires amended_acceptance_criteria")
            if not (self.amendment_rationale and self.amendment_rationale.strip()):
                raise ValueError("AMEND_SPEC requires amendment_rationale")
        if action == HumanResolutionAction.SPLIT_TASK and not self.new_tasks:
            raise ValueError("SPLIT_TASK requires new_tasks")
        if action == HumanResolutionAction.REVISE_PLAN and not self.revised_micro_plan:
            raise ValueError("REVISE_PLAN requires revised_micro_plan")
        if action == HumanResolutionAction.PROVIDE_FIX:
            if not (self.fix_description and self.fix_description.strip()):
                raise ValueError("PROVIDE_FIX requires fix_description")
            if not self.fix_artifacts:
                raise ValueError("PROVIDE_FIX requires fix_artifacts")
        if action == HumanResolutionAction.ABANDON_TASK and not (self.rationale and self.rationale.strip()):
            raise ValueError("ABANDON_TASK requires rationale")
        return self
