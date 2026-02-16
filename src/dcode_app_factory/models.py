from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    HALTED = "halted"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    location: str
    message: str


@dataclass(frozen=True)
class IOContractSketch:
    """Contract sketch required at Product Loop stage."""

    inputs: list[str]
    outputs: list[str]
    error_surfaces: list[str]
    effects: list[str]
    modes: list[str]

    def validate_complete(self) -> None:
        fields = {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error_surfaces": self.error_surfaces,
            "effects": self.effects,
            "modes": self.modes,
        }
        placeholders = {"tbd", "todo", "n/a", "na", "none"}
        for field_name, values in fields.items():
            if not values:
                raise ValueError(f"io_contract_sketch.{field_name} cannot be empty")
            lowered = {value.strip().lower() for value in values if value.strip()}
            if lowered & placeholders:
                raise ValueError(f"io_contract_sketch.{field_name} contains placeholders")


@dataclass(frozen=True)
class MicroModuleContract:
    """Final micro-module contract used by Engineering Loop."""

    module_id: str
    name: str
    description: str
    inputs: list[dict[str, str]]
    outputs: list[dict[str, str]]
    error_surfaces: list[dict[str, str]]
    effects: list[str]
    modes: list[str]
    depends_on: list[str] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        payload = {
            "module_id": self.module_id,
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error_surfaces": self.error_surfaces,
            "effects": self.effects,
            "modes": self.modes,
            "depends_on": self.depends_on,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ShipEvidence:
    task_id: str
    adjudication: str
    tests_run: list[str]
    execution_logs: list[str]


@dataclass
class Task:
    task_id: str
    name: str
    description: str
    subtasks: list[str]
    acceptance_criteria: list[str]
    io_contract_sketch: IOContractSketch
    depends_on: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    contract: MicroModuleContract | None = None
    ship_evidence: ShipEvidence | None = None


@dataclass
class Story:
    story_id: str
    name: str
    description: str
    user_facing_behavior: str
    tasks: list[Task]


@dataclass
class Epic:
    epic_id: str
    name: str
    description: str
    success_criteria: list[str]
    stories: list[Story]


@dataclass
class Pillar:
    pillar_id: str
    name: str
    description: str
    rationale: str
    epics: list[Epic]


@dataclass
class StructuredSpec:
    product_name: str
    vision: str
    constraints: list[str]
    pillars: list[Pillar]
    version: str = "2026.1"

    def iter_tasks(self) -> list[Task]:
        return [
            task
            for pillar in self.pillars
            for epic in pillar.epics
            for story in epic.stories
            for task in story.tasks
        ]

    def validate(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        task_ids: set[str] = set()
        verbs = re.compile(r"\b(returns|displays|raises|writes|emits|rejects|validates|produces|creates)\b", re.IGNORECASE)

        if not self.pillars:
            issues.append(ValidationIssue("ERROR", "pillars", "Spec must include at least one pillar"))

        for p_idx, pillar in enumerate(self.pillars):
            p_loc = f"pillars[{p_idx}]"
            if not pillar.epics:
                issues.append(ValidationIssue("ERROR", p_loc, "Every pillar must have at least one epic"))
            for field_name in ("pillar_id", "name", "description", "rationale"):
                if not getattr(pillar, field_name).strip():
                    issues.append(ValidationIssue("ERROR", p_loc, f"{field_name} must be non-empty"))

            for e_idx, epic in enumerate(pillar.epics):
                e_loc = f"{p_loc}.epics[{e_idx}]"
                if not epic.stories:
                    issues.append(ValidationIssue("ERROR", e_loc, "Every epic must have at least one story"))
                if not epic.success_criteria:
                    issues.append(ValidationIssue("ERROR", e_loc, "Every epic must have at least one success criterion"))
                for s_idx, story in enumerate(epic.stories):
                    s_loc = f"{e_loc}.stories[{s_idx}]"
                    if not story.tasks:
                        issues.append(ValidationIssue("ERROR", s_loc, "Every story must have at least one task"))
                    for t_idx, task in enumerate(story.tasks):
                        t_loc = f"{s_loc}.tasks[{t_idx}]"
                        if task.task_id in task_ids:
                            issues.append(ValidationIssue("ERROR", t_loc, f"Duplicate task_id {task.task_id}"))
                        task_ids.add(task.task_id)
                        if len(task.subtasks) < 2:
                            issues.append(ValidationIssue("ERROR", t_loc, "Task requires at least two subtasks"))
                        if len(task.acceptance_criteria) < 2:
                            issues.append(ValidationIssue("ERROR", t_loc, "Task requires at least two acceptance criteria"))
                        if len({s.lower().strip() for s in task.subtasks}) != len(task.subtasks):
                            issues.append(ValidationIssue("WARNING", t_loc, "Subtasks appear duplicated"))
                        for criterion in task.acceptance_criteria:
                            if not verbs.search(criterion):
                                issues.append(ValidationIssue("WARNING", t_loc, f"Non-testable acceptance criterion: {criterion}"))
                        if not task.io_contract_sketch.error_surfaces:
                            issues.append(ValidationIssue("WARNING", t_loc, "error_surfaces should contain specific error conditions"))
                        try:
                            task.io_contract_sketch.validate_complete()
                        except ValueError as exc:
                            issues.append(ValidationIssue("ERROR", t_loc, str(exc)))

        by_id = {task.task_id: task for task in self.iter_tasks()}
        for task in self.iter_tasks():
            for dep in task.depends_on:
                if dep not in by_id:
                    issues.append(ValidationIssue("ERROR", task.task_id, f"depends_on references unknown task {dep}"))

        return issues


@dataclass(frozen=True)
class ContextPack:
    """Deterministic context pack for one agent invocation."""

    task_id: str
    objective: str
    interfaces: list[str]
    allowed_files: list[str]
    denied_files: list[str]
    context_budget_tokens: int
    required_sections: list[str]


@dataclass(frozen=True)
class AgentConfig:
    """Runtime config for one agent role."""

    stage: str
    role: str
    model_tier: str
    temperature: float
    max_context_tokens: int
    context_policy: str
    allowed_context_sections: list[str]


@dataclass
class DebateTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    proposal: str = ""
    challenge: str = ""
    adjudication: str = ""
    passed: bool = False


@dataclass(frozen=True)
class EscalationArtifact:
    task_id: str
    reason: str
    debate_trace: DebateTrace
