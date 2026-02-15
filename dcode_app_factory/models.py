from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    HALTED = "halted"


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


@dataclass(frozen=True)
class ContextPack:
    """Deterministic context pack for one agent invocation."""

    task_id: str
    objective: str
    interfaces: list[str]
    allowed_files: list[str]
    denied_files: list[str]


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
