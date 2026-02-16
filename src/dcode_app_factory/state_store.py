from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import Task, TaskStatus


ALLOWED_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED},
    TaskStatus.IN_PROGRESS: {TaskStatus.COMPLETED, TaskStatus.HALTED},
    TaskStatus.BLOCKED: {TaskStatus.PENDING},
    TaskStatus.HALTED: set(),
    TaskStatus.COMPLETED: set(),
}


@dataclass(frozen=True)
class ArtifactEnvelope:
    artifact_type: str
    artifact_id: str
    version: str
    status: str
    created_at: str
    payload: dict[str, Any]

    @classmethod
    def build(cls, artifact_type: str, artifact_id: str, payload: dict[str, Any], *, version: str = "2026.1", status: str = "CURRENT") -> "ArtifactEnvelope":
        return cls(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            version=version,
            status=status,
            created_at=datetime.now(UTC).isoformat(),
            payload=payload,
        )


@dataclass(frozen=True)
class TaskStateEntry:
    task_id: str
    status: str
    depends_on: list[str]
    declaration_order: int

    @classmethod
    def from_task(cls, task: Task, declaration_order: int) -> "TaskStateEntry":
        return cls(
            task_id=task.task_id,
            status=task.status.value,
            depends_on=list(task.depends_on),
            declaration_order=declaration_order,
        )


@dataclass
class ProjectStateMachine:
    tasks: dict[str, TaskStateEntry] = field(default_factory=dict)

    def assert_valid_transition(self, task_id: str, old: TaskStatus, new: TaskStatus) -> None:
        allowed = ALLOWED_TRANSITIONS[old]
        if new not in allowed:
            raise ValueError(f"Illegal task status transition for {task_id}: {old.value} -> {new.value}")

    def to_json(self) -> str:
        payload = {
            "tasks": [asdict(entry) for entry in sorted(self.tasks.values(), key=lambda e: e.declaration_order)]
        }
        return json.dumps(payload, indent=2, sort_keys=True)


class FilesystemStateStore:
    """Minimal filesystem-backed state store for project state machine artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.state_machine_dir = self.root / "state_machine"
        self.artifacts_dir = self.root / "artifacts"
        self.state_machine_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def state_json_path(self) -> Path:
        return self.state_machine_dir / "state.json"

    def write_state_machine(self, state: ProjectStateMachine) -> None:
        self.state_json_path.write_text(state.to_json(), encoding="utf-8")

    def write_artifact(self, envelope: ArtifactEnvelope) -> Path:
        path = self.artifacts_dir / f"{envelope.artifact_id}.json"
        path.write_text(json.dumps(asdict(envelope), indent=2, sort_keys=True), encoding="utf-8")
        return path
