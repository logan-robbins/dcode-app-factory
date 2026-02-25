from __future__ import annotations

import fcntl
import json
import re
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from .backends import mark_module_immutable, seal_module_version
from .models import (
    ARTIFACT_STATUS_TRANSITIONS,
    Adjudication,
    ArtifactEnvelope,
    ArtifactStatus,
    ArtifactType,
    Challenge,
    ContextPack,
    EscalationArtifact,
    InterfaceChangeException,
    ClassContract,
    MicroModuleContract,
    MicroPlan,
    ProductSpec,
    ProjectState,
    ProjectTaskState,
    Proposal,
    ServiceContract,
    ShipEvidence,
    SystemContract,
    TaskStatus,
)


@contextmanager
def _locked_file(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class FactoryStateStore:
    """Filesystem state store implementing the full schema from the spec."""

    def __init__(self, root: Path, *, project_id: str | None = None) -> None:
        self.base_root = root
        self.project_id = project_id
        self.root = project_scoped_root(root, project_id)
        self.product_dir = self.root / "product"
        self.project_dir = self.root / "project"
        self.tasks_dir = self.root / "tasks"
        self.artifacts_dir = self.root / "artifacts"
        self.modules_dir = self.root / "modules"
        self.system_contracts_dir = self.root / "system_contracts"
        self.service_contracts_dir = self.root / "service_contracts"
        self.class_contracts_dir = self.root / "class_contracts"
        self.code_index_dir = self.root / "code_index"
        self.debates_dir = self.root / "debates"
        self.context_packs_dir = self.root / "context_packs"
        self.agent_outputs_dir = self.root / "agent_outputs"
        self.exceptions_dir = self.root / "exceptions"
        self.escalations_dir = self.root / "escalations"
        self.release_dir = self.root / "release"
        self.ensure_structure()

    def ensure_structure(self) -> None:
        for directory in (
            self.root,
            self.product_dir,
            self.project_dir,
            self.tasks_dir,
            self.artifacts_dir,
            self.modules_dir,
            self.system_contracts_dir,
            self.service_contracts_dir,
            self.class_contracts_dir,
            self.code_index_dir,
            self.debates_dir,
            self.context_packs_dir,
            self.agent_outputs_dir,
            self.exceptions_dir,
            self.escalations_dir,
            self.release_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        if self.project_id is not None:
            metadata_path = self.root / "project_id.txt"
            metadata_path.write_text(f"{self.project_id}\n", encoding="utf-8")

    @property
    def state_machine_path(self) -> Path:
        return self.project_dir / "state_machine.json"

    @property
    def product_spec_json_path(self) -> Path:
        return self.product_dir / "spec.json"

    @property
    def product_spec_md_path(self) -> Path:
        return self.product_dir / "spec.md"

    def write_product_spec(self, spec: ProductSpec, markdown: str) -> None:
        self.product_spec_json_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        self.product_spec_md_path.write_text(markdown, encoding="utf-8")

    def write_project_state(self, state: ProjectState) -> None:
        with _locked_file(self.state_machine_path):
            self.state_machine_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    def read_project_state(self) -> ProjectState:
        if not self.state_machine_path.is_file():
            raise FileNotFoundError(f"state machine not found: {self.state_machine_path}")
        with _locked_file(self.state_machine_path):
            return ProjectState.model_validate_json(self.state_machine_path.read_text(encoding="utf-8"))

    def write_task_file(self, task_id: str, content: str) -> Path:
        path = self.tasks_dir / f"{task_id}.md"
        path.write_text(content, encoding="utf-8")
        return path

    def write_context_pack(self, context_pack: ContextPack) -> Path:
        path = self.context_packs_dir / f"{context_pack.cp_id}.json"
        path.write_text(context_pack.model_dump_json(indent=2), encoding="utf-8")
        return path

    def read_context_pack(self, cp_id: str) -> ContextPack:
        path = self.context_packs_dir / f"{cp_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"context pack not found: {path}")
        return ContextPack.model_validate_json(path.read_text(encoding="utf-8"))

    def write_agent_output(self, *, stage: str, role: str, run_id: str, payload: dict[str, Any]) -> Path:
        safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "-", stage.strip()).strip("-")
        safe_role = re.sub(r"[^A-Za-z0-9_.-]+", "-", role.strip()).strip("-")
        safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id.strip()).strip("-")
        if not safe_stage or not safe_role or not safe_run:
            raise ValueError("stage, role, and run_id must contain filesystem-safe characters")
        path = self.agent_outputs_dir / safe_stage / safe_role / f"{safe_run}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return path

    def write_interface_change_exception(self, exc: InterfaceChangeException) -> Path:
        path = self.exceptions_dir / f"{exc.exception_id}.json"
        path.write_text(exc.model_dump_json(indent=2), encoding="utf-8")
        return path

    def write_escalation(self, escalation: EscalationArtifact) -> Path:
        path = self.escalations_dir / f"{escalation.escalation_id}.json"
        path.write_text(escalation.model_dump_json(indent=2), encoding="utf-8")
        return path

    def write_debate(
        self,
        artifact_id: str,
        proposal: Proposal,
        challenge: Challenge,
        adjudication: Adjudication,
    ) -> Path:
        debate_dir = self.debates_dir / artifact_id
        debate_dir.mkdir(parents=True, exist_ok=True)
        (debate_dir / "proposal.json").write_text(proposal.model_dump_json(indent=2), encoding="utf-8")
        (debate_dir / "challenge.json").write_text(challenge.model_dump_json(indent=2), encoding="utf-8")
        (debate_dir / "adjudication.json").write_text(adjudication.model_dump_json(indent=2), encoding="utf-8")
        return debate_dir

    def write_micro_plan(self, plan: MicroPlan) -> Path:
        return self.write_artifact_payload(
            artifact_type=ArtifactType.MICRO_PLAN,
            artifact_id=plan.micro_plan_id,
            payload=plan.model_dump(mode="json"),
        )

    def write_system_contract(self, contract: SystemContract) -> Path:
        contract_dir = self.system_contracts_dir / contract.system_id / contract.system_version
        contract_dir.mkdir(parents=True, exist_ok=True)
        path = contract_dir / "contract.json"
        path.write_text(contract.model_dump_json(indent=2), encoding="utf-8")
        return path

    def write_service_contract(self, contract: ServiceContract) -> Path:
        contract_dir = self.service_contracts_dir / contract.service_id / contract.service_version
        contract_dir.mkdir(parents=True, exist_ok=True)
        path = contract_dir / "contract.json"
        path.write_text(contract.model_dump_json(indent=2), encoding="utf-8")
        return path

    def write_class_contract(self, contract: ClassContract) -> Path:
        contract_dir = self.class_contracts_dir / contract.class_id / contract.class_version
        contract_dir.mkdir(parents=True, exist_ok=True)
        path = contract_dir / "contract.json"
        path.write_text(contract.model_dump_json(indent=2), encoding="utf-8")
        return path

    def write_module_contract(self, contract: MicroModuleContract) -> Path:
        module_dir = self.modules_dir / contract.module_id / contract.module_version
        module_dir.mkdir(parents=True, exist_ok=True)
        path = module_dir / "contract.json"
        path.write_text(contract.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
        return path

    def write_ship_evidence(self, evidence: ShipEvidence) -> Path:
        module_dir = self.modules_dir / evidence.module_id / evidence.module_version
        module_dir.mkdir(parents=True, exist_ok=True)
        path = module_dir / "ship.json"
        path.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")
        return path

    def seal_module(self, module_id: str, version: str) -> None:
        module_version_dir = self.modules_dir / module_id / version
        seal_module_version(module_version_dir)
        mark_module_immutable(module_version_dir)

    def write_artifact(self, envelope: ArtifactEnvelope) -> Path:
        artifact_dir = self.artifacts_dir / envelope.artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        envelope_path = artifact_dir / "envelope.json"
        if envelope_path.exists():
            raise ValueError(f"Artifact already exists: {envelope.artifact_id}")
        envelope_path.write_text(envelope.model_dump_json(indent=2), encoding="utf-8")
        payload_path = artifact_dir / "payload.json"
        payload_path.write_text(json.dumps(envelope.payload, indent=2, sort_keys=True), encoding="utf-8")
        return envelope_path

    def write_artifact_payload(self, artifact_type: ArtifactType, artifact_id: str, payload: dict[str, Any]) -> Path:
        artifact_dir = self.artifacts_dir / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"{artifact_type.value.lower()}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return path

    def read_artifact(self, artifact_id: str) -> ArtifactEnvelope:
        path = self.artifacts_dir / artifact_id / "envelope.json"
        return ArtifactEnvelope.model_validate_json(path.read_text(encoding="utf-8"))

    def transition_artifact_status(self, artifact_id: str, new_status: ArtifactStatus) -> ArtifactEnvelope:
        envelope = self.read_artifact(artifact_id)
        allowed = ARTIFACT_STATUS_TRANSITIONS[envelope.status]
        if new_status not in allowed:
            raise ValueError(
                f"Illegal artifact status transition for {artifact_id}: {envelope.status.value} -> {new_status.value}"
            )
        envelope.status = new_status
        path = self.artifacts_dir / artifact_id / "envelope.json"
        path.write_text(envelope.model_dump_json(indent=2), encoding="utf-8")
        return envelope


class ArtifactStoreService:
    """Universal envelope service for all persisted artifact writes."""

    def __init__(self, store: FactoryStateStore) -> None:
        self.store = store

    def create(self, envelope: ArtifactEnvelope) -> Path:
        return self.store.write_artifact(envelope)

    def challenge(self, artifact_id: str) -> ArtifactEnvelope:
        return self.store.transition_artifact_status(artifact_id, ArtifactStatus.CHALLENGED)

    def adjudicate(self, artifact_id: str) -> ArtifactEnvelope:
        return self.store.transition_artifact_status(artifact_id, ArtifactStatus.ADJUDICATED)

    def ship(self, artifact_id: str) -> ArtifactEnvelope:
        return self.store.transition_artifact_status(artifact_id, ArtifactStatus.SHIPPED)

    def deprecate(self, artifact_id: str) -> ArtifactEnvelope:
        return self.store.transition_artifact_status(artifact_id, ArtifactStatus.DEPRECATED)


def sanitize_project_id(project_id: str) -> str:
    value = project_id.strip()
    if not value:
        raise ValueError("project_id must be non-empty")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    if not value:
        raise ValueError("project_id contains no filesystem-safe characters")
    return value[:128]


def project_scoped_root(root: Path, project_id: str | None) -> Path:
    if project_id is None:
        return root
    slug = sanitize_project_id(project_id)
    if root.name == slug and root.parent.name == "projects":
        return root
    return root / "projects" / slug


def build_project_state(spec: ProductSpec, *, project_id: str | None = None) -> ProjectState:
    tasks: dict[str, ProjectTaskState] = {}
    declaration_order = 0
    for pillar in spec.pillars:
        for epic in pillar.epics:
            for story in epic.stories:
                for task in story.tasks:
                    tasks[task.task_id] = ProjectTaskState(
                        pillar=pillar.name,
                        epic=epic.name,
                        story=story.name,
                        task=task.name,
                        status=TaskStatus.PENDING,
                        depends_on=list(task.depends_on),
                        declaration_order=declaration_order,
                    )
                    declaration_order += 1

    return ProjectState(
        project_id=project_id if project_id is not None else spec.spec_id,
        spec_version=spec.spec_version,
        updated_at=datetime.now(UTC),
        tasks=tasks,
    )
