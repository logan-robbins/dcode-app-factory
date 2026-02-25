from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from pydantic import ValidationError

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File locking helpers
# ---------------------------------------------------------------------------

_LOCK_SUFFIX = ".lock"


@contextmanager
def _locked_file(path: Path) -> Iterator[None]:
    """Acquire an exclusive file lock for the duration of the context.

    Uses a separate .lock sidecar file so the actual data file can be
    atomically replaced via ``os.replace`` without disturbing the lock
    handle.  The lock file is created in the same directory as *path*
    so ``os.replace`` stays on the same filesystem.
    """
    lock_path = path.with_suffix(path.suffix + _LOCK_SUFFIX)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _atomic_write_text(path: Path, content: str) -> None:
    """Write *content* to *path* atomically.

    Writes to a temporary file in the same directory, then renames
    (``os.replace``) into place.  This prevents partial/corrupt reads
    if the process crashes mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_handle:
            tmp_handle.write(content)
            tmp_handle.flush()
            os.fsync(tmp_handle.fileno())
        os.replace(tmp_path, str(path))
    except BaseException:
        # Clean up the temp file on any failure.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _safe_read_json(path: Path, model_name: str) -> str:
    """Read a JSON file and raise a clear error if missing or unreadable.

    Args:
        path: Filesystem path to read.
        model_name: Human-readable label used in error messages.

    Returns:
        The raw file text.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or contains non-UTF-8 data.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{model_name} not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{model_name} at {path} contains invalid UTF-8 data") from exc
    if not text.strip():
        raise ValueError(f"{model_name} at {path} is empty")
    return text


# ---------------------------------------------------------------------------
# Debate record (read-side container)
# ---------------------------------------------------------------------------

class DebateRecord:
    """Typed container returned by ``read_debate``."""

    __slots__ = ("proposal", "challenge", "adjudication")

    def __init__(
        self,
        proposal: Proposal,
        challenge: Challenge,
        adjudication: Adjudication,
    ) -> None:
        self.proposal = proposal
        self.challenge = challenge
        self.adjudication = adjudication


# ---------------------------------------------------------------------------
# FactoryStateStore
# ---------------------------------------------------------------------------

class FactoryStateStore:
    """Filesystem state store implementing the full schema from the spec.

    All writes use atomic temp-file-then-rename to prevent corruption on
    crash.  Concurrency-sensitive paths (project state, artifact status
    transitions) are additionally guarded by ``fcntl`` exclusive file
    locks so multiple pipeline processes sharing the same directory
    do not race.
    """

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
        """Create all required directories if they do not exist."""
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

    # ------------------------------------------------------------------
    # Path properties
    # ------------------------------------------------------------------

    @property
    def state_machine_path(self) -> Path:
        """Path to the project state machine JSON file."""
        return self.project_dir / "state_machine.json"

    @property
    def product_spec_json_path(self) -> Path:
        """Path to the product spec JSON file."""
        return self.product_dir / "spec.json"

    @property
    def product_spec_md_path(self) -> Path:
        """Path to the product spec markdown file."""
        return self.product_dir / "spec.md"

    # ------------------------------------------------------------------
    # Product spec
    # ------------------------------------------------------------------

    def write_product_spec(self, spec: ProductSpec, markdown: str) -> None:
        """Persist the product spec as JSON and markdown.

        Args:
            spec: Validated ProductSpec model.
            markdown: Rendered markdown representation.
        """
        _atomic_write_text(self.product_spec_json_path, spec.model_dump_json(indent=2))
        _atomic_write_text(self.product_spec_md_path, markdown)

    def read_product_spec(self) -> ProductSpec:
        """Read and validate the persisted product spec.

        Returns:
            The deserialized ProductSpec.

        Raises:
            FileNotFoundError: If the spec file does not exist.
            ValueError: If the file is corrupt or fails validation.
        """
        text = _safe_read_json(self.product_spec_json_path, "product spec")
        try:
            return ProductSpec.model_validate_json(text)
        except ValidationError as exc:
            raise ValueError(f"product spec at {self.product_spec_json_path} failed validation: {exc}") from exc

    # ------------------------------------------------------------------
    # Project state (locked - hot path for concurrent processes)
    # ------------------------------------------------------------------

    def write_project_state(self, state: ProjectState) -> None:
        """Persist project state under an exclusive file lock.

        Uses atomic write to prevent corruption on crash.

        Args:
            state: Validated ProjectState model.
        """
        with _locked_file(self.state_machine_path):
            _atomic_write_text(self.state_machine_path, state.model_dump_json(indent=2))

    def read_project_state(self) -> ProjectState:
        """Read project state under an exclusive file lock.

        Returns:
            The deserialized ProjectState.

        Raises:
            FileNotFoundError: If the state machine file does not exist.
            ValueError: If the file is corrupt or fails validation.
        """
        if not self.state_machine_path.is_file():
            raise FileNotFoundError(f"state machine not found: {self.state_machine_path}")
        with _locked_file(self.state_machine_path):
            text = _safe_read_json(self.state_machine_path, "project state")
            try:
                return ProjectState.model_validate_json(text)
            except ValidationError as exc:
                raise ValueError(
                    f"project state at {self.state_machine_path} failed validation: {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Task files
    # ------------------------------------------------------------------

    def write_task_file(self, task_id: str, content: str) -> Path:
        """Write a task markdown file.

        Args:
            task_id: Task identifier used as the filename stem.
            content: Markdown content for the task.

        Returns:
            Path to the written file.
        """
        path = self.tasks_dir / f"{task_id}.md"
        _atomic_write_text(path, content)
        return path

    # ------------------------------------------------------------------
    # Context packs
    # ------------------------------------------------------------------

    def write_context_pack(self, context_pack: ContextPack) -> Path:
        """Persist a context pack.

        Args:
            context_pack: Validated ContextPack model.

        Returns:
            Path to the written file.
        """
        path = self.context_packs_dir / f"{context_pack.cp_id}.json"
        _atomic_write_text(path, context_pack.model_dump_json(indent=2))
        return path

    def read_context_pack(self, cp_id: str) -> ContextPack:
        """Read a context pack by ID.

        Args:
            cp_id: The context pack identifier (e.g. ``CP-1234abcd``).

        Returns:
            The deserialized ContextPack.

        Raises:
            FileNotFoundError: If the context pack does not exist.
            ValueError: If the file is corrupt or fails validation.
        """
        path = self.context_packs_dir / f"{cp_id}.json"
        text = _safe_read_json(path, "context pack")
        try:
            return ContextPack.model_validate_json(text)
        except ValidationError as exc:
            raise ValueError(f"context pack at {path} failed validation: {exc}") from exc

    def list_context_packs(self) -> list[str]:
        """Return a list of all stored context pack IDs.

        Returns:
            Sorted list of context pack IDs.
        """
        return sorted(
            p.stem for p in self.context_packs_dir.glob("*.json")
        )

    # ------------------------------------------------------------------
    # Agent outputs
    # ------------------------------------------------------------------

    def write_agent_output(self, *, stage: str, role: str, run_id: str, payload: dict[str, Any]) -> Path:
        """Persist an agent output payload.

        Args:
            stage: Pipeline stage name.
            role: Agent role name.
            run_id: Unique run identifier.
            payload: Arbitrary JSON-serializable dict to persist.

        Returns:
            Path to the written file.

        Raises:
            ValueError: If any identifier contains no filesystem-safe characters.
        """
        safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "-", stage.strip()).strip("-")
        safe_role = re.sub(r"[^A-Za-z0-9_.-]+", "-", role.strip()).strip("-")
        safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id.strip()).strip("-")
        if not safe_stage or not safe_role or not safe_run:
            raise ValueError("stage, role, and run_id must contain filesystem-safe characters")
        path = self.agent_outputs_dir / safe_stage / safe_role / f"{safe_run}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str))
        return path

    # ------------------------------------------------------------------
    # Interface change exceptions
    # ------------------------------------------------------------------

    def write_interface_change_exception(self, exc: InterfaceChangeException) -> Path:
        """Persist an interface change exception.

        Args:
            exc: Validated InterfaceChangeException model.

        Returns:
            Path to the written file.
        """
        path = self.exceptions_dir / f"{exc.exception_id}.json"
        _atomic_write_text(path, exc.model_dump_json(indent=2))
        return path

    def read_interface_change_exception(self, exception_id: str) -> InterfaceChangeException:
        """Read an interface change exception by ID.

        Args:
            exception_id: The exception identifier (e.g. ``ICE-deadbeef``).

        Returns:
            The deserialized InterfaceChangeException.

        Raises:
            FileNotFoundError: If the exception does not exist.
            ValueError: If the file is corrupt or fails validation.
        """
        path = self.exceptions_dir / f"{exception_id}.json"
        text = _safe_read_json(path, "interface change exception")
        try:
            return InterfaceChangeException.model_validate_json(text)
        except ValidationError as exc:
            raise ValueError(
                f"interface change exception at {path} failed validation: {exc}"
            ) from exc

    def list_interface_change_exceptions(self) -> list[str]:
        """Return a sorted list of all stored exception IDs.

        Returns:
            Sorted list of exception IDs.
        """
        return sorted(p.stem for p in self.exceptions_dir.glob("*.json"))

    # ------------------------------------------------------------------
    # Escalations
    # ------------------------------------------------------------------

    def write_escalation(self, escalation: EscalationArtifact) -> Path:
        """Persist an escalation artifact.

        Args:
            escalation: Validated EscalationArtifact model.

        Returns:
            Path to the written file.
        """
        path = self.escalations_dir / f"{escalation.escalation_id}.json"
        _atomic_write_text(path, escalation.model_dump_json(indent=2))
        return path

    def read_escalation(self, escalation_id: str) -> EscalationArtifact:
        """Read an escalation artifact by ID.

        Args:
            escalation_id: The escalation identifier (e.g. ``ESC-deadbeef``).

        Returns:
            The deserialized EscalationArtifact.

        Raises:
            FileNotFoundError: If the escalation does not exist.
            ValueError: If the file is corrupt or fails validation.
        """
        path = self.escalations_dir / f"{escalation_id}.json"
        text = _safe_read_json(path, "escalation")
        try:
            return EscalationArtifact.model_validate_json(text)
        except ValidationError as exc:
            raise ValueError(f"escalation at {path} failed validation: {exc}") from exc

    def list_escalations(self) -> list[str]:
        """Return a sorted list of all stored escalation IDs.

        Returns:
            Sorted list of escalation IDs.
        """
        return sorted(p.stem for p in self.escalations_dir.glob("*.json"))

    # ------------------------------------------------------------------
    # Debates
    # ------------------------------------------------------------------

    def write_debate(
        self,
        artifact_id: str,
        proposal: Proposal,
        challenge: Challenge,
        adjudication: Adjudication,
    ) -> Path:
        """Persist a complete debate (proposal + challenge + adjudication).

        Args:
            artifact_id: The target artifact ID this debate is about.
            proposal: Validated Proposal model.
            challenge: Validated Challenge model.
            adjudication: Validated Adjudication model.

        Returns:
            Path to the debate directory.
        """
        debate_dir = self.debates_dir / artifact_id
        debate_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(debate_dir / "proposal.json", proposal.model_dump_json(indent=2))
        _atomic_write_text(debate_dir / "challenge.json", challenge.model_dump_json(indent=2))
        _atomic_write_text(debate_dir / "adjudication.json", adjudication.model_dump_json(indent=2))
        return debate_dir

    def read_debate(self, artifact_id: str) -> DebateRecord:
        """Read a complete debate by target artifact ID.

        Args:
            artifact_id: The target artifact ID the debate is about.

        Returns:
            A DebateRecord containing the proposal, challenge, and adjudication.

        Raises:
            FileNotFoundError: If the debate directory or any of its files
                do not exist.
            ValueError: If any file is corrupt or fails validation.
        """
        debate_dir = self.debates_dir / artifact_id
        if not debate_dir.is_dir():
            raise FileNotFoundError(f"debate not found: {debate_dir}")

        proposal_text = _safe_read_json(debate_dir / "proposal.json", "debate proposal")
        challenge_text = _safe_read_json(debate_dir / "challenge.json", "debate challenge")
        adjudication_text = _safe_read_json(debate_dir / "adjudication.json", "debate adjudication")

        try:
            proposal = Proposal.model_validate_json(proposal_text)
        except ValidationError as exc:
            raise ValueError(f"debate proposal at {debate_dir} failed validation: {exc}") from exc
        try:
            challenge = Challenge.model_validate_json(challenge_text)
        except ValidationError as exc:
            raise ValueError(f"debate challenge at {debate_dir} failed validation: {exc}") from exc
        try:
            adjudication = Adjudication.model_validate_json(adjudication_text)
        except ValidationError as exc:
            raise ValueError(f"debate adjudication at {debate_dir} failed validation: {exc}") from exc

        return DebateRecord(proposal=proposal, challenge=challenge, adjudication=adjudication)

    def list_debates(self) -> list[str]:
        """Return a sorted list of all stored debate artifact IDs.

        Returns:
            Sorted list of artifact IDs that have debate records.
        """
        return sorted(
            d.name for d in self.debates_dir.iterdir()
            if d.is_dir() and (d / "proposal.json").is_file()
        )

    # ------------------------------------------------------------------
    # Micro plans
    # ------------------------------------------------------------------

    def write_micro_plan(self, plan: MicroPlan) -> Path:
        """Persist a micro plan as an artifact payload.

        Args:
            plan: Validated MicroPlan model.

        Returns:
            Path to the written file.
        """
        return self.write_artifact_payload(
            artifact_type=ArtifactType.MICRO_PLAN,
            artifact_id=plan.micro_plan_id,
            payload=plan.model_dump(mode="json"),
        )

    # ------------------------------------------------------------------
    # System contracts
    # ------------------------------------------------------------------

    def write_system_contract(self, contract: SystemContract) -> Path:
        """Persist a system contract.

        Args:
            contract: Validated SystemContract model.

        Returns:
            Path to the written file.
        """
        contract_dir = self.system_contracts_dir / contract.system_id / contract.system_version
        contract_dir.mkdir(parents=True, exist_ok=True)
        path = contract_dir / "contract.json"
        _atomic_write_text(path, contract.model_dump_json(indent=2))
        return path

    def read_system_contract(self, system_id: str, system_version: str) -> SystemContract:
        """Read a system contract by ID and version.

        Args:
            system_id: The system identifier (e.g. ``SYS-core``).
            system_version: Semver version string.

        Returns:
            The deserialized SystemContract.

        Raises:
            FileNotFoundError: If the contract does not exist.
            ValueError: If the file is corrupt or fails validation.
        """
        path = self.system_contracts_dir / system_id / system_version / "contract.json"
        text = _safe_read_json(path, "system contract")
        try:
            return SystemContract.model_validate_json(text)
        except ValidationError as exc:
            raise ValueError(f"system contract at {path} failed validation: {exc}") from exc

    def list_system_contracts(self) -> list[str]:
        """Return sorted list of ``system_id/version`` strings for all stored system contracts.

        Returns:
            Sorted list of ``system_id/version`` identifiers.
        """
        results: list[str] = []
        for contract_file in self.system_contracts_dir.rglob("contract.json"):
            version_dir = contract_file.parent
            system_dir = version_dir.parent
            results.append(f"{system_dir.name}/{version_dir.name}")
        return sorted(results)

    # ------------------------------------------------------------------
    # Service contracts
    # ------------------------------------------------------------------

    def write_service_contract(self, contract: ServiceContract) -> Path:
        """Persist a service contract.

        Args:
            contract: Validated ServiceContract model.

        Returns:
            Path to the written file.
        """
        contract_dir = self.service_contracts_dir / contract.service_id / contract.service_version
        contract_dir.mkdir(parents=True, exist_ok=True)
        path = contract_dir / "contract.json"
        _atomic_write_text(path, contract.model_dump_json(indent=2))
        return path

    def read_service_contract(self, service_id: str, service_version: str) -> ServiceContract:
        """Read a service contract by ID and version.

        Args:
            service_id: The service identifier (e.g. ``SVC-api``).
            service_version: Semver version string.

        Returns:
            The deserialized ServiceContract.

        Raises:
            FileNotFoundError: If the contract does not exist.
            ValueError: If the file is corrupt or fails validation.
        """
        path = self.service_contracts_dir / service_id / service_version / "contract.json"
        text = _safe_read_json(path, "service contract")
        try:
            return ServiceContract.model_validate_json(text)
        except ValidationError as exc:
            raise ValueError(f"service contract at {path} failed validation: {exc}") from exc

    def list_service_contracts(self) -> list[str]:
        """Return sorted list of ``service_id/version`` strings for all stored service contracts.

        Returns:
            Sorted list of ``service_id/version`` identifiers.
        """
        results: list[str] = []
        for contract_file in self.service_contracts_dir.rglob("contract.json"):
            version_dir = contract_file.parent
            service_dir = version_dir.parent
            results.append(f"{service_dir.name}/{version_dir.name}")
        return sorted(results)

    # ------------------------------------------------------------------
    # Class contracts
    # ------------------------------------------------------------------

    def write_class_contract(self, contract: ClassContract) -> Path:
        """Persist a class contract.

        Args:
            contract: Validated ClassContract model.

        Returns:
            Path to the written file.
        """
        contract_dir = self.class_contracts_dir / contract.class_id / contract.class_version
        contract_dir.mkdir(parents=True, exist_ok=True)
        path = contract_dir / "contract.json"
        _atomic_write_text(path, contract.model_dump_json(indent=2))
        return path

    def read_class_contract(self, class_id: str, class_version: str) -> ClassContract:
        """Read a class contract by ID and version.

        Args:
            class_id: The class identifier (e.g. ``CLS-validator``).
            class_version: Semver version string.

        Returns:
            The deserialized ClassContract.

        Raises:
            FileNotFoundError: If the contract does not exist.
            ValueError: If the file is corrupt or fails validation.
        """
        path = self.class_contracts_dir / class_id / class_version / "contract.json"
        text = _safe_read_json(path, "class contract")
        try:
            return ClassContract.model_validate_json(text)
        except ValidationError as exc:
            raise ValueError(f"class contract at {path} failed validation: {exc}") from exc

    def list_class_contracts(self) -> list[str]:
        """Return sorted list of ``class_id/version`` strings for all stored class contracts.

        Returns:
            Sorted list of ``class_id/version`` identifiers.
        """
        results: list[str] = []
        for contract_file in self.class_contracts_dir.rglob("contract.json"):
            version_dir = contract_file.parent
            class_dir = version_dir.parent
            results.append(f"{class_dir.name}/{version_dir.name}")
        return sorted(results)

    # ------------------------------------------------------------------
    # Module contracts
    # ------------------------------------------------------------------

    def write_module_contract(self, contract: MicroModuleContract) -> Path:
        """Persist a micro module contract.

        Args:
            contract: Validated MicroModuleContract model.

        Returns:
            Path to the written file.
        """
        module_dir = self.modules_dir / contract.module_id / contract.module_version
        module_dir.mkdir(parents=True, exist_ok=True)
        path = module_dir / "contract.json"
        _atomic_write_text(path, contract.model_dump_json(indent=2, by_alias=True))
        return path

    def read_module_contract(self, module_id: str, module_version: str) -> MicroModuleContract:
        """Read a micro module contract by ID and version.

        Args:
            module_id: The module identifier (e.g. ``MM-validator``).
            module_version: Semver version string.

        Returns:
            The deserialized MicroModuleContract.

        Raises:
            FileNotFoundError: If the contract does not exist.
            ValueError: If the file is corrupt or fails validation.
        """
        path = self.modules_dir / module_id / module_version / "contract.json"
        text = _safe_read_json(path, "module contract")
        try:
            return MicroModuleContract.model_validate_json(text)
        except ValidationError as exc:
            raise ValueError(f"module contract at {path} failed validation: {exc}") from exc

    def list_module_contracts(self) -> list[str]:
        """Return sorted list of ``module_id/version`` strings for all stored module contracts.

        Returns:
            Sorted list of ``module_id/version`` identifiers.
        """
        results: list[str] = []
        for contract_file in self.modules_dir.rglob("contract.json"):
            version_dir = contract_file.parent
            module_dir = version_dir.parent
            results.append(f"{module_dir.name}/{version_dir.name}")
        return sorted(results)

    # ------------------------------------------------------------------
    # Ship evidence
    # ------------------------------------------------------------------

    def write_ship_evidence(self, evidence: ShipEvidence) -> Path:
        """Persist ship evidence for a module.

        Args:
            evidence: Validated ShipEvidence model.

        Returns:
            Path to the written file.
        """
        module_dir = self.modules_dir / evidence.module_id / evidence.module_version
        module_dir.mkdir(parents=True, exist_ok=True)
        path = module_dir / "ship.json"
        _atomic_write_text(path, evidence.model_dump_json(indent=2))
        return path

    def read_ship_evidence(self, module_id: str, module_version: str) -> ShipEvidence:
        """Read ship evidence for a module by ID and version.

        Args:
            module_id: The module identifier.
            module_version: Semver version string.

        Returns:
            The deserialized ShipEvidence.

        Raises:
            FileNotFoundError: If the ship evidence does not exist.
            ValueError: If the file is corrupt or fails validation.
        """
        path = self.modules_dir / module_id / module_version / "ship.json"
        text = _safe_read_json(path, "ship evidence")
        try:
            return ShipEvidence.model_validate_json(text)
        except ValidationError as exc:
            raise ValueError(f"ship evidence at {path} failed validation: {exc}") from exc

    # ------------------------------------------------------------------
    # Module sealing
    # ------------------------------------------------------------------

    def seal_module(self, module_id: str, version: str) -> None:
        """Seal a module version, marking it immutable.

        Args:
            module_id: The module identifier.
            version: Semver version string.
        """
        module_version_dir = self.modules_dir / module_id / version
        seal_module_version(module_version_dir)
        mark_module_immutable(module_version_dir)

    # ------------------------------------------------------------------
    # Artifact envelopes (locked - immutable after creation)
    # ------------------------------------------------------------------

    def write_artifact(self, envelope: ArtifactEnvelope) -> Path:
        """Persist an artifact envelope.

        The envelope is immutable after creation; attempting to write
        an artifact that already exists raises ``ValueError``.

        Args:
            envelope: Validated ArtifactEnvelope model.

        Returns:
            Path to the written envelope file.

        Raises:
            ValueError: If an artifact with this ID already exists.
        """
        artifact_dir = self.artifacts_dir / envelope.artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        envelope_path = artifact_dir / "envelope.json"
        with _locked_file(envelope_path):
            if envelope_path.exists():
                raise ValueError(f"Artifact already exists: {envelope.artifact_id}")
            _atomic_write_text(envelope_path, envelope.model_dump_json(indent=2))
            payload_path = artifact_dir / "payload.json"
            _atomic_write_text(payload_path, json.dumps(envelope.payload, indent=2, sort_keys=True))
        return envelope_path

    def write_artifact_payload(self, artifact_type: ArtifactType, artifact_id: str, payload: dict[str, Any]) -> Path:
        """Persist an artifact payload file (typed by artifact type).

        Args:
            artifact_type: The artifact type enum.
            artifact_id: The artifact identifier.
            payload: JSON-serializable dict to persist.

        Returns:
            Path to the written file.
        """
        artifact_dir = self.artifacts_dir / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"{artifact_type.value.lower()}.json"
        _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str))
        return path

    def read_artifact(self, artifact_id: str) -> ArtifactEnvelope:
        """Read an artifact envelope by ID.

        Args:
            artifact_id: The artifact identifier.

        Returns:
            The deserialized ArtifactEnvelope.

        Raises:
            FileNotFoundError: If the artifact does not exist.
            ValueError: If the file is corrupt or fails validation.
        """
        path = self.artifacts_dir / artifact_id / "envelope.json"
        text = _safe_read_json(path, f"artifact {artifact_id}")
        try:
            return ArtifactEnvelope.model_validate_json(text)
        except ValidationError as exc:
            raise ValueError(f"artifact {artifact_id} at {path} failed validation: {exc}") from exc

    def list_artifacts(self) -> list[str]:
        """Return a sorted list of all stored artifact IDs.

        Returns:
            Sorted list of artifact IDs.
        """
        return sorted(
            d.name for d in self.artifacts_dir.iterdir()
            if d.is_dir() and (d / "envelope.json").is_file()
        )

    def transition_artifact_status(self, artifact_id: str, new_status: ArtifactStatus) -> ArtifactEnvelope:
        """Transition an artifact to a new status under an exclusive lock.

        The read-modify-write is performed atomically under the lock to
        prevent race conditions.

        Args:
            artifact_id: The artifact identifier.
            new_status: The target status.

        Returns:
            The updated ArtifactEnvelope.

        Raises:
            FileNotFoundError: If the artifact does not exist.
            ValueError: If the transition is not allowed by the state machine.
        """
        path = self.artifacts_dir / artifact_id / "envelope.json"
        with _locked_file(path):
            text = _safe_read_json(path, f"artifact {artifact_id}")
            try:
                envelope = ArtifactEnvelope.model_validate_json(text)
            except ValidationError as exc:
                raise ValueError(f"artifact {artifact_id} at {path} failed validation: {exc}") from exc

            allowed = ARTIFACT_STATUS_TRANSITIONS[envelope.status]
            if new_status not in allowed:
                raise ValueError(
                    f"Illegal artifact status transition for {artifact_id}: "
                    f"{envelope.status.value} -> {new_status.value}"
                )
            envelope.status = new_status
            _atomic_write_text(path, envelope.model_dump_json(indent=2))
        return envelope


# ---------------------------------------------------------------------------
# ArtifactStoreService
# ---------------------------------------------------------------------------

class ArtifactStoreService:
    """Universal envelope service for all persisted artifact writes.

    Provides a high-level API over ``FactoryStateStore`` for managing
    artifact lifecycle transitions.
    """

    def __init__(self, store: FactoryStateStore) -> None:
        self.store = store

    def create(self, envelope: ArtifactEnvelope) -> Path:
        """Create a new artifact.

        Args:
            envelope: The artifact envelope to persist.

        Returns:
            Path to the written envelope file.

        Raises:
            ValueError: If an artifact with this ID already exists.
        """
        return self.store.write_artifact(envelope)

    def challenge(self, artifact_id: str) -> ArtifactEnvelope:
        """Transition an artifact to CHALLENGED status.

        Args:
            artifact_id: The artifact identifier.

        Returns:
            The updated ArtifactEnvelope.
        """
        return self.store.transition_artifact_status(artifact_id, ArtifactStatus.CHALLENGED)

    def adjudicate(self, artifact_id: str) -> ArtifactEnvelope:
        """Transition an artifact to ADJUDICATED status.

        Args:
            artifact_id: The artifact identifier.

        Returns:
            The updated ArtifactEnvelope.
        """
        return self.store.transition_artifact_status(artifact_id, ArtifactStatus.ADJUDICATED)

    def ship(self, artifact_id: str) -> ArtifactEnvelope:
        """Transition an artifact to SHIPPED status.

        Args:
            artifact_id: The artifact identifier.

        Returns:
            The updated ArtifactEnvelope.
        """
        return self.store.transition_artifact_status(artifact_id, ArtifactStatus.SHIPPED)

    def deprecate(self, artifact_id: str) -> ArtifactEnvelope:
        """Transition an artifact to DEPRECATED status.

        Args:
            artifact_id: The artifact identifier.

        Returns:
            The updated ArtifactEnvelope.
        """
        return self.store.transition_artifact_status(artifact_id, ArtifactStatus.DEPRECATED)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def sanitize_project_id(project_id: str) -> str:
    """Sanitize a project ID for use as a filesystem path component.

    Args:
        project_id: Raw project identifier.

    Returns:
        A filesystem-safe version of the project ID, truncated to 128 chars.

    Raises:
        ValueError: If the project ID is empty or contains no safe characters.
    """
    value = project_id.strip()
    if not value:
        raise ValueError("project_id must be non-empty")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    if not value:
        raise ValueError("project_id contains no filesystem-safe characters")
    return value[:128]


def project_scoped_root(root: Path, project_id: str | None) -> Path:
    """Return the project-scoped root directory.

    If *project_id* is ``None``, returns *root* unchanged.  Otherwise
    returns ``root / "projects" / sanitize_project_id(project_id)``
    unless *root* is already the scoped path.

    Args:
        root: Base state store root.
        project_id: Optional project identifier.

    Returns:
        The resolved project-scoped root path.
    """
    if project_id is None:
        return root
    slug = sanitize_project_id(project_id)
    if root.name == slug and root.parent.name == "projects":
        return root
    return root / "projects" / slug


def build_project_state(spec: ProductSpec, *, project_id: str | None = None) -> ProjectState:
    """Build a fresh ProjectState from a ProductSpec.

    Iterates all tasks in the spec and creates corresponding
    ``ProjectTaskState`` entries with ``PENDING`` status and
    monotonically increasing ``declaration_order``.

    Args:
        spec: The validated product specification.
        project_id: Optional project identifier; defaults to ``spec.spec_id``.

    Returns:
        A new ProjectState with all tasks in PENDING status.
    """
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
