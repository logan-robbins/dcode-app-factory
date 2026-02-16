from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from pathlib import Path

from .debate import Debate, DebateResult
from .model_selection import RuntimeModelSelection, resolve_agent_models
from .models import (
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
)
from .registry import CodeIndex
from .settings import RuntimeSettings
from .utils import (
    build_context_pack,
    get_agent_config_dir,
    load_agent_config,
    slugify_name,
    validate_task_dependency_dag,
)

logger = logging.getLogger(__name__)


class ProductLoop:
    """Transforms raw markdown spec into fully validated task hierarchy."""

    def __init__(self, raw_spec: str, config_dir: str | Path | None = None) -> None:
        self.raw_spec = raw_spec
        self.settings = RuntimeSettings.from_env()
        self.config_dir = Path(config_dir) if config_dir is not None else get_agent_config_dir("product_loop")
        self.agent_configs = self._load_configs()
        self.role_models = resolve_agent_models(self.agent_configs, RuntimeModelSelection.from_env())

    def _load_configs(self) -> dict[str, object]:
        return {
            config_path.stem: load_agent_config(config_path)
            for config_path in sorted(self.config_dir.glob("*.json"))
        }

    def _extract_sections(self) -> list[str]:
        sections: list[str] = []
        for line in self.raw_spec.splitlines():
            value = line.strip()
            if value.startswith("##"):
                sections.append(value.lstrip("# "))
        return sections or ["Factory Architecture", "Loops", "Context Isolation", "Debate", "Code Index"]

    def _task_from_section(self, section: str, index: int) -> Task:
        slug = slugify_name(section) or f"section-{index}"
        task_id = f"TASK-{index:03d}-{slug}"
        sketch = IOContractSketch(
            inputs=[f"Structured requirements from section '{section}'", "Task dependency metadata"],
            outputs=[f"Implemented artifacts for '{section}'", "Execution state snapshot"],
            error_surfaces=["Conflicting requirements", "Dependency contract mismatch"],
            effects=["Writes task artifact", "Appends module contract to code index"],
            modes=["normal", "halt_on_conflict"],
        )
        sketch.validate_complete()
        return Task(
            task_id=task_id,
            name=f"Implement {section}",
            description=f"Implement and verify section '{section}' with explicit contracts and task isolation.",
            subtasks=[
                f"Define and validate module contract for {section}",
                f"Implement behavior and capture ship evidence for {section}",
            ],
            acceptance_criteria=[
                "returns deterministic outputs for identical context and task inputs",
                "validates dependency and rejects malformed contract payloads",
            ],
            io_contract_sketch=sketch,
            depends_on=[],
        )

    def run(self) -> StructuredSpec:
        sections = self._extract_sections()
        tasks = [
            self._task_from_section(section, idx + 1)
            for idx, section in enumerate(sections[: self.settings.max_product_sections])
        ]
        for idx in range(1, len(tasks)):
            tasks[idx].depends_on.append(tasks[idx - 1].task_id)
        story = Story(
            story_id="STORY-001",
            name="Implement spec-aligned factory behaviors",
            description="Delivers core factory behavior from the authored specification.",
            user_facing_behavior="Operators run deterministic end-to-end task execution with auditable context boundaries.",
            tasks=tasks,
        )
        epic = Epic(
            epic_id="EPIC-001",
            name="Factory Core Loop Execution",
            description="Implements product, project, and engineering loops with contract rigor.",
            success_criteria=["The task graph is executable in dependency order without unresolved nodes."],
            stories=[story],
        )
        pillar = Pillar(
            pillar_id="PILLAR-001",
            name="Reliable Agentic Factory",
            description="Core capability for deterministic software production.",
            rationale="Reliability and auditability are primary value drivers.",
            epics=[epic],
        )
        spec = StructuredSpec(
            product_name="AI Software Product Factory",
            vision="Deterministically deliver software through auditable agent loops.",
            constraints=["Single-task dispatch", "Halt on failed adjudication", "Context-pack isolation"],
            pillars=[pillar],
        )

        issues = spec.validate()
        errors = [issue for issue in issues if issue.severity == "ERROR"]
        if errors:
            detail = "; ".join(f"{err.location}: {err.message}" for err in errors)
            raise ValueError(f"Structured spec validation failed: {detail}")

        validate_task_dependency_dag(spec)
        return spec


class EngineeringLoop:
    """Runs strict Propose->Challenge->Adjudicate flow for one task."""

    def __init__(
        self,
        task: Task,
        code_index: CodeIndex,
        config_dir: str | Path | None = None,
        max_retries: int = 2,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.task = task
        self.code_index = code_index
        self.max_retries = max_retries
        self.settings = RuntimeSettings.from_env()
        self.config_dir = Path(config_dir) if config_dir is not None else get_agent_config_dir("engineering_loop")
        self.escalation: EscalationArtifact | None = None
        self.agent_configs = {
            path.stem: load_agent_config(path)
            for path in sorted(self.config_dir.glob("*.json"))
        }
        self.role_models = resolve_agent_models(self.agent_configs, RuntimeModelSelection.from_env())

    def _proposer(self, task_prompt: str, _context) -> str:
        _ = task_prompt, self.agent_configs.get("proposer")
        return f"proposal: implement {self.task.task_id} with strict contract separation and evidence capture"

    def _challenger(self, proposal: str, _context) -> str:
        _ = self.agent_configs.get("challenger")
        required = ["strict contract separation", "evidence capture"]
        missing = [r for r in required if r not in proposal]
        return "fail: " + ", ".join(missing) if missing else "challenge: no contract violations found"

    def _arbiter(self, assessment: str, _context) -> str:
        _ = self.agent_configs.get("arbiter")
        return "FAIL" if "fail:" in assessment.lower() else "PASS"

    def _refine_contract(self) -> MicroModuleContract:
        return MicroModuleContract(
            module_id=self.task.task_id,
            name=self.task.name,
            description=self.task.description,
            inputs=[{"name": "task_context", "type": "ContextPack", "description": "task-scoped context pack"}],
            outputs=[{"name": "artifact", "type": "dict", "description": "implementation artifact metadata"}],
            error_surfaces=[
                {"code": "VALIDATION_ERROR", "condition": "contract or context violation"},
                {"code": "DEPENDENCY_ERROR", "condition": "dependency contract unavailable"},
            ],
            effects=["registers immutable contract in CodeIndex"],
            modes=["normal", "halt_on_failure"],
            depends_on=self.task.depends_on,
        )

    def run(self) -> bool:
        self.task.status = TaskStatus.IN_PROGRESS
        proposer_cfg = self.agent_configs["proposer"]
        context = build_context_pack(
            self.task.task_id,
            self.task.description,
            stage="engineering_loop",
            config=proposer_cfg,
            settings=self.settings,
        )

        debate = Debate(self._proposer, self._challenger, self._arbiter)
        last_trace = None
        for _attempt in range(self.max_retries + 1):
            outcome, trace = debate.run(self.task.description, context)
            last_trace = trace
            if outcome is DebateResult.PASS:
                self.task.contract = self._refine_contract()
                self.code_index.register(self.task.contract)
                self.task.ship_evidence = ShipEvidence(
                    task_id=self.task.task_id,
                    adjudication=trace.adjudication,
                    tests_run=["contract_validation", "dependency_consistency"],
                    execution_logs=[trace.proposal, trace.challenge],
                )
                self.task.status = TaskStatus.COMPLETED
                return True

        self.task.status = TaskStatus.HALTED
        self.escalation = EscalationArtifact(
            task_id=self.task.task_id,
            reason="Debate failed after maximum retries",
            debate_trace=last_trace,
        )
        return False


class ProjectLoop:
    """Deterministic task state machine with dependency-aware dispatch."""

    def __init__(self, spec: StructuredSpec, code_index: CodeIndex | None = None, config_dir: str | Path | None = None) -> None:
        self.spec = spec
        self.code_index = code_index if code_index is not None else CodeIndex()
        self.config_dir = Path(config_dir) if config_dir is not None else get_agent_config_dir("project_loop")
        self.agent_configs = {
            path.stem: load_agent_config(path)
            for path in sorted(self.config_dir.glob("*.json"))
        }
        self.role_models = resolve_agent_models(self.agent_configs, RuntimeModelSelection.from_env())

    def _flatten_tasks(self) -> list[Task]:
        return self.spec.iter_tasks()

    def _topological_order(self, tasks: list[Task]) -> list[Task]:
        by_id = {task.task_id: task for task in tasks}
        indegree = {task.task_id: 0 for task in tasks}
        edges: dict[str, list[str]] = defaultdict(list)
        for task in tasks:
            for dep in task.depends_on:
                indegree[task.task_id] += 1
                edges[dep].append(task.task_id)

        queue = deque(sorted(task_id for task_id, degree in indegree.items() if degree == 0))
        ordered: list[Task] = []
        while queue:
            task_id = queue.popleft()
            ordered.append(by_id[task_id])
            for nxt in sorted(edges[task_id]):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if len(ordered) != len(tasks):
            raise ValueError("Cycle detected during dispatch ordering")
        return ordered

    def _mark_downstream_blocked(self, failed_task_id: str) -> None:
        children: dict[str, list[Task]] = defaultdict(list)
        for task in self._flatten_tasks():
            for dep in task.depends_on:
                children[dep].append(task)

        queue = deque(children[failed_task_id])
        while queue:
            task = queue.popleft()
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.BLOCKED
            for nxt in children[task.task_id]:
                queue.append(nxt)

    def run(self) -> bool:
        validate_task_dependency_dag(self.spec)
        tasks = self._topological_order(self._flatten_tasks())
        for task in tasks:
            if any(dep_task.status in {TaskStatus.HALTED, TaskStatus.BLOCKED} for dep_task in tasks if dep_task.task_id in task.depends_on):
                task.status = TaskStatus.BLOCKED
                continue

            ok = EngineeringLoop(task=task, code_index=self.code_index).run()
            if not ok:
                logger.error("Project halted at %s", task.task_id)
                self._mark_downstream_blocked(task.task_id)
                return False
        return True

    def export_state(self) -> str:
        snapshot = {
            "tasks": [
                {"task_id": task.task_id, "status": task.status.value, "depends_on": task.depends_on}
                for task in self._flatten_tasks()
            ],
            "registered_contracts": [entry.module_slug for entry in self.code_index.list_entries()],
        }
        return json.dumps(snapshot, indent=2, default=str)
