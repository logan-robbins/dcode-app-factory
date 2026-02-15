from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from pathlib import Path

from .debate import Debate, DebateResult
from .models import (
    ContextPack,
    Epic,
    IOContractSketch,
    MicroModuleContract,
    Pillar,
    Story,
    StructuredSpec,
    Task,
    TaskStatus,
)
from .registry import CodeIndex
from .utils import build_context_pack, load_agent_config, slugify_name, validate_task_dependency_dag

logger = logging.getLogger(__name__)


class ProductLoop:
    """Transforms raw markdown spec into structured plan with concrete task contracts."""

    def __init__(self, raw_spec: str, config_dir: str = "agent_configs/product_loop") -> None:
        self.raw_spec = raw_spec
        self.config_dir = Path(config_dir)
        self.agent_configs = self._load_configs()

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
        return sections or ["Factory Architecture", "Loops", "Context Isolation"]

    def _task_from_section(self, section: str, index: int) -> Task:
        slug = slugify_name(section) or f"section-{index}"
        task_id = f"TASK-{index:03d}-{slug}"
        sketch = IOContractSketch(
            inputs=[f"Structured input extracted from section '{section}'"],
            outputs=[f"Validated deliverable artifacts for '{section}'"],
            error_surfaces=["Invalid or conflicting section requirements"],
            effects=["Writes task artifact and implementation diff"],
            modes=["normal", "halt_on_conflict"],
        )
        sketch.validate_complete()
        return Task(
            task_id=task_id,
            name=f"Implement {section}",
            description=f"Implement and verify section '{section}' using contract-first design.",
            subtasks=[
                f"Define module contract for {section}",
                f"Implement and validate behavior for {section}",
            ],
            acceptance_criteria=[
                "returns deterministic output under identical inputs",
                "validates and rejects unsupported mode or malformed input",
            ],
            io_contract_sketch=sketch,
            depends_on=[],
        )

    def run(self) -> StructuredSpec:
        sections = self._extract_sections()
        tasks = [self._task_from_section(section, idx + 1) for idx, section in enumerate(sections[:5])]
        for idx in range(1, len(tasks)):
            tasks[idx].depends_on.append(tasks[idx - 1].task_id)
        story = Story(
            story_id="STORY-001",
            name="Implement spec-aligned factory behaviors",
            description="Delivers core factory behavior from the authored specification.",
            user_facing_behavior="Operators can run end-to-end task execution with deterministic outcomes.",
            tasks=tasks,
        )
        epic = Epic(
            epic_id="EPIC-001",
            name="Factory Core Loop Execution",
            description="Implements product, project, and engineering loops with contract rigor.",
            success_criteria=["All generated tasks are executable in dependency order."],
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
            constraints=["Single-task dispatch", "Halt on failed adjudication"],
            pillars=[pillar],
        )
        validate_task_dependency_dag(spec)
        return spec


class EngineeringLoop:
    """Runs strict Propose->Challenge->Adjudicate flow for one task."""

    def __init__(self, task: Task, code_index: CodeIndex, config_dir: str = "agent_configs/engineering_loop") -> None:
        self.task = task
        self.code_index = code_index
        self.config_dir = Path(config_dir)
        self.agent_configs = {
            path.stem: load_agent_config(path)
            for path in sorted(self.config_dir.glob("*.json"))
        }

    def _proposer(self, task_prompt: str, context: ContextPack) -> str:
        _ = task_prompt, context, self.agent_configs.get("proposer")
        return f"proposal: implement {self.task.task_id} with strict contract separation"

    def _challenger(self, proposal: str, context: ContextPack) -> str:
        _ = context, self.agent_configs.get("challenger")
        if "strict contract separation" not in proposal:
            return "fail: proposal does not enforce context isolation"
        return "challenge: no contract violations found"

    def _arbiter(self, assessment: str, context: ContextPack) -> str:
        _ = context, self.agent_configs.get("arbiter")
        return "PASS" if "fail:" not in assessment.lower() else "FAIL"

    def _refine_contract(self) -> MicroModuleContract:
        return MicroModuleContract(
            module_id=self.task.task_id,
            name=self.task.name,
            description=self.task.description,
            inputs=[{"name": "task_context", "type": "ContextPack", "description": "task scoped context pack"}],
            outputs=[{"name": "artifact", "type": "dict", "description": "implementation artifact metadata"}],
            error_surfaces=[{"code": "VALIDATION_ERROR", "condition": "contract or context violation"}],
            effects=["registers immutable contract in CodeIndex"],
            modes=["normal", "halt_on_failure"],
            depends_on=self.task.depends_on,
        )

    def run(self) -> bool:
        self.task.status = TaskStatus.IN_PROGRESS
        context = build_context_pack(self.task.task_id, self.task.description, stage="engineering_loop")
        debate = Debate(self._proposer, self._challenger, self._arbiter)
        outcome, _trace = debate.run(self.task.description, context)
        if outcome is DebateResult.FAIL:
            self.task.status = TaskStatus.HALTED
            return False

        self.task.contract = self._refine_contract()
        self.code_index.register(self.task.contract)
        self.task.status = TaskStatus.COMPLETED
        return True


class ProjectLoop:
    """Deterministic task state-machine with dependency-aware dispatch."""

    def __init__(self, spec: StructuredSpec, code_index: CodeIndex | None = None, config_dir: str = "agent_configs/project_loop") -> None:
        self.spec = spec
        self.code_index = code_index if code_index is not None else CodeIndex()
        self.config_dir = Path(config_dir)
        self.agent_configs = {
            path.stem: load_agent_config(path)
            for path in sorted(self.config_dir.glob("*.json"))
        }

    def _flatten_tasks(self) -> list[Task]:
        return [
            task
            for pillar in self.spec.pillars
            for epic in pillar.epics
            for story in epic.stories
            for task in story.tasks
        ]

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

    def run(self) -> bool:
        validate_task_dependency_dag(self.spec)
        tasks = self._topological_order(self._flatten_tasks())
        for task in tasks:
            ok = EngineeringLoop(task=task, code_index=self.code_index).run()
            if not ok:
                logger.error("Project halted at %s", task.task_id)
                return False
        return True

    def export_state(self) -> str:
        snapshot = {
            "tasks": [
                {"task_id": task.task_id, "status": task.status, "depends_on": task.depends_on}
                for task in self._flatten_tasks()
            ],
            "registered_contracts": [entry.module_slug for entry in self.code_index.list_entries()],
        }
        return json.dumps(snapshot, indent=2, default=str)
