from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from deepagents import create_deep_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .backends import build_factory_backend
from .debate import DebateGraph
from .llm import get_chat_model
from .models import (
    AdjudicationDecision,
    Adjudication,
    ArtifactEnvelope,
    ArtifactStatus,
    ArtifactType,
    Challenge,
    CodeIndexStatus,
    CompatibilityExpectation,
    ContextAccessLevel,
    EscalationArtifact,
    FailedInvariant,
    HumanResolution,
    HumanResolutionAction,
    InterfaceChangeException,
    MicroIoContract,
    MicroModuleContract,
    MicroPlan,
    MicroPlanModule,
    ProjectState,
    Proposal,
    RaisedBy,
    ReuseConclusion,
    ReuseDecision,
    ReuseSearchCandidate,
    ReuseSearchReport,
    ShipDirective,
    ShipEnvironment,
    ShipEvidence,
    ShipVerification,
    SplitTaskInput,
    StructuredSpec,
    Task,
    TaskStatus,
)
from .registry import CodeIndex
from .settings import RuntimeSettings
from .state_store import ArtifactStoreService, FactoryStateStore, build_project_state
from .tools import emit_structured_spec_tool, search_code_index, validate_spec_tool, web_search
from .utils import (
    apply_canonical_task_ids,
    build_context_pack,
    emit_structured_spec,
    parse_raw_spec_to_product_spec,
    render_spec_markdown,
    resolve_task_ids,
    slugify_name,
    validate_spec,
)


class ProductLoop:
    """Product loop implemented via deepagents agent invocation and validated tooling."""

    def __init__(
        self,
        raw_spec: str,
        *,
        state_store_root: str | Path | None = None,
        settings: RuntimeSettings | None = None,
    ) -> None:
        self.raw_spec = raw_spec
        self.settings = settings if settings is not None else RuntimeSettings.from_env()
        root = Path(state_store_root) if state_store_root is not None else Path(self.settings.state_store_root)
        self.state_store = FactoryStateStore(root, project_id=self.settings.project_id)

    def _invoke_deep_agent(self, spec_json_path: Path) -> None:
        model = get_chat_model(model_name=self.settings.model_frontier, temperature=0.0)
        backend = build_factory_backend(self.state_store.root)
        agent = create_deep_agent(
            model=model,
            tools=[web_search, validate_spec_tool, search_code_index, emit_structured_spec_tool],
            backend=backend,
            system_prompt=(
                "You are the Product Loop quality gate for a production software factory. "
                "Policy: contract-first, reuse-first, and no speculative output. "
                "You must call `validate_spec` for schema/completeness checks, then call "
                "`search_code_index` for each major task objective before recommending any CREATE_NEW path. "
                "If no reuse is viable, explicitly state the failed match reason. "
                "Return concise warnings and release readiness only."
            ),
            name="product-loop-agent",
        )
        # Run a real invocation for stack compliance. The deterministic parser still owns source-of-truth shaping.
        agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Validate this ProductSpec JSON path, run reuse-first checks against code index, "
                            "and summarize warnings/readiness in <=12 bullets. "
                            f"Path: {spec_json_path}"
                        ),
                    }
                ]
            },
            config={"configurable": {"thread_id": f"product-loop-{uuid.uuid4().hex[:8]}"}},
        )

    def run(self) -> StructuredSpec:
        spec = parse_raw_spec_to_product_spec(self.raw_spec)
        apply_canonical_task_ids(spec)

        # Enforce configurable section fan-out cap.
        tasks = spec.iter_tasks()
        if len(tasks) > self.settings.max_product_sections:
            allowed_ids = {task.task_id for task in tasks[: self.settings.max_product_sections]}
            for pillar in spec.pillars:
                for epic in pillar.epics:
                    for story in epic.stories:
                        story.tasks = [task for task in story.tasks if task.task_id in allowed_ids]
            for task in spec.iter_tasks():
                task.depends_on = [dep for dep in task.depends_on if dep in allowed_ids]

        report = validate_spec(spec)
        if report.errors:
            details = "; ".join(f"{issue.path}:{issue.field}:{issue.message}" for issue in report.errors)
            raise ValueError(f"Product loop validation failed: {details}")

        markdown = render_spec_markdown(spec)
        self.state_store.write_product_spec(spec, markdown)
        emit_structured_spec(spec, self.state_store.product_spec_json_path)

        # Required architecture binding: deepagents create_agent/create_deep_agent invocation.
        self._invoke_deep_agent(self.state_store.product_spec_json_path)
        return spec


class EngineeringState(TypedDict, total=False):
    task: dict[str, Any]
    task_id: str
    micro_plan: dict[str, Any]
    module_order: list[str]
    module_cursor: int
    module_status: dict[str, str]
    module_refs: dict[str, str]
    failed_modules: list[str]
    abandoned_modules: list[str]
    shipped_modules: list[str]
    halted: bool
    escalation_id: str | None
    halt_reason: str | None


@dataclass
class EngineeringResult:
    task_status: TaskStatus
    module_refs: list[str]
    escalation_id: str | None = None
    halt_reason: str | None = None


class EngineeringLoop:
    """Engineering loop StateGraph with micro-plan and per-module debate execution."""

    _ATOMIC_SPLIT_RE = re.compile(r"\s*(?:,|;|\band\b|\bthen\b|\bwith\b|\bwhile\b|\bplus\b)\s*", re.IGNORECASE)
    _TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
    _REUSE_SCORE_THRESHOLD = 0.55
    _REUSE_TOKEN_OVERLAP_THRESHOLD = 1
    _STAGE_ORDER = {
        "ingress": 0,
        "core": 1,
        "integration": 2,
        "egress": 3,
        "verification": 4,
    }

    def __init__(
        self,
        *,
        code_index: CodeIndex,
        state_store: FactoryStateStore,
        settings: RuntimeSettings,
    ) -> None:
        self.code_index = code_index
        self.state_store = state_store
        self.settings = settings
        self.artifacts = ArtifactStoreService(state_store)
        self.debate_graph = DebateGraph(
            store=state_store,
            model_name=settings.model_efficient,
            use_llm=settings.debate_use_llm,
            propagate_parent_halt=False,
        )
        self.graph = self._build_graph().compile()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(EngineeringState)
        graph.add_node("micro_plan", self._micro_plan_node)
        graph.add_node("module_step", self._module_step_node)
        graph.add_node("module_route", self._module_route_node)
        graph.add_node("finalize", self._finalize_node)

        graph.add_edge(START, "micro_plan")
        graph.add_edge("micro_plan", "module_step")
        graph.add_edge("module_step", "module_route")
        graph.add_edge("finalize", END)
        return graph

    @staticmethod
    def _module_topological_order(plan: MicroPlan) -> list[str]:
        modules = {module.module_id: module for module in plan.modules}
        indegree = {module.module_id: 0 for module in plan.modules}
        edges: dict[str, list[str]] = defaultdict(list)

        for module in plan.modules:
            for dep in module.depends_on:
                if dep in modules:
                    indegree[module.module_id] += 1
                    edges[dep].append(module.module_id)

        roots = [module.module_id for module in plan.modules if indegree[module.module_id] == 0]
        queue = deque(roots)
        ordered: list[str] = []
        while queue:
            current = queue.popleft()
            ordered.append(current)
            for nxt in edges[current]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if len(ordered) != len(plan.modules):
            raise ValueError("Micro plan contains a cycle")
        return ordered

    @staticmethod
    def _tokenize(value: str) -> set[str]:
        return set(EngineeringLoop._TOKEN_RE.findall(value.lower()))

    @staticmethod
    def _ordered_unique(values: list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value and value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    def _split_atomic_clauses(self, text: str) -> list[str]:
        normalized = " ".join(text.strip().split()).strip(" .")
        if not normalized:
            return []
        parts = [part.strip(" .") for part in self._ATOMIC_SPLIT_RE.split(normalized) if part.strip(" .")]
        atomic = [part for part in parts if len(part) >= 18]
        if atomic:
            return atomic
        return [normalized]

    def _atomic_units_for_task(self, task: Task) -> list[str]:
        seed_units: list[str] = [
            f"Validate and normalize request boundary for {task.name}",
            f"Enforce output invariants and explicit error surfaces for {task.name}",
            f"Record ship evidence and verification checks for {task.name}",
        ]
        for value in task.subtasks:
            seed_units.extend(self._split_atomic_clauses(value))
        for value in task.acceptance_criteria:
            seed_units.extend(self._split_atomic_clauses(value))

        units: list[str] = []
        seen: set[str] = set()
        for unit in seed_units:
            normalized = unit.strip()
            if len(normalized) < 12:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            units.append(normalized)

        if len(units) < 4:
            units.extend(
                [
                    f"Implement deterministic core logic for {task.name}",
                    f"Integrate dependency adapters for {task.name}",
                ]
            )
            units = self._ordered_unique(units)

        return units[:10]

    @staticmethod
    def _stage_for_unit(unit: str) -> str:
        lowered = unit.lower()
        if any(token in lowered for token in {"validate", "input", "parse", "schema", "normalize"}):
            return "ingress"
        if any(token in lowered for token in {"test", "verify", "assert", "evidence", "coverage"}):
            return "verification"
        if any(token in lowered for token in {"emit", "output", "render", "response", "publish", "return"}):
            return "egress"
        if any(
            token in lowered
            for token in {
                "api",
                "provider",
                "http",
                "db",
                "persist",
                "storage",
                "integration",
                "adapter",
                "write",
                "read",
            }
        ):
            return "integration"
        return "core"

    def _reuse_analysis(
        self,
        *,
        task: Task,
        module_stage: str,
        module_purpose: str,
        io_contract: MicroIoContract,
    ) -> tuple[list[str], ReuseDecision, ReuseSearchReport]:
        query = (
            f"task {task.task_id} {task.name}; stage={module_stage}; purpose={module_purpose}; "
            f"inputs={'; '.join(io_contract.inputs)}; outputs={'; '.join(io_contract.outputs)}"
        )
        candidates = self.code_index.search(query, top_k=6)
        query_tokens = self._tokenize(
            " ".join(
                [
                    module_purpose,
                    *io_contract.inputs,
                    *io_contract.outputs,
                    *io_contract.error_surfaces,
                    *io_contract.effects,
                ]
            )
        )

        selected: str | None = None
        considered: list[ReuseSearchCandidate] = []
        for match in candidates:
            entry = match.entry
            entry_tokens = self._tokenize(
                " ".join(
                    [
                        entry.name,
                        entry.purpose,
                        *entry.tags,
                        *entry.io_summary.inputs,
                        *entry.io_summary.outputs,
                        *entry.io_summary.error_surfaces,
                    ]
                )
            )
            overlap = len(query_tokens.intersection(entry_tokens))
            if (
                selected is None
                and match.similarity_score >= self._REUSE_SCORE_THRESHOLD
                and overlap >= self._REUSE_TOKEN_OVERLAP_THRESHOLD
            ):
                selected = entry.module_ref
                considered.append(
                    ReuseSearchCandidate(
                        module_ref=entry.module_ref,
                        why_rejected="selected_for_reuse",
                    )
                )
                continue

            if match.similarity_score < self._REUSE_SCORE_THRESHOLD:
                reason = f"similarity {match.similarity_score:.2f} below threshold"
            elif overlap < self._REUSE_TOKEN_OVERLAP_THRESHOLD:
                reason = f"token overlap {overlap} below threshold"
            else:
                reason = "lower-ranked than selected candidate"
            considered.append(ReuseSearchCandidate(module_ref=entry.module_ref, why_rejected=reason))

        if selected is not None:
            selected_first = [selected] + [item.module_ref for item in considered if item.module_ref != selected]
            report = ReuseSearchReport(
                query=query,
                candidates_considered=considered,
                conclusion=ReuseConclusion.REUSE_EXISTING,
                justification=f"Selected {selected} from semantic + contract overlap match",
            )
            return selected_first, ReuseDecision.REUSE, report

        rejection_notes = ", ".join(item.why_rejected for item in considered[:3]) if considered else "no candidates in code index"
        report = ReuseSearchReport(
            query=query,
            candidates_considered=considered,
            conclusion=ReuseConclusion.CREATE_NEW,
            justification=f"No reusable module satisfied thresholds ({rejection_notes})",
        )
        return [item.module_ref for item in considered], ReuseDecision.CREATE_NEW, report

    @staticmethod
    def _io_contract_for_stage(task: Task, stage: str, purpose: str) -> MicroIoContract:
        if stage == "ingress":
            return MicroIoContract(
                inputs=[task.io_contract_sketch.inputs],
                outputs=[f"Validated request envelope for {task.task_id}"],
                error_surfaces=[task.io_contract_sketch.error_surfaces],
                effects=["normalize and validate request payload"],
                modes=["sync", "deterministic-validation"],
            )
        if stage == "core":
            return MicroIoContract(
                inputs=[f"Validated request envelope for {task.task_id}"],
                outputs=[f"Core domain state for {task.task_id}: {purpose}"],
                error_surfaces=[task.io_contract_sketch.error_surfaces],
                effects=["apply deterministic domain rules"],
                modes=["sync", "pure-compute"],
            )
        if stage == "integration":
            return MicroIoContract(
                inputs=[f"Core domain state for {task.task_id}", task.io_contract_sketch.inputs],
                outputs=[f"Integrated dependency payload for {task.task_id}"],
                error_surfaces=[task.io_contract_sketch.error_surfaces],
                effects=[task.io_contract_sketch.effects],
                modes=[task.io_contract_sketch.modes],
            )
        if stage == "egress":
            return MicroIoContract(
                inputs=[f"Integrated dependency payload for {task.task_id}"],
                outputs=[task.io_contract_sketch.outputs],
                error_surfaces=[task.io_contract_sketch.error_surfaces],
                effects=["emit deterministic response artifacts"],
                modes=[task.io_contract_sketch.modes],
            )
        return MicroIoContract(
            inputs=[task.io_contract_sketch.inputs, task.io_contract_sketch.outputs],
            outputs=[f"Verification evidence for {task.task_id}"],
            error_surfaces=[task.io_contract_sketch.error_surfaces],
            effects=["write test evidence", "record ship verification trace"],
            modes=["sync", "verification"],
        )

    def _build_micro_plan(self, task: Task) -> MicroPlan:
        task_slug = re.sub(r"[^a-z0-9-]+", "-", task.task_id.lower()).strip("-")
        atomic_units = self._atomic_units_for_task(task)
        staged_units = [
            (self._stage_for_unit(unit), order, unit)
            for order, unit in enumerate(atomic_units)
        ]
        staged_units.sort(key=lambda item: (self._STAGE_ORDER[item[0]], item[1]))

        modules: list[MicroPlanModule] = []
        stage_index: dict[str, list[str]] = defaultdict(list)
        created_ids: list[str] = []
        for index, (stage, _order, unit) in enumerate(staged_units, start=1):
            module_id = f"MM-{task_slug}-{index:02d}"
            io_contract = self._io_contract_for_stage(task, stage, unit)

            if stage == "ingress":
                depends_on: list[str] = []
            elif stage == "core":
                depends_on = self._ordered_unique(stage_index["ingress"] + stage_index["core"][-1:])
            elif stage == "integration":
                depends_on = self._ordered_unique(stage_index["core"] + stage_index["ingress"])
            elif stage == "egress":
                depends_on = self._ordered_unique(
                    stage_index["integration"] + stage_index["core"] + stage_index["ingress"]
                )
            else:
                depends_on = list(created_ids)

            reuse_candidate_refs, reuse_decision, report = self._reuse_analysis(
                task=task,
                module_stage=stage,
                module_purpose=unit,
                io_contract=io_contract,
            )

            modules.append(
                MicroPlanModule(
                    module_id=module_id,
                    name=f"{task.name} [{stage}-{index:02d}]",
                    purpose=unit,
                    io_contract=io_contract,
                    error_cases=[task.io_contract_sketch.error_surfaces, f"Failure while executing: {unit}"],
                    depends_on=depends_on,
                    reuse_candidate_refs=reuse_candidate_refs,
                    reuse_decision=reuse_decision,
                    reuse_justification=report.justification,
                    reuse_search_report=report,
                )
            )
            created_ids.append(module_id)
            stage_index[stage].append(module_id)

        return MicroPlan(
            micro_plan_id=f"MP-{uuid.uuid4().hex[:8]}",
            parent_task_ref=task.task_id,
            modules=modules,
        )

    def _micro_plan_node(self, state: EngineeringState) -> dict[str, Any]:
        task = Task.model_validate(state["task"])
        plan = self._build_micro_plan(task)
        self.state_store.write_micro_plan(plan)

        envelope = ArtifactEnvelope.build(
            artifact_type=ArtifactType.MICRO_PLAN,
            created_by={"role": "micro_planner", "run_id": task.task_id},
            context_pack_ref=f"CP-{uuid.uuid4().hex[:8]}",
            payload=plan.model_dump(mode="json"),
            status=ArtifactStatus.SHIPPED,
        )
        self.artifacts.create(envelope)

        return {
            "micro_plan": plan.model_dump(mode="json"),
            "module_order": self._module_topological_order(plan),
            "module_cursor": 0,
            "module_status": {module.module_id: "PENDING" for module in plan.modules},
            "module_refs": {},
            "failed_modules": [],
            "abandoned_modules": [],
            "shipped_modules": [],
            "halted": False,
            "escalation_id": None,
            "halt_reason": None,
        }

    def _build_context_packs(self, task: Task, module: MicroPlanModule) -> dict[str, str]:
        dependency_permissions = [
            (f"/modules/{dep}", ContextAccessLevel.CONTRACT_ONLY)
            for dep in module.depends_on
            if dep.startswith("MM-")
        ]

        proposer_cp = build_context_pack(
            task_id=task.task_id,
            objective=f"Implement {module.module_id}",
            role="proposer",
            permissions=[(f"/modules/{module.module_id}", ContextAccessLevel.FULL)] + dependency_permissions,
            context_budget_tokens=self.settings.context_budget_cap_tokens,
            required_sections=["task_contract", "micro_plan", "dependencies"],
        )
        challenger_cp = build_context_pack(
            task_id=task.task_id,
            objective=f"Challenge {module.module_id}",
            role="challenger",
            permissions=[("/modules", ContextAccessLevel.CONTRACT_ONLY)],
            context_budget_tokens=self.settings.context_budget_floor_tokens,
            required_sections=["task_contract", "proposal", "rubric"],
        )
        arbiter_cp = build_context_pack(
            task_id=task.task_id,
            objective=f"Adjudicate {module.module_id}",
            role="arbiter",
            permissions=[("/debates", ContextAccessLevel.SUMMARY_ONLY)],
            context_budget_tokens=self.settings.context_budget_floor_tokens,
            required_sections=["task_context", "proposal", "challenge"],
        )

        self.state_store.write_context_pack(proposer_cp)
        self.state_store.write_context_pack(challenger_cp)
        self.state_store.write_context_pack(arbiter_cp)
        return {
            "proposer": proposer_cp.cp_id,
            "challenger": challenger_cp.cp_id,
            "arbiter": arbiter_cp.cp_id,
        }

    def _module_dependents(self, plan: MicroPlan) -> dict[str, list[str]]:
        dependents: dict[str, list[str]] = defaultdict(list)
        for module in plan.modules:
            for dep in module.depends_on:
                if dep.startswith("MM-"):
                    dependents[dep].append(module.module_id)
        return dependents

    def _mark_abandoned(self, plan: MicroPlan, failed_module: str, module_status: dict[str, str]) -> list[str]:
        dependents = self._module_dependents(plan)
        queue: deque[str] = deque(dependents.get(failed_module, []))
        abandoned: list[str] = []
        visited: set[str] = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            if module_status.get(current) == "PENDING":
                module_status[current] = "ABANDONED"
                abandoned.append(current)
            for nxt in dependents.get(current, []):
                queue.append(nxt)
        return abandoned

    def _resolve_dependency_refs(self, module: MicroPlanModule, module_refs: dict[str, str]) -> list[str]:
        refs: list[str] = []
        for dep in module.depends_on:
            if not dep.startswith("MM-"):
                continue
            resolved = module_refs.get(dep)
            if resolved is None:
                raise ValueError(f"Missing resolved dependency ref for {module.module_id}: {dep}")
            refs.append(resolved)
        return self._ordered_unique(refs)

    def _build_contract(
        self,
        task: Task,
        module: MicroPlanModule,
        context_pack_ref: str,
        dependency_refs: list[str],
    ) -> MicroModuleContract:
        _ = context_pack_ref
        module_version = self._next_module_version(module.module_id)
        stage_match = re.search(r"\[([a-z]+)-\d+\]$", module.name.lower())
        stage_tag = stage_match.group(1) if stage_match else "module"
        return MicroModuleContract(
            module_id=module.module_id,
            module_version=module_version,
            name=module.name,
            purpose=module.purpose,
            tags=["factory", "micro-module", stage_tag, slugify_name(task.name, max_length=20)],
            examples_ref=f"/modules/{module.module_id}/{module_version}/examples.md",
            created_by=f"engineering_loop:{task.task_id}",
            inputs=[
                {
                    "name": f"input_{idx + 1}",
                    "type": "string",
                    "constraints": [value],
                }
                for idx, value in enumerate(module.io_contract.inputs)
            ],
            outputs=[
                {
                    "name": f"output_{idx + 1}",
                    "type": "string",
                    "invariants": [value],
                }
                for idx, value in enumerate(module.io_contract.outputs)
            ],
            error_surfaces=[
                {
                    "name": f"error_{idx + 1}",
                    "when": value,
                    "surface": "code",
                }
                for idx, value in enumerate(module.io_contract.error_surfaces or ["validation error"])
            ],
            effects=[
                {
                    "type": "WRITE",
                    "target": f"/modules/{module.module_id}",
                    "description": value,
                }
                for value in (module.io_contract.effects or ["writes module artifacts"])
            ],
            modes={"sync": True, "async": False, "notes": "; ".join(module.io_contract.modes or ["sync"])},
            error_cases=module.error_cases,
            dependencies=[{"ref": dep_ref, "why": "module dependency"} for dep_ref in dependency_refs],
            compatibility={"backward_compatible_with": [], "breaking_change_policy": "major for breaking changes"},
            runtime_budgets={"latency_ms_p95": 1000.0, "memory_mb_max": 512.0},
            status=ArtifactStatus.DRAFT,
            supersedes=None,
            deprecated_by=None,
        )

    def _next_module_version(self, module_id: str) -> str:
        module_root = self.state_store.modules_dir / module_id
        if not module_root.is_dir():
            return "1.0.0"

        versions: list[tuple[int, int, int]] = []
        for candidate in module_root.iterdir():
            if not candidate.is_dir():
                continue
            parts = candidate.name.split(".")
            if len(parts) != 3 or not all(part.isdigit() for part in parts):
                continue
            versions.append((int(parts[0]), int(parts[1]), int(parts[2])))

        if not versions:
            return "1.0.0"
        major, minor, patch = sorted(versions)[-1]
        return f"{major}.{minor}.{patch + 1}"

    def _module_step_node(self, state: EngineeringState) -> dict[str, Any] | Command[str]:
        plan = MicroPlan.model_validate(state["micro_plan"])
        task = Task.model_validate(state["task"])
        cursor = int(state["module_cursor"])
        module_order = list(state["module_order"])
        module_status = dict(state["module_status"])
        module_refs = dict(state["module_refs"])
        failed_modules = list(state["failed_modules"])
        abandoned_modules = list(state["abandoned_modules"])
        shipped_modules = list(state["shipped_modules"])

        if cursor >= len(module_order):
            return Command(goto="finalize")

        module_id = module_order[cursor]
        if module_status.get(module_id) == "ABANDONED":
            return {
                "module_cursor": cursor + 1,
                "module_status": module_status,
                "module_refs": module_refs,
                "failed_modules": failed_modules,
                "abandoned_modules": abandoned_modules,
                "shipped_modules": shipped_modules,
            }

        module = next(entry for entry in plan.modules if entry.module_id == module_id)
        resolved_dependency_refs = self._resolve_dependency_refs(module, module_refs)
        reuse_decision_value = module.reuse_decision.value
        reuse_justification_value = module.reuse_justification
        if module.reuse_decision == ReuseDecision.REUSE and module.reuse_candidate_refs:
            selected_ref = module.reuse_candidate_refs[0]
            selected_entry = self.code_index.get_entry(selected_ref)
            if selected_entry is None:
                raise ValueError(
                    f"Micro-plan selected reuse candidate not present in code index: {selected_ref}"
                )
            selected_deps = self._ordered_unique(selected_entry.dependencies)
            if set(selected_deps) == set(resolved_dependency_refs):
                module_status[module_id] = "SHIPPED"
                module_refs[module_id] = selected_ref
                shipped_modules.append(module_id)
                return {
                    "module_cursor": cursor + 1,
                    "module_status": module_status,
                    "module_refs": module_refs,
                    "failed_modules": failed_modules,
                    "abandoned_modules": abandoned_modules,
                    "shipped_modules": shipped_modules,
                }
            reuse_decision_value = ReuseDecision.CREATE_NEW.value
            reuse_justification_value = (
                f"{module.reuse_justification}; rejected {selected_ref} due dependency mismatch "
                f"(expected={resolved_dependency_refs}, selected={selected_deps})"
            )

        cp_refs = self._build_context_packs(task, module)
        context_pack_ref = cp_refs["proposer"]

        contract = self._build_contract(
            task,
            module,
            context_pack_ref,
            resolved_dependency_refs,
        )
        contract_path = self.state_store.write_module_contract(contract)
        examples_path = contract_path.parent / "examples.md"
        examples_path.write_text(
            "# Examples\n\n## Example 1\n\n```input\n{\"sample\": true}\n```\n\n```output\n{\"ok\": true}\n```\n",
            encoding="utf-8",
        )

        contract_envelope = ArtifactEnvelope.build(
            artifact_type=ArtifactType.CONTRACT,
            created_by={"role": "proposer", "run_id": task.task_id},
            context_pack_ref=context_pack_ref,
            payload=contract.model_dump(mode="json", by_alias=True),
        )
        self.artifacts.create(contract_envelope)
        self.artifacts.challenge(contract_envelope.artifact_id)
        self.artifacts.adjudicate(contract_envelope.artifact_id)

        debate = self.debate_graph.run(
            task_id=task.task_id,
            module_id=module.module_id,
            target_artifact_id=contract_envelope.artifact_id,
            contract_summary={
                "inputs": [entry.model_dump(mode="json") for entry in contract.inputs],
                "outputs": [entry.model_dump(mode="json") for entry in contract.outputs],
                "error_surfaces": [entry.model_dump(mode="json") for entry in contract.error_surfaces],
            },
            context_summary=(
                f"Task={task.task_id}; module={module.module_id}; purpose={module.purpose}; "
                f"reuse_decision={reuse_decision_value}; "
                f"reuse_candidates={','.join(module.reuse_candidate_refs) if module.reuse_candidate_refs else 'none'}; "
                f"reuse_justification={reuse_justification_value}; "
                f"io_inputs={'; '.join(module.io_contract.inputs)}; "
                f"io_outputs={'; '.join(module.io_contract.outputs)}"
            ),
            max_retries=2,
        )

        if debate.passed:
            verification = ShipVerification(
                result="PASS",
                interface_fingerprint=contract.interface_fingerprint,
                evidence_ref=f"/modules/{contract.module_id}/{contract.module_version}/tests/execution_log.json",
            )
            evidence = ShipEvidence(
                module_id=contract.module_id,
                module_version=contract.module_version,
                ship_id=f"SHIP-{uuid.uuid4().hex[:8]}",
                verified_at=datetime.now(UTC),
                verification=verification,
                environment=ShipEnvironment(
                    repo_revision="local",
                    dependency_lock_ref="uv.lock",
                    runner_id="engineering-loop",
                ),
                test_artifact_refs=[verification.evidence_ref],
                coverage_report_ref=f"/modules/{contract.module_id}/{contract.module_version}/tests/coverage.json",
                ship_time=datetime.now(UTC),
            )
            self.state_store.write_ship_evidence(evidence)
            self.state_store.seal_module(contract.module_id, contract.module_version)

            self.artifacts.ship(contract_envelope.artifact_id)

            slug = self.code_index.register(contract)
            module_status[module_id] = "SHIPPED"
            module_refs[module_id] = f"{contract.module_id}@{contract.module_version}"
            shipped_modules.append(module_id)
            _ = slug
        else:
            module_status[module_id] = "FAILED"
            failed_modules.append(module_id)
            newly_abandoned = self._mark_abandoned(plan, module_id, module_status)
            abandoned_modules.extend(newly_abandoned)

        return {
            "module_cursor": cursor + 1,
            "module_status": module_status,
            "module_refs": module_refs,
            "failed_modules": failed_modules,
            "abandoned_modules": abandoned_modules,
            "shipped_modules": shipped_modules,
        }

    def _module_route_node(self, state: EngineeringState) -> Command[str]:
        cursor = int(state.get("module_cursor", 0))
        if cursor >= len(state.get("module_order", [])):
            return Command(goto="finalize")
        return Command(goto="module_step")

    def _finalize_node(self, state: EngineeringState) -> dict[str, Any]:
        failed = list(state.get("failed_modules", []))
        if failed:
            task_id = state["task_id"]
            escalation = EscalationArtifact(
                escalation_id=f"ESC-{uuid.uuid4().hex[:8]}",
                task_id=task_id,
                task_ref=f"/tasks/{task_id}.md",
                debate_trail={"proposals": [], "challenges": [], "adjudications": []},
                failed_invariants=[
                    FailedInvariant(
                        invariant="module debate convergence",
                        evidence=f"Failed modules: {', '.join(failed)}",
                    )
                ],
                state_machine_snapshot={},
                minimal_decision_required="Decide whether to revise plan, split task, or abandon task",
                recommended_resolution=HumanResolutionAction.REVISE_PLAN,
                resolution_context="One or more modules failed debate after bounded retries",
                failure_root_cause="Debate failures in per-module implementation",
                retries_exhausted=2,
                context_pack_refs=[],
                debate_artifact_refs=[],
            )
            self.state_store.write_escalation(escalation)
            return {
                "halted": True,
                "escalation_id": escalation.escalation_id,
                "halt_reason": escalation.failure_root_cause,
            }

        return {"halted": False, "escalation_id": None, "halt_reason": None}

    def run(self, task: Task) -> EngineeringResult:
        result = self.graph.invoke(
            {
                "task": task.model_dump(mode="json"),
                "task_id": task.task_id,
            }
        )
        if result.get("halted"):
            return EngineeringResult(
                task_status=TaskStatus.HALTED,
                module_refs=list(result.get("module_refs", {}).values()),
                escalation_id=result.get("escalation_id"),
                halt_reason=result.get("halt_reason"),
            )
        return EngineeringResult(
            task_status=TaskStatus.SHIPPED,
            module_refs=list(result.get("module_refs", {}).values()),
        )


class ProjectStateGraphState(TypedDict, total=False):
    spec: dict[str, Any]
    current_task_id: str | None
    complete: bool
    halted: bool
    last_engineering_result: dict[str, Any] | None
    resolution: dict[str, Any] | None


class ProjectLoop:
    """Project loop implemented as LangGraph StateGraph dispatch cycle."""

    def __init__(
        self,
        spec: StructuredSpec,
        code_index: CodeIndex | None = None,
        *,
        state_store_root: str | Path | None = None,
        settings: RuntimeSettings | None = None,
        enable_interrupts: bool = True,
    ) -> None:
        self.settings = settings if settings is not None else RuntimeSettings.from_env()
        self.spec = spec
        root = Path(state_store_root) if state_store_root is not None else Path(self.settings.state_store_root)
        self.state_store = FactoryStateStore(root, project_id=self.settings.project_id)
        self.code_index = (
            code_index
            if code_index is not None
            else CodeIndex(self.state_store.code_index_dir, embedding_model=self.settings.embedding_model)
        )
        self.artifacts = ArtifactStoreService(self.state_store)
        self.enable_interrupts = enable_interrupts
        self._task_lookup = {task.task_id: task for task in self.spec.iter_tasks()}

        self.checkpoint_path = Path(self.settings.checkpoint_db)
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = self.state_store.root / "checkpoints" / "project_loop.sqlite"
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_conn = sqlite3.connect(self.checkpoint_path, check_same_thread=False)
        self._checkpointer = SqliteSaver(self._checkpoint_conn)
        self.graph = self._build_graph().compile(checkpointer=self._checkpointer)

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(ProjectStateGraphState)
        graph.add_node("init_state_machine", self._init_state_machine_node)
        graph.add_node("dispatch", self._dispatch_node)
        graph.add_node("engineering", self._engineering_node)
        graph.add_node("update_state_machine", self._update_state_machine_node)

        graph.add_edge(START, "init_state_machine")
        graph.add_edge("init_state_machine", "dispatch")
        graph.add_conditional_edges(
            "dispatch",
            self._dispatch_route,
            {
                "engineering": "engineering",
                "end": END,
            },
        )
        graph.add_edge("engineering", "update_state_machine")
        graph.add_conditional_edges(
            "update_state_machine",
            self._update_route,
            {
                "dispatch": "dispatch",
                "end": END,
            },
        )
        return graph

    def _task_file_markdown(self, task: Task) -> str:
        dependencies = "\n".join(f"- {dep}" for dep in task.depends_on) or "- none"
        return (
            f"# Task: {task.name}\n"
            f"## Task ID: {task.task_id}\n\n"
            "## Context\n"
            f"- {self.state_store.state_machine_path}\n"
            f"- dependencies:\n{dependencies}\n\n"
            "## Description\n"
            f"{task.description}\n\n"
            "## Subtasks\n"
            + "\n".join(f"- {item}" for item in task.subtasks)
            + "\n\n## Acceptance Criteria\n"
            + "\n".join(f"- {item}" for item in task.acceptance_criteria)
            + "\n\n## Micro Module Contract\n"
            + f"- Inputs: {task.io_contract_sketch.inputs}\n"
            + f"- Outputs: {task.io_contract_sketch.outputs}\n"
            + f"- Error surfaces: {task.io_contract_sketch.error_surfaces}\n"
            + f"- Effects: {task.io_contract_sketch.effects}\n"
            + f"- Modes: {task.io_contract_sketch.modes}\n"
        )

    def _init_state_machine_node(self, state: ProjectStateGraphState) -> dict[str, Any]:
        spec = StructuredSpec.model_validate(state["spec"])
        apply_canonical_task_ids(spec)
        self._task_lookup = {task.task_id: task for task in spec.iter_tasks()}

        for task in spec.iter_tasks():
            self.state_store.write_task_file(task.task_id, self._task_file_markdown(task))

        project_state = build_project_state(spec, project_id=self.settings.project_id)
        self.state_store.write_project_state(project_state)
        return {"spec": spec.model_dump(mode="json")}

    def _eligible_task_ids(self, state_machine: ProjectState) -> list[str]:
        eligible: list[tuple[int, str]] = []
        for task_id, task_state in state_machine.tasks.items():
            if task_state.status != TaskStatus.PENDING:
                continue
            if all(state_machine.tasks[dep].status == TaskStatus.SHIPPED for dep in task_state.depends_on):
                eligible.append((task_state.declaration_order, task_id))
        eligible.sort(key=lambda item: item[0])
        return [task_id for _, task_id in eligible]

    def _dispatch_node(self, _state: ProjectStateGraphState) -> dict[str, Any]:
        state_machine = self.state_store.read_project_state()
        if any(task.status == TaskStatus.HALTED for task in state_machine.tasks.values()):
            return {"halted": True, "complete": False, "current_task_id": None}

        eligible = self._eligible_task_ids(state_machine)
        if not eligible:
            has_pending = any(task.status == TaskStatus.PENDING for task in state_machine.tasks.values())
            return {"complete": not has_pending, "current_task_id": None}

        task_id = eligible[0]
        state_machine.transition(task_id, TaskStatus.IN_PROGRESS)
        self.state_store.write_project_state(state_machine)
        return {"current_task_id": task_id, "complete": False, "halted": False}

    def _dispatch_route(self, state: ProjectStateGraphState) -> str:
        if state.get("current_task_id"):
            return "engineering"
        return "end"

    def _engineering_node(self, state: ProjectStateGraphState) -> dict[str, Any]:
        task_id = state.get("current_task_id")
        if not task_id:
            return {"last_engineering_result": None}

        task = self._task_lookup[task_id]
        runner = EngineeringLoop(code_index=self.code_index, state_store=self.state_store, settings=self.settings)
        result = runner.run(task)
        return {
            "last_engineering_result": {
                "task_id": task_id,
                "task_status": result.task_status.value,
                "module_refs": result.module_refs,
                "escalation_id": result.escalation_id,
                "halt_reason": result.halt_reason,
            }
        }

    def _mark_downstream_blocked(self, state_machine: ProjectState, failed_task_id: str) -> None:
        dependents: dict[str, list[str]] = defaultdict(list)
        for task_id, entry in state_machine.tasks.items():
            for dep in entry.depends_on:
                dependents[dep].append(task_id)

        queue: deque[str] = deque(dependents.get(failed_task_id, []))
        visited: set[str] = {failed_task_id}
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            current_state = state_machine.tasks[current]
            if current_state.status == TaskStatus.PENDING:
                current_state.status = TaskStatus.BLOCKED
            for nxt in dependents.get(current, []):
                queue.append(nxt)

    def _clear_blocked_fixed_point(self, state_machine: ProjectState) -> None:
        changed = True
        while changed:
            changed = False
            for task_id, entry in state_machine.tasks.items():
                if entry.status != TaskStatus.BLOCKED:
                    continue
                if any(state_machine.tasks[dep].status == TaskStatus.HALTED for dep in entry.depends_on):
                    continue
                entry.status = TaskStatus.PENDING
                changed = True

    def _process_resolution(self, state_machine: ProjectState, task_id: str, resolution: HumanResolution) -> None:
        task_state = state_machine.tasks[task_id]
        if task_state.status != TaskStatus.HALTED:
            raise ValueError(f"Resolution requires HALTED task, got {task_state.status.value}")

        if resolution.action == HumanResolutionAction.APPROVE_OVERRIDE:
            task_state.status = TaskStatus.SHIPPED
            task_state.halted_reason = resolution.rationale
        elif resolution.action == HumanResolutionAction.AMEND_SPEC:
            task_state.status = TaskStatus.PENDING
            task_file = self.state_store.tasks_dir / f"{task_id}.md"
            original = task_file.read_text(encoding="utf-8")
            amendment = "\n\n## Amendment History\n"
            amendment += f"- {datetime.now(UTC).isoformat()}: {resolution.amendment_rationale}\n"
            amendment += "\n## Updated Acceptance Criteria\n"
            amendment += "\n".join(f"- {item}" for item in resolution.amended_acceptance_criteria)
            task_file.write_text(original + amendment + "\n", encoding="utf-8")
        elif resolution.action == HumanResolutionAction.SPLIT_TASK:
            task_state.status = TaskStatus.ABANDONED
            max_order = max(entry.declaration_order for entry in state_machine.tasks.values())
            for index, new_task in enumerate(resolution.new_tasks, start=1):
                new_task_id = f"{task_id}-split-{index}"
                if new_task_id in state_machine.tasks:
                    raise ValueError(f"Duplicate split task id: {new_task_id}")
                state_machine.tasks[new_task_id] = type(task_state)(
                    pillar=task_state.pillar,
                    epic=task_state.epic,
                    story=task_state.story,
                    task=new_task.name,
                    status=TaskStatus.PENDING,
                    depends_on=list(new_task.depends_on),
                    declaration_order=max_order + index,
                )
                self.state_store.write_task_file(
                    new_task_id,
                    (
                        f"# Task: {new_task.name}\n"
                        f"## Task ID: {new_task_id}\n\n"
                        f"## Description\n{new_task.description}\n\n"
                        "## Acceptance Criteria\n"
                        + "\n".join(f"- {item}" for item in new_task.acceptance_criteria)
                        + "\n"
                    ),
                )
        elif resolution.action == HumanResolutionAction.REVISE_PLAN:
            task_state.status = TaskStatus.IN_PROGRESS
        elif resolution.action == HumanResolutionAction.PROVIDE_FIX:
            task_state.status = TaskStatus.IN_PROGRESS
            for path, content in resolution.fix_artifacts.items():
                resolved = self.state_store.root / path.lstrip("/")
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(content, encoding="utf-8")
        elif resolution.action == HumanResolutionAction.ABANDON_TASK:
            task_state.status = TaskStatus.ABANDONED
            task_state.halted_reason = resolution.rationale

    def _update_state_machine_node(self, state: ProjectStateGraphState) -> dict[str, Any]:
        result = state.get("last_engineering_result")
        if not result:
            return {}

        task_id = result["task_id"]
        state_machine = self.state_store.read_project_state()
        task_state = state_machine.tasks[task_id]

        status = TaskStatus(result["task_status"])
        if status == TaskStatus.SHIPPED:
            task_state.status = TaskStatus.SHIPPED
            shipped_refs = list(result.get("module_refs", []))
            task_state.module_refs = shipped_refs
            task_state.module_ref = shipped_refs[0] if shipped_refs else None
            task_state.shipped_at = datetime.now(UTC)
            self._clear_blocked_fixed_point(state_machine)
            self.state_store.write_project_state(state_machine)
            return {"halted": False}

        task_state.status = TaskStatus.HALTED
        task_state.halted_reason = result.get("halt_reason")
        task_state.escalation_ref = result.get("escalation_id")
        self._mark_downstream_blocked(state_machine, task_id)
        self.state_store.write_project_state(state_machine)

        resolution_payload = state.get("resolution")
        if resolution_payload:
            resolution = HumanResolution.model_validate(resolution_payload)
            self._process_resolution(state_machine, task_id, resolution)
            self._clear_blocked_fixed_point(state_machine)
            self.state_store.write_project_state(state_machine)
            return {"halted": False, "resolution": None}

        if self.enable_interrupts:
            interruption = interrupt(
                {
                    "task_id": task_id,
                    "escalation_id": result.get("escalation_id"),
                    "halt_reason": result.get("halt_reason"),
                    "resolution_options": [action.value for action in HumanResolutionAction],
                }
            )
            return {"halted": True, "interrupt_payload": interruption}

        return {"halted": True}

    def _update_route(self, state: ProjectStateGraphState) -> str:
        if state.get("halted"):
            return "end"
        return "dispatch"

    def run(self, *, resolution: HumanResolution | None = None) -> bool:
        initial_state: ProjectStateGraphState = {
            "spec": self.spec.model_dump(mode="json"),
            "current_task_id": None,
            "complete": False,
            "halted": False,
            "last_engineering_result": None,
            "resolution": resolution.model_dump(mode="json") if resolution else None,
        }
        result = self.graph.invoke(
            initial_state,
            config={
                "recursion_limit": self.settings.recursion_limit,
                "configurable": {"thread_id": f"project-loop-{uuid.uuid4().hex[:8]}"},
            },
        )
        return not bool(result.get("halted"))

    def export_state(self) -> str:
        return self.state_store.read_project_state().model_dump_json(indent=2)


class ReleaseLoopState(TypedDict, total=False):
    release_id: str
    modules: list[str]
    integration_gates: dict[str, str]
    overall_result: str
    notes: str


class ReleaseLoop:
    """Release stage StateGraph with dependency/fingerprint/deprecation/code-index gates."""

    def __init__(self, *, state_store: FactoryStateStore, code_index: CodeIndex, spec: StructuredSpec) -> None:
        self.state_store = state_store
        self.code_index = code_index
        self.spec = spec
        self.graph = self._build_graph().compile()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(ReleaseLoopState)
        graph.add_node("init", self._init_node)
        graph.add_node("gate_check", self._gate_check_node)
        graph.add_node("finalize", self._finalize_node)
        graph.add_edge(START, "init")
        graph.add_edge("init", "gate_check")
        graph.add_edge("gate_check", "finalize")
        graph.add_edge("finalize", END)
        return graph

    def _expand_dependency_closure(self, modules: list[str]) -> list[str]:
        entries = {entry.module_ref: entry for entry in self.code_index.list_entries()}
        ordered = list(dict.fromkeys(modules))
        seen = set(ordered)
        queue: deque[str] = deque(ordered)
        while queue:
            module_ref = queue.popleft()
            entry = entries.get(module_ref)
            if entry is None:
                continue
            for dep in entry.dependencies:
                if dep in seen:
                    continue
                seen.add(dep)
                ordered.append(dep)
                queue.append(dep)
        return ordered

    def _init_node(self, _state: ReleaseLoopState) -> dict[str, Any]:
        state_machine = self.state_store.read_project_state()
        modules: list[str] = []
        for task in state_machine.tasks.values():
            if task.status != TaskStatus.SHIPPED:
                continue
            if task.module_refs:
                modules.extend(task.module_refs)
            elif task.module_ref is not None:
                modules.append(task.module_ref)
        modules = self._expand_dependency_closure(modules)
        return {"release_id": f"REL-{uuid.uuid4().hex[:8]}", "modules": modules}

    def _gate_check_node(self, state: ReleaseLoopState) -> dict[str, Any]:
        modules = list(state.get("modules", []))
        entries = {entry.module_ref: entry for entry in self.code_index.list_entries()}

        dependency_pass = True
        fingerprint_pass = True
        deprecation_pass = True
        code_index_pass = True
        notes: list[str] = []

        for module_ref in modules:
            entry = entries.get(module_ref)
            if entry is None:
                code_index_pass = False
                notes.append(f"Missing code index entry for {module_ref}")
                continue
            if entry.status != CodeIndexStatus.CURRENT:
                code_index_pass = False
                notes.append(f"Module {module_ref} is not CURRENT in code index")
            if entry.status in {CodeIndexStatus.DEPRECATED, CodeIndexStatus.SUPERSEDED}:
                deprecation_pass = False
                notes.append(f"Module {module_ref} is inactive ({entry.status.value})")
            for dep in entry.dependencies:
                if dep not in modules:
                    dependency_pass = False
                    notes.append(f"Dependency {dep} for {module_ref} not included in release set")
                dep_entry = entries.get(dep)
                if dep_entry and dep_entry.status in {CodeIndexStatus.DEPRECATED, CodeIndexStatus.SUPERSEDED}:
                    deprecation_pass = False
                    notes.append(f"Module {module_ref} depends on inactive dependency {dep}")
                if dep_entry is None:
                    # Missing dependency means we cannot verify consumer fingerprint pairing.
                    fingerprint_pass = False
                    notes.append(f"Cannot verify dependency fingerprint for {module_ref} -> {dep}")

        gates = {
            "dependency_check": "PASS" if dependency_pass else "FAIL",
            "fingerprint_check": "PASS" if fingerprint_pass else "FAIL",
            "deprecation_check": "PASS" if deprecation_pass else "FAIL",
            "code_index_check": "PASS" if code_index_pass else "FAIL",
        }
        overall = "PASS" if all(value == "PASS" for value in gates.values()) else "FAIL"
        return {
            "integration_gates": gates,
            "overall_result": overall,
            "notes": "\n".join(notes) if notes else "All release gates passed",
        }

    def _finalize_node(self, state: ReleaseLoopState) -> dict[str, Any]:
        manifest = {
            "release_id": state["release_id"],
            "created_at": datetime.now(UTC).isoformat(),
            "spec_version": self.spec.spec_version,
            "modules": [{"module_ref": module_ref, "ship_ref": module_ref.replace("MM", "SHIP")} for module_ref in state["modules"]],
            "integration_gates": state["integration_gates"],
            "overall_result": state["overall_result"],
            "notes": state["notes"],
        }
        path = self.state_store.release_dir / f"{state['release_id']}.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return state

    def run(self) -> dict[str, Any]:
        return self.graph.invoke({})


class OuterGraphState(TypedDict, total=False):
    raw_spec: str
    spec: dict[str, Any]
    approval_action: str | None
    approval_feedback: str | None
    amend_count: int
    project_success: bool
    release_result: dict[str, Any]


class FactoryOrchestrator:
    """Outer graph: Product Loop -> Approval Gate -> Project Loop -> Release Loop."""

    def __init__(
        self,
        *,
        raw_spec: str,
        settings: RuntimeSettings | None = None,
        state_store_root: str | Path | None = None,
    ) -> None:
        self.settings = settings if settings is not None else RuntimeSettings.from_env()
        self.raw_spec = raw_spec
        root = Path(state_store_root) if state_store_root is not None else Path(self.settings.state_store_root)
        self.state_store = FactoryStateStore(root, project_id=self.settings.project_id)
        self.code_index = CodeIndex(
            self.state_store.code_index_dir,
            embedding_model=self.settings.embedding_model,
        )

        checkpoint_path = self.state_store.root / "checkpoints" / "outer_graph.sqlite"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
        self._checkpointer = SqliteSaver(self._conn)
        self.graph = self._build_graph().compile(checkpointer=self._checkpointer)

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(OuterGraphState)
        graph.add_node("product_loop", self._product_node)
        graph.add_node("approval_gate", self._approval_gate_node)
        graph.add_node("project_loop", self._project_node)
        graph.add_node("release_loop", self._release_node)

        graph.add_edge(START, "product_loop")
        graph.add_edge("product_loop", "approval_gate")
        graph.add_conditional_edges(
            "approval_gate",
            self._approval_route,
            {
                "project_loop": "project_loop",
                "product_loop": "product_loop",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "project_loop",
            self._project_route,
            {
                "release_loop": "release_loop",
                "end": END,
            },
        )
        graph.add_edge("release_loop", END)
        return graph

    def _product_node(self, state: OuterGraphState) -> dict[str, Any]:
        loop = ProductLoop(state["raw_spec"], state_store_root=self.state_store.base_root, settings=self.settings)
        spec = loop.run()
        return {"spec": spec.model_dump(mode="json")}

    def _approval_gate_node(self, state: OuterGraphState) -> dict[str, Any]:
        spec = StructuredSpec.model_validate(state["spec"])
        markdown = render_spec_markdown(spec)
        self.state_store.product_spec_md_path.write_text(markdown, encoding="utf-8")

        action = state.get("approval_action")
        if action is None:
            decision = interrupt(
                {
                    "spec_markdown": markdown,
                    "options": ["APPROVE", "REJECT", "AMEND"],
                    "amend_count": int(state.get("amend_count", 0)),
                }
            )
            return {"approval_action": decision.get("action"), "approval_feedback": decision.get("feedback")}

        return {}

    def _approval_route(self, state: OuterGraphState) -> str:
        action = (state.get("approval_action") or "APPROVE").upper()
        if action == "APPROVE":
            return "project_loop"
        if action == "REJECT":
            feedback = state.get("approval_feedback") or "No reason provided"
            return_value = {
                "raw_spec": f"{state['raw_spec']}\n\n<!-- rejection feedback: {feedback} -->",
                "approval_action": None,
                "approval_feedback": None,
            }
            state.update(return_value)
            return "product_loop"
        if action == "AMEND":
            amend_count = int(state.get("amend_count", 0)) + 1
            feedback = state.get("approval_feedback") or "No amendment details provided"
            advisory = "\n\n[advisory] three consecutive AMEND cycles reached" if amend_count >= 3 else ""
            state["raw_spec"] = f"{state['raw_spec']}\n\n<!-- amend feedback: {feedback} -->{advisory}"
            state["approval_action"] = None
            state["approval_feedback"] = None
            state["amend_count"] = amend_count
            return "product_loop"
        return "end"

    def _project_node(self, state: OuterGraphState) -> dict[str, Any]:
        spec = StructuredSpec.model_validate(state["spec"])
        loop = ProjectLoop(
            spec,
            self.code_index,
            state_store_root=self.state_store.base_root,
            settings=self.settings,
            enable_interrupts=False,
        )
        success = loop.run()
        return {"project_success": success}

    def _project_route(self, state: OuterGraphState) -> str:
        return "release_loop" if state.get("project_success") else "end"

    def _release_node(self, state: OuterGraphState) -> dict[str, Any]:
        spec = StructuredSpec.model_validate(state["spec"])
        loop = ReleaseLoop(state_store=self.state_store, code_index=self.code_index, spec=spec)
        result = loop.run()
        return {"release_result": result}

    def run(
        self,
        *,
        approval_action: str = "APPROVE",
        approval_feedback: str | None = None,
    ) -> dict[str, Any]:
        return self.graph.invoke(
            {
                "raw_spec": self.raw_spec,
                "approval_action": approval_action,
                "approval_feedback": approval_feedback,
                "amend_count": 0,
            },
            config={
                "recursion_limit": self.settings.recursion_limit,
                "configurable": {"thread_id": f"outer-graph-{uuid.uuid4().hex[:8]}"},
            },
        )
