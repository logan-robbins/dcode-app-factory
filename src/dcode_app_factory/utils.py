from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path

from .canonical import to_canonical_json
from .models import (
    AgentConfig,
    ContextAccessLevel,
    ContextPack,
    ContextPermission,
    IOContractSketch,
    Pillar,
    ProductSpec,
    Severity,
    Story,
    StructuredSpec,
    Task,
    ValidationIssue,
    ValidationReport,
)


TESTABLE_VERB_RE = re.compile(
    r"\b(returns|displays|raises|writes|emits|rejects|validates|produces|creates|records|updates)\b",
    re.IGNORECASE,
)


def get_agent_config_dir(stage: str) -> Path:
    """Return package-relative path to agent configs for a stage."""
    return Path(__file__).resolve().parent / "agent_configs" / stage


def load_agent_config(path: Path) -> AgentConfig:
    return AgentConfig.model_validate_json(path.read_text(encoding="utf-8"))


def slugify_name(name: str, *, max_length: int = 24) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:max_length].rstrip("-")


def dedupe_slug(base_slug: str, used: set[str], *, max_length: int = 24) -> str:
    if base_slug not in used:
        used.add(base_slug)
        return base_slug

    suffix_ord = ord("a")
    while True:
        suffix = f"-{chr(suffix_ord)}"
        candidate = f"{base_slug[: max_length - len(suffix)]}{suffix}".rstrip("-")
        if candidate not in used:
            used.add(candidate)
            return candidate
        suffix_ord += 1
        if suffix_ord > ord("z"):
            raise ValueError(f"unable to disambiguate slug for base '{base_slug}'")


def validate_task_dependency_dag(spec: ProductSpec | StructuredSpec) -> None:
    tasks = {task.task_id: task for task in spec.iter_tasks()}
    indegree = {task_id: 0 for task_id in tasks}
    edges: dict[str, list[str]] = defaultdict(list)

    for task in tasks.values():
        for dep in task.depends_on:
            if dep not in tasks:
                raise ValueError(f"Task {task.task_id} depends on unknown task {dep}")
            indegree[task.task_id] += 1
            edges[dep].append(task.task_id)

    queue = deque(sorted(task_id for task_id, degree in indegree.items() if degree == 0))
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for nxt in sorted(edges[current]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if visited != len(tasks):
        raise ValueError("Task dependency graph contains a cycle")


def validate_spec(spec: ProductSpec | StructuredSpec) -> ValidationReport:
    issues: list[ValidationIssue] = []

    if not spec.pillars:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                path="pillars",
                field="pillars",
                message="Spec must include at least one pillar",
            )
        )

    task_ids: set[str] = set()
    for p_idx, pillar in enumerate(spec.pillars):
        p_path = f"pillars[{p_idx}]"
        if not pillar.epics:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    path=p_path,
                    field="epics",
                    message="Every pillar must include at least one epic",
                )
            )

        for e_idx, epic in enumerate(pillar.epics):
            e_path = f"{p_path}.epics[{e_idx}]"
            if not epic.stories:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        path=e_path,
                        field="stories",
                        message="Every epic must include at least one story",
                    )
                )
            if not epic.success_criteria:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        path=e_path,
                        field="success_criteria",
                        message="Every epic must include at least one success criterion",
                    )
                )

            for s_idx, story in enumerate(epic.stories):
                s_path = f"{e_path}.stories[{s_idx}]"
                if not story.tasks:
                    issues.append(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            path=s_path,
                            field="tasks",
                            message="Every story must include at least one task",
                        )
                    )
                for t_idx, task in enumerate(story.tasks):
                    t_path = f"{s_path}.tasks[{t_idx}]"
                    if task.task_id in task_ids:
                        issues.append(
                            ValidationIssue(
                                severity=Severity.ERROR,
                                path=t_path,
                                field="task_id",
                                message=f"Duplicate task_id: {task.task_id}",
                            )
                        )
                    task_ids.add(task.task_id)

                    if len(task.subtasks) < 2:
                        issues.append(
                            ValidationIssue(
                                severity=Severity.ERROR,
                                path=t_path,
                                field="subtasks",
                                message="Task requires at least 2 subtasks",
                            )
                        )
                    if len(task.acceptance_criteria) < 2:
                        issues.append(
                            ValidationIssue(
                                severity=Severity.ERROR,
                                path=t_path,
                                field="acceptance_criteria",
                                message="Task requires at least 2 acceptance criteria",
                            )
                        )

                    io_values = task.io_contract_sketch.model_dump()
                    for io_key, io_value in io_values.items():
                        if not io_value.strip() or io_value.strip().lower() in {"tbd", "todo", "n/a", "na"}:
                            issues.append(
                                ValidationIssue(
                                    severity=Severity.ERROR,
                                    path=t_path,
                                    field=f"io_contract_sketch.{io_key}",
                                    message="I/O sketch dimensions must be non-empty and non-placeholder",
                                )
                            )

                    if len(task.description.strip()) < 20:
                        issues.append(
                            ValidationIssue(
                                severity=Severity.WARNING,
                                path=t_path,
                                field="description",
                                message="Description should be at least 20 characters",
                            )
                        )

                    for criterion in task.acceptance_criteria:
                        if not TESTABLE_VERB_RE.search(criterion):
                            issues.append(
                                ValidationIssue(
                                    severity=Severity.WARNING,
                                    path=t_path,
                                    field="acceptance_criteria",
                                    message=f"Acceptance criterion may not be testable: {criterion}",
                                )
                            )

                    distinct = {value.strip().lower() for value in task.subtasks}
                    if len(distinct) != len(task.subtasks):
                        issues.append(
                            ValidationIssue(
                                severity=Severity.WARNING,
                                path=t_path,
                                field="subtasks",
                                message="Subtasks appear duplicated",
                            )
                        )

                    if "error" not in task.io_contract_sketch.error_surfaces.lower():
                        issues.append(
                            ValidationIssue(
                                severity=Severity.WARNING,
                                path=t_path,
                                field="io_contract_sketch.error_surfaces",
                                message="Error surfaces should contain at least one specific error condition",
                            )
                        )

    by_task = {task.task_id: task for task in spec.iter_tasks()}
    for task in spec.iter_tasks():
        for dep in task.depends_on:
            if dep not in by_task:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        path=task.task_id,
                        field="depends_on",
                        message=f"Dependency {dep} does not exist",
                    )
                )

    try:
        validate_task_dependency_dag(spec)
    except ValueError as exc:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                path="tasks",
                field="depends_on",
                message=str(exc),
            )
        )

    report = ValidationReport()
    report.errors = [issue for issue in issues if issue.severity == Severity.ERROR]
    report.warnings = [issue for issue in issues if issue.severity == Severity.WARNING]
    return report


def emit_structured_spec(spec: ProductSpec | StructuredSpec, path: Path) -> Path:
    report = validate_spec(spec)
    if report.errors:
        deficiency = "; ".join(f"{entry.path}:{entry.field} {entry.message}" for entry in report.errors)
        raise ValueError(f"Spec emission blocked by validation errors: {deficiency}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    return path


def render_spec_markdown(spec: ProductSpec | StructuredSpec) -> str:
    lines: list[str] = [f"# {spec.title}", "", spec.description, ""]
    for pillar in spec.pillars:
        lines.extend([f"## {pillar.name}", pillar.description, "", f"Rationale: {pillar.rationale}", ""])
        for epic in pillar.epics:
            lines.extend([f"### {epic.name}", epic.description, ""])
            lines.append("Success Criteria:")
            for criterion in epic.success_criteria:
                lines.append(f"- {criterion}")
            lines.append("")
            for story in epic.stories:
                lines.extend([f"#### {story.name}", story.description, "", f"Behavior: {story.user_facing_behavior}", ""])
                for task in story.tasks:
                    lines.extend([f"##### {task.task_id} {task.name}", task.description, ""])
                    lines.append("Subtasks:")
                    for subtask in task.subtasks:
                        lines.append(f"- {subtask}")
                    lines.append("")
                    lines.append("Acceptance Criteria:")
                    for criterion in task.acceptance_criteria:
                        lines.append(f"- {criterion}")
                    lines.append("")
                    lines.append("I/O Contract Sketch:")
                    lines.append(f"- Inputs: {task.io_contract_sketch.inputs}")
                    lines.append(f"- Outputs: {task.io_contract_sketch.outputs}")
                    lines.append(f"- Error surfaces: {task.io_contract_sketch.error_surfaces}")
                    lines.append(f"- Effects: {task.io_contract_sketch.effects}")
                    lines.append(f"- Modes: {task.io_contract_sketch.modes}")
                    lines.append("")
    return "\n".join(lines).strip() + "\n"


def _extract_sections(raw_spec: str) -> list[str]:
    sections: list[str] = []
    for line in raw_spec.splitlines():
        value = line.strip()
        if value.startswith("##"):
            sections.append(value.lstrip("# "))
    return sections


def parse_raw_spec_to_product_spec(raw_spec: str, *, spec_id: str = "SPEC-001") -> StructuredSpec:
    sections = _extract_sections(raw_spec)
    if not sections:
        sections = ["Factory Core", "Project Orchestration", "Engineering Debate"]

    tasks: list[Task] = []
    for idx, section in enumerate(sections, start=1):
        task_id = f"TSK-{idx:03d}"
        tasks.append(
            Task(
                task_id=task_id,
                name=f"Implement {section}",
                description=f"Implement the capability described in '{section}' with explicit contracts and test evidence.",
                subtasks=[
                    f"Define explicit input/output/error contract boundaries for {section}",
                    f"Implement deterministic core behavior for {section}",
                    f"Integrate dependency adapters and side-effect handling for {section}",
                    f"Verify acceptance criteria and ship evidence for {section}",
                ],
                acceptance_criteria=[
                    "returns deterministic outputs for identical inputs",
                    "validates and rejects malformed requests",
                    "records verification evidence for shipped behavior",
                ],
                io_contract_sketch=IOContractSketch(
                    inputs=f"Inputs required for section '{section}' including constraints and validation rules.",
                    outputs=f"Outputs produced by section '{section}' with clear invariants.",
                    error_surfaces="Validation error conditions and failure surfaces with explicit handling.",
                    effects="Writes artifacts and updates state snapshots in state store.",
                    modes="sync with explicit retry boundaries and deterministic execution.",
                ),
                depends_on=[f"TSK-{idx - 1:03d}"] if idx > 1 else [],
            )
        )

    now = datetime.now(UTC)
    spec = StructuredSpec(
        spec_id=spec_id,
        spec_version="1.0.0",
        title="AI Software Product Factory",
        description="Contract-first factory specification generated from markdown source.",
        created_at=now,
        updated_at=now,
        pillars=[
            Pillar(
                pillar_id="PIL-001",
                name="Reliable Factory Core",
                description="Deterministic and auditable software factory execution.",
                rationale="Reliability and traceability are required for autonomous delivery.",
                epics=[
                    {
                        "epic_id": "EPC-001",
                        "name": "End-to-end Orchestration",
                        "description": "Implements product, project, engineering, and release flows.",
                        "success_criteria": [
                            "dispatches tasks deterministically",
                        ],
                        "stories": [
                            Story(
                                story_id="STR-001",
                                name="Deliver deterministic execution",
                                description="Ensure execution remains deterministic and auditable.",
                                user_facing_behavior="Operators can run and inspect a complete auditable pipeline.",
                                tasks=tasks,
                            )
                        ],
                    }
                ],
            )
        ],
    )

    report = validate_spec(spec)
    if report.errors:
        deficiency = "; ".join(f"{entry.path}:{entry.message}" for entry in report.errors)
        raise ValueError(f"Unable to construct valid ProductSpec from raw spec: {deficiency}")
    return spec


def build_context_pack(
    task_id: str,
    objective: str,
    role: str,
    permissions: list[tuple[str, ContextAccessLevel]],
    context_budget_tokens: int,
    required_sections: list[str],
) -> ContextPack:
    cp_id = f"CP-{to_canonical_json({'task_id': task_id, 'role': role, 'ts': datetime.now(UTC).isoformat()}).encode('utf-8').hex()[:8]}"
    return ContextPack(
        cp_id=cp_id,
        task_id=task_id,
        role=role,
        objective=objective,
        permissions=[ContextPermission(path=path, access_level=level) for path, level in permissions],
        context_budget_tokens=context_budget_tokens,
        required_sections=required_sections,
    )


def resolve_task_ids(spec: ProductSpec | StructuredSpec) -> dict[str, str]:
    """Return mapping from old task_id to canonical task ID scheme.

    Canonical format: T-{pillar_slug}-{epic_slug}-{story_slug}-{seq}
    """

    mapping: dict[str, str] = {}
    used: set[str] = set()

    for pillar in spec.pillars:
        pillar_slug = slugify_name(pillar.name)
        for epic in pillar.epics:
            epic_slug = slugify_name(epic.name)
            for story in epic.stories:
                story_slug = slugify_name(story.name)
                for idx, task in enumerate(story.tasks, start=1):
                    base_id = f"T-{pillar_slug}-{epic_slug}-{story_slug}-{idx}"
                    candidate = base_id
                    suffix_ord = ord("a")
                    while candidate in used:
                        suffix = f"-{chr(suffix_ord)}"
                        candidate = f"{base_id[: 128 - len(suffix)]}{suffix}"
                        suffix_ord += 1
                    used.add(candidate)
                    mapping[task.task_id] = candidate
    return mapping


def apply_canonical_task_ids(spec: ProductSpec | StructuredSpec) -> ProductSpec | StructuredSpec:
    mapping = resolve_task_ids(spec)
    for task in spec.iter_tasks():
        old = task.task_id
        task.task_id = mapping[old]
    for task in spec.iter_tasks():
        task.depends_on = [mapping.get(dep, dep) for dep in task.depends_on]
    return spec


def to_json_file(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def from_json_file(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
