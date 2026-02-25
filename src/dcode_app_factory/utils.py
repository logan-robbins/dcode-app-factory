from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path

from .canonical import to_canonical_json
from .models import (
    AgentConfig,
    BoundaryLevel,
    ContextAccessLevel,
    ContextPack,
    ContextPermission,
    IOContractSketch,
    Pillar,
    ProductSpec,
    Severity,
    Story,
    Task,
    ValidationIssue,
    ValidationReport,
)


TESTABLE_VERB_RE = re.compile(
    r"\b(returns|displays|raises|writes|emits|rejects|validates|produces|creates|records|updates)\b",
    re.IGNORECASE,
)
REQUEST_KIND_VALUES = frozenset({"AUTO", "FULL_APP", "FEATURE", "BUGFIX", "REFACTOR", "TASK"})
_BULLET_RE = re.compile(r"^(?:[-*+]|[0-9]+[.)])\s+(.+)$")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]\s+")


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


def validate_task_dependency_dag(spec: ProductSpec) -> None:
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


def validate_spec(spec: ProductSpec) -> ValidationReport:
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


def emit_structured_spec(spec: ProductSpec, path: Path) -> Path:
    report = validate_spec(spec)
    if report.errors:
        deficiency = "; ".join(f"{entry.path}:{entry.field} {entry.message}" for entry in report.errors)
        raise ValueError(f"Spec emission blocked by validation errors: {deficiency}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    return path


def render_spec_markdown(spec: ProductSpec) -> str:
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


def normalize_request_kind(request_kind: str) -> str:
    normalized = request_kind.strip().upper()
    if normalized not in REQUEST_KIND_VALUES:
        choices = ", ".join(sorted(REQUEST_KIND_VALUES))
        raise ValueError(f"request_kind must be one of: {choices}; got {request_kind!r}")
    return normalized


def _extract_sections(raw_request: str) -> list[str]:
    sections: list[str] = []
    for line in raw_request.splitlines():
        value = line.strip()
        if value.startswith("##"):
            section = value.lstrip("# ").strip()
            if section:
                sections.append(section)
    return sections


def _extract_plain_request_sections(raw_request: str) -> list[str]:
    bullets: list[str] = []
    prose_lines: list[str] = []
    for line in raw_request.splitlines():
        value = line.strip()
        if not value:
            continue
        if value.startswith("#"):
            continue
        bullet_match = _BULLET_RE.match(value)
        if bullet_match:
            bullets.append(bullet_match.group(1).strip())
            continue
        prose_lines.append(value)

    if bullets:
        return [entry for entry in bullets if entry][:6]

    prose = " ".join(prose_lines).strip()
    if not prose:
        return []
    sentences = [segment.strip(" -") for segment in _SENTENCE_SPLIT_RE.split(prose) if segment.strip()]
    if len(sentences) >= 2:
        return sentences[:6]
    return [prose[:160]]


def _extract_title(raw_request: str) -> str | None:
    for line in raw_request.splitlines():
        value = line.strip()
        if value.startswith("# "):
            title = value[2:].strip()
            if title:
                return title
    return None


def _summarize_request(raw_request: str, sections: list[str]) -> str:
    title = _extract_title(raw_request)
    if title:
        return title[:96]
    if sections:
        joined = "; ".join(sections[:3])
        return joined[:96]
    compact = " ".join(raw_request.split())
    return compact[:96] if compact else "requested change"


def _infer_request_kind(raw_request: str, sections: list[str]) -> str:
    content = raw_request.lower()
    if "# product" in content or len(sections) >= 2:
        return "FULL_APP"
    return "TASK"


def _full_app_tasks(sections: list[str]) -> list[Task]:
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
                    f"Audit current implementation surfaces and reuse candidates for {section}",
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
    return tasks


def _incremental_tasks(*, request_summary: str, request_sections: list[str], codebase_hint: str | None) -> list[Task]:
    scope = "; ".join(request_sections[:5]) if request_sections else request_summary
    codebase_scope = (
        f"Target codebase root: {codebase_hint}."
        if codebase_hint
        else "Target codebase root: current workspace checkout."
    )
    return [
        Task(
            task_id="TSK-001",
            name=f"Analyze architecture impact for {request_summary}",
            description=(
                f"Perform product-owner and architect analysis for '{scope}'. "
                f"Document impacted modules, invariants, dependencies, and delivery risks. {codebase_scope}"
            ),
            subtasks=[
                "Map impacted components, services, and data boundaries in the existing codebase",
                "Identify reusable modules and mark create-new boundaries with rationale",
                "Enumerate migration, rollout, and failure-mode risks with mitigation notes",
            ],
            acceptance_criteria=[
                "produces a concrete impact map tied to named code surfaces",
                "records reuse-first decisions and explicit no-reuse reasons",
                "raises blocking risks with clear decision paths",
            ],
            io_contract_sketch=IOContractSketch(
                inputs="Feature request context, current architecture constraints, and repository topology.",
                outputs="Approved architecture impact and reuse decision summary.",
                error_surfaces="Ambiguous requirements, missing constraints, and incompatible existing contracts.",
                effects="Writes architecture and risk decisions into tracked task artifacts.",
                modes="sync with deterministic analysis and fail-fast on missing prerequisites.",
            ),
            depends_on=[],
        ),
        Task(
            task_id="TSK-002",
            name=f"Define contract-first delivery plan for {request_summary}",
            description=(
                f"Define system, service, and component boundaries for '{scope}' with explicit contract surfaces "
                "before implementation."
            ),
            subtasks=[
                "Define contract boundaries across L2/L3/L4 interfaces for the requested change",
                "Specify dependency and compatibility expectations for each boundary",
                "Generate acceptance criteria aligned to verifiable runtime behavior",
            ],
            acceptance_criteria=[
                "produces explicit I/O and error contracts for each planned boundary",
                "validates dependency ordering and compatibility expectations",
                "records testable acceptance criteria for delivery",
            ],
            io_contract_sketch=IOContractSketch(
                inputs="Impact analysis outputs and reusable contract references.",
                outputs="Contract-first implementation plan with ordered module boundaries.",
                error_surfaces="Contract ambiguities, dependency cycles, and unresolved compatibility gaps.",
                effects="Writes contract artifacts and context packs for engineering execution.",
                modes="sync with deterministic dependency ordering and strict validation.",
            ),
            depends_on=["TSK-001"],
        ),
        Task(
            task_id="TSK-003",
            name=f"Implement and verify {request_summary}",
            description=(
                f"Implement '{scope}' against approved boundaries, run verification, and produce release-quality evidence."
            ),
            subtasks=[
                "Implement module changes using approved reusable artifacts and defined contracts",
                "Execute verification checks and capture ship evidence",
                "Confirm release readiness and compatibility for downstream consumers",
            ],
            acceptance_criteria=[
                "creates implementation artifacts that satisfy declared contracts",
                "produces verification evidence demonstrating acceptance coverage",
                "updates release metadata and compatibility outcomes",
            ],
            io_contract_sketch=IOContractSketch(
                inputs="Approved contract plan, context packs, and existing dependency artifacts.",
                outputs="Shippable module artifacts with verification evidence.",
                error_surfaces="Implementation regressions, failed verification checks, and dependency incompatibilities.",
                effects="Writes shipped artifacts, debate outputs, and release gate evidence.",
                modes="sync with deterministic retries bounded by adjudication policy.",
            ),
            depends_on=["TSK-002"],
        ),
    ]


def parse_raw_request_to_product_spec(
    raw_request: str,
    *,
    spec_id: str = "SPEC-001",
    request_kind: str = "AUTO",
    target_codebase_root: str | None = None,
) -> ProductSpec:
    if not raw_request.strip():
        raise ValueError("raw_request must be non-empty")

    sections = _extract_sections(raw_request)
    if not sections:
        sections = _extract_plain_request_sections(raw_request)
    if not sections:
        sections = ["Factory Core", "Project Orchestration", "Engineering Debate"]

    normalized_kind = normalize_request_kind(request_kind)
    if normalized_kind == "AUTO":
        normalized_kind = _infer_request_kind(raw_request, sections)

    request_summary = _summarize_request(raw_request, sections)

    if normalized_kind == "FULL_APP":
        tasks = _full_app_tasks(sections)
        pillar_name = "Reliable Factory Core"
        pillar_description = "Deterministic and auditable software factory execution."
        pillar_rationale = "Reliability and traceability are required for autonomous delivery."
        epic_name = "End-to-end Orchestration"
        epic_description = "Implements product, project, engineering, and release flows."
        story_name = "Deliver deterministic execution"
        story_description = "Ensure execution remains deterministic and auditable."
        behavior = "Operators can run and inspect a complete auditable pipeline."
        title = _extract_title(raw_request) or "AI Software Product Factory"
        description = "Contract-first factory specification generated from markdown source."
    else:
        tasks = _incremental_tasks(
            request_summary=request_summary,
            request_sections=sections,
            codebase_hint=target_codebase_root,
        )
        pillar_name = "Incremental Change Delivery"
        pillar_description = "Contract-first and reuse-first delivery for existing systems."
        pillar_rationale = "Feature velocity must preserve architecture integrity and compatibility."
        epic_name = "Task-Oriented Delivery Loop"
        epic_description = "Runs product, project, engineering, and release loops for any scoped change request."
        story_name = "Deliver change safely in existing codebases"
        story_description = "Turn a task request into analyzed, contract-defined, and verified shipped work."
        behavior = "Operators can submit a single task request and receive a release-gated result."
        title = _extract_title(raw_request) or f"{normalized_kind.title()} Delivery Request"
        description = (
            f"Contract-first work plan generated from a {normalized_kind.lower()} request with "
            "reuse-first and separation-of-concerns constraints."
        )

    now = datetime.now(UTC)
    spec = ProductSpec(
        spec_id=spec_id,
        spec_version="1.0.0",
        title=title,
        description=description,
        created_at=now,
        updated_at=now,
        pillars=[
            Pillar(
                pillar_id="PIL-001",
                name=pillar_name,
                description=pillar_description,
                rationale=pillar_rationale,
                epics=[
                    {
                        "epic_id": "EPC-001",
                        "name": epic_name,
                        "description": epic_description,
                        "success_criteria": [
                            "dispatches tasks deterministically",
                        ],
                        "stories": [
                            Story(
                                story_id="STR-001",
                                name=story_name,
                                description=story_description,
                                user_facing_behavior=behavior,
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
        raise ValueError(f"Unable to construct valid ProductSpec from raw request: {deficiency}")
    return spec


def parse_raw_spec_to_product_spec(raw_spec: str, *, spec_id: str = "SPEC-001") -> ProductSpec:
    return parse_raw_request_to_product_spec(raw_spec, spec_id=spec_id, request_kind="FULL_APP")


def build_context_pack(
    task_id: str,
    objective: str,
    role: str,
    permissions: list[
        tuple[str, ContextAccessLevel]
        | tuple[str, ContextAccessLevel, BoundaryLevel | None, str | None]
    ],
    context_budget_tokens: int,
    required_sections: list[str],
) -> ContextPack:
    cp_id = f"CP-{to_canonical_json({'task_id': task_id, 'role': role, 'ts': datetime.now(UTC).isoformat()}).encode('utf-8').hex()[:8]}"
    normalized_permissions: list[ContextPermission] = []
    for permission in permissions:
        if len(permission) == 2:
            path, access_level = permission
            normalized_permissions.append(ContextPermission(path=path, access_level=access_level))
            continue
        path, access_level, level, contract_ref = permission
        normalized_permissions.append(
            ContextPermission(
                path=path,
                access_level=access_level,
                level=level,
                contract_ref=contract_ref,
            )
        )

    return ContextPack(
        cp_id=cp_id,
        task_id=task_id,
        role=role,
        objective=objective,
        permissions=normalized_permissions,
        context_budget_tokens=context_budget_tokens,
        required_sections=required_sections,
    )


def resolve_task_ids(spec: ProductSpec) -> dict[str, str]:
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


def apply_canonical_task_ids(spec: ProductSpec) -> ProductSpec:
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
