from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path

from .canonical import to_canonical_json
from .llm import get_structured_chat_model
from .models import (
    AgentConfig,
    BoundaryLevel,
    ContextAccessLevel,
    ContextPack,
    ContextPermission,
    ProductSpec,
    Severity,
    ValidationIssue,
    ValidationReport,
)
from .settings import RuntimeSettings


TESTABLE_VERB_RE = re.compile(
    r"\b(returns|displays|raises|writes|emits|rejects|validates|produces|creates|records|updates)\b",
    re.IGNORECASE,
)
REQUEST_KIND_VALUES = frozenset({"AUTO", "FULL_APP", "FEATURE", "BUGFIX", "REFACTOR", "TASK"})


def get_agent_config_dir(stage: str) -> Path:
    """Return package-relative path to agent configs for a stage."""
    return Path(__file__).resolve().parent / "agent_configs" / stage


def load_agent_config(path: Path) -> AgentConfig:
    """Load and validate an AgentConfig from a JSON file on disk."""
    return AgentConfig.model_validate_json(path.read_text(encoding="utf-8"))


def slugify_name(name: str, *, max_length: int = 24) -> str:
    """Convert a human-readable name to a lowercase hyphenated slug, capped at max_length."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:max_length].rstrip("-")


def dedupe_slug(base_slug: str, used: set[str], *, max_length: int = 24) -> str:
    """Return a unique slug by appending a suffix if base_slug is already in used."""
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
    """Verify the task dependency graph is a valid DAG with no unknown references."""
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
    """Validate a ProductSpec for structural completeness and quality.

    This function is called both programmatically during spec creation and by
    agent tools during the product loop review.  It must catch every deficiency
    that would cause downstream engineering failures.
    """
    issues: list[ValidationIssue] = []

    # -- Top-level field checks --
    if not spec.title.strip():
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                path="title",
                field="title",
                message="Spec title must be non-empty",
            )
        )

    if len(spec.description.strip()) < 50:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                path="description",
                field="description",
                message="Spec description must be at least 50 characters of substantive content",
            )
        )

    if not spec.pillars:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                path="pillars",
                field="pillars",
                message="Spec must include at least one pillar",
            )
        )

    # -- Pillar name uniqueness --
    pillar_names: set[str] = set()
    for p_idx, pillar in enumerate(spec.pillars):
        p_name_lower = pillar.name.strip().lower()
        if p_name_lower in pillar_names:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    path=f"pillars[{p_idx}]",
                    field="name",
                    message=f"Duplicate pillar name: {pillar.name}",
                )
            )
        pillar_names.add(p_name_lower)

    task_ids: set[str] = set()
    for p_idx, pillar in enumerate(spec.pillars):
        p_path = f"pillars[{p_idx}]"

        # -- Epic name uniqueness within pillar --
        epic_names: set[str] = set()
        for e_idx, epic in enumerate(pillar.epics):
            e_name_lower = epic.name.strip().lower()
            if e_name_lower in epic_names:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        path=f"{p_path}.epics[{e_idx}]",
                        field="name",
                        message=f"Duplicate epic name within pillar: {epic.name}",
                    )
                )
            epic_names.add(e_name_lower)

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

            # -- Story name uniqueness within epic --
            story_names: set[str] = set()
            for s_idx, story in enumerate(epic.stories):
                s_name_lower = story.name.strip().lower()
                if s_name_lower in story_names:
                    issues.append(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            path=f"{e_path}.stories[{s_idx}]",
                            field="name",
                            message=f"Duplicate story name within epic: {story.name}",
                        )
                    )
                story_names.add(s_name_lower)

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

                    if len(task.description.strip()) < 30:
                        issues.append(
                            ValidationIssue(
                                severity=Severity.ERROR,
                                path=t_path,
                                field="description",
                                message="Task description must be at least 30 characters of substantive content",
                            )
                        )

                    for st_idx, subtask_text in enumerate(task.subtasks):
                        if len(subtask_text.strip()) < 10:
                            issues.append(
                                ValidationIssue(
                                    severity=Severity.ERROR,
                                    path=t_path,
                                    field=f"subtasks[{st_idx}]",
                                    message="Each subtask must be at least 10 characters of meaningful content",
                                )
                            )

                    for ac_idx, criterion_text in enumerate(task.acceptance_criteria):
                        if len(criterion_text.strip()) < 10:
                            issues.append(
                                ValidationIssue(
                                    severity=Severity.ERROR,
                                    path=t_path,
                                    field=f"acceptance_criteria[{ac_idx}]",
                                    message="Each acceptance criterion must be at least 10 characters of meaningful content",
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
        # Self-referencing dependency check
        if task.task_id in task.depends_on:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    path=task.task_id,
                    field="depends_on",
                    message=f"Task {task.task_id} depends on itself",
                )
            )
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
    """Validate and write a ProductSpec to disk as indented JSON. Returns the written path."""
    report = validate_spec(spec)
    if report.errors:
        deficiency = "; ".join(f"{entry.path}:{entry.field} {entry.message}" for entry in report.errors)
        raise ValueError(f"Spec emission blocked by validation errors: {deficiency}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    return path


def render_spec_markdown(spec: ProductSpec) -> str:
    """Render a ProductSpec as a human-readable Markdown document."""
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
    """Normalize and validate a request kind string. Raises ValueError if invalid."""
    normalized = request_kind.strip().upper()
    if normalized not in REQUEST_KIND_VALUES:
        choices = ", ".join(sorted(REQUEST_KIND_VALUES))
        raise ValueError(f"request_kind must be one of: {choices}; got {request_kind!r}")
    return normalized


def _request_kind_hint_for_planner(raw_request: str, normalized_kind: str) -> str:
    if normalized_kind != "AUTO":
        return normalized_kind
    lowered = raw_request.lower()
    if "# product" in lowered:
        return "FULL_APP"
    return "TASK"


def _invoke_product_spec_planner(
    *,
    raw_request: str,
    spec_id: str,
    request_kind: str,
    target_codebase_root: str | None,
) -> ProductSpec:
    settings = RuntimeSettings.from_env()
    planner = get_structured_chat_model(
        model_name=settings.model_frontier,
        schema=ProductSpec,
        temperature=0.0,
        method="function_calling",
        strict=True,
        include_raw=False,
    )
    now_iso = datetime.now(UTC).isoformat()
    kind_hint = _request_kind_hint_for_planner(raw_request, request_kind)
    target_scope = target_codebase_root or "current workspace checkout"
    prompt = (
        "You are the ProductSpec planner for a production software factory. "
        "Your output is a structured ProductSpec JSON that will be machine-validated and "
        "then reviewed by researcher, structurer, and validator agents before reaching "
        "engineering. If your output fails validation, the pipeline halts.\n\n"
        "Return ProductSpec JSON only; do not wrap in markdown.\n\n"
        f"spec_id must be exactly: {spec_id}\n"
        "spec_version must be exactly: 1.0.0\n"
        f"request_kind hint: {kind_hint}\n"
        f"target_codebase_root: {target_scope}\n"
        f"current_time_utc: {now_iso}\n\n"
        "STRUCTURAL REQUIREMENTS (violations cause hard failures):\n"
        "- At least 1 pillar, each with at least 1 epic, each with at least 1 story, "
        "each with at least 1 task.\n"
        "- Use task IDs in TSK-NNN format (e.g., TSK-001). These will be canonically remapped.\n"
        "- Every task must have >=2 subtasks and >=2 acceptance criteria.\n"
        "- Every task must include all io_contract_sketch fields (inputs, outputs, "
        "error_surfaces, effects, modes) as non-empty, non-placeholder strings.\n"
        "- Task depends_on values must reference existing task IDs and form an acyclic DAG.\n"
        "- No task may list itself in its own depends_on.\n"
        "- Pillar names must be unique. Epic names must be unique within their pillar. "
        "Story names must be unique within their epic.\n\n"
        "QUALITY REQUIREMENTS (violations cause reviewer rejections):\n"
        "- Title must clearly describe the product or feature being built.\n"
        "- Description must be a substantive paragraph (>=50 characters) explaining scope "
        "and purpose, not a restatement of the title.\n"
        "- Each pillar must represent a distinct architectural concern (e.g., data layer, "
        "API surface, auth/identity, observability) -- not a restatement of another pillar.\n"
        "- Each pillar must have a non-trivial rationale explaining why it is a separate concern.\n"
        "- Each epic must have at least one concrete success criterion that is measurable.\n"
        "- Each story must describe user-observable behaviour, not implementation steps.\n"
        "- Each task description must be >=30 characters of real content explaining WHAT "
        "the task delivers and WHY.\n"
        "- Subtasks must collectively cover the task scope. Each subtask must be a distinct "
        "action, not a restatement of another subtask.\n"
        "- Acceptance criteria must use testable/observable verbs (returns, displays, raises, "
        "writes, emits, rejects, validates, produces, creates, records, updates). Each "
        "criterion must specify the expected observable outcome.\n"
        "- io_contract_sketch fields must specify concrete types and formats:\n"
        "  - inputs: name the data types, sources, and required fields.\n"
        "  - outputs: name the data types, formats, and destination.\n"
        "  - error_surfaces: list specific error conditions and how they manifest.\n"
        "  - effects: describe observable side effects (file writes, DB mutations, API calls).\n"
        "  - modes: describe operational modes or feature flags if applicable, or 'single-mode' "
        "    with justification.\n"
        "- Dependencies must be justified: only add depends_on when task B literally cannot "
        "start without task A's output. Do not add dependencies for convenience.\n\n"
        "PROHIBITED PATTERNS:\n"
        "- No mock, stub, fake, or placeholder implementations.\n"
        "- No 'TBD', 'TODO', 'N/A', 'any', or 'data' as io_contract_sketch values.\n"
        "- No generic names like 'Pillar 1', 'Epic A', 'Story X', 'Task 1'.\n"
        "- No duplicate subtask text within a task.\n"
        "- No acceptance criteria that are untestable opinions (e.g., 'code is clean').\n\n"
        "RAW REQUEST:\n"
        f"{raw_request}"
    )
    return planner.invoke(prompt)


def parse_raw_request_to_product_spec(
    raw_request: str,
    *,
    spec_id: str = "SPEC-001",
    request_kind: str = "AUTO",
    target_codebase_root: str | None = None,
) -> ProductSpec:
    """Parse a raw text request into a validated ProductSpec via LLM-driven planning.

    Invokes the frontier model to generate a structured ProductSpec, then validates
    the result for structural completeness and DAG integrity.

    Raises:
        ValueError: If the raw_request is empty, kind is invalid, or the generated
            spec fails validation.
    """
    if not raw_request.strip():
        raise ValueError("raw_request must be non-empty")
    normalized_kind = normalize_request_kind(request_kind)
    spec = _invoke_product_spec_planner(
        raw_request=raw_request,
        spec_id=spec_id,
        request_kind=normalized_kind,
        target_codebase_root=target_codebase_root,
    )
    spec.spec_id = spec_id
    spec.spec_version = "1.0.0"
    if spec.updated_at < spec.created_at:
        spec.updated_at = spec.created_at

    report = validate_spec(spec)
    if report.errors:
        deficiency = "; ".join(f"{entry.path}:{entry.message}" for entry in report.errors)
        raise ValueError(f"Unable to construct valid ProductSpec from raw request: {deficiency}")
    validate_task_dependency_dag(spec)
    return spec


def parse_raw_spec_to_product_spec(raw_spec: str, *, spec_id: str = "SPEC-001") -> ProductSpec:
    """Convenience wrapper: parse a raw spec as a FULL_APP request kind."""
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
    """Replace all task IDs in the spec with canonical hierarchical IDs in place."""
    mapping = resolve_task_ids(spec)
    for task in spec.iter_tasks():
        old = task.task_id
        task.task_id = mapping[old]
    for task in spec.iter_tasks():
        task.depends_on = [mapping.get(dep, dep) for dep in task.depends_on]
    return spec


def to_json_file(data: dict[str, object], path: Path) -> None:
    """Write a dictionary to a JSON file with sorted keys and 2-space indent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def from_json_file(path: Path) -> dict[str, object]:
    """Read and parse a JSON file, returning the deserialized dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))
