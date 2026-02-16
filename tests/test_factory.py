from dcode_app_factory import (
    CodeIndex,
    EngineeringLoop,
    ProductLoop,
    ProjectLoop,
    TaskStatus,
    get_agent_config_dir,
    slugify_name,
    to_canonical_json,
)
from dcode_app_factory.models import IOContractSketch, StructuredSpec
from dcode_app_factory.utils import build_context_pack, load_agent_config, validate_task_dependency_dag

import pytest


def test_slugify_name() -> None:
    assert slugify_name("Hello World!") == "hello-world"
    assert slugify_name("Python_3.12") == "python-3-12"


def test_canonical_json() -> None:
    assert to_canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_io_contract_sketch_validation_rejects_placeholders() -> None:
    sketch = IOContractSketch(
        inputs=["task payload"],
        outputs=["artifact metadata"],
        error_surfaces=["TODO"],
        effects=["writes artifact"],
        modes=["normal"],
    )
    with pytest.raises(ValueError):
        sketch.validate_complete()


def test_agent_config_files_load_for_all_stages() -> None:
    config_paths = []
    for stage in ("product_loop", "project_loop", "engineering_loop"):
        config_paths.extend(get_agent_config_dir(stage).glob("*.json"))
    assert len(config_paths) >= 11
    for path in config_paths:
        cfg = load_agent_config(path)
        assert cfg.max_context_tokens > 0
        assert cfg.allowed_context_sections


def test_product_loop_generates_structurally_valid_spec() -> None:
    raw_spec = "# Product\n## Loop A\n## Loop B\n## Loop C\n## Loop D"
    spec: StructuredSpec = ProductLoop(raw_spec).run()
    issues = spec.validate()
    assert not [issue for issue in issues if issue.severity == "ERROR"]
    validate_task_dependency_dag(spec)


def test_context_pack_budget_respects_agent_limits() -> None:
    cfg = load_agent_config(get_agent_config_dir("engineering_loop") / "proposer.json")
    context = build_context_pack("TASK-001-a", "test", "engineering_loop", cfg)
    assert context.context_budget_tokens <= cfg.max_context_tokens
    assert "task_contract" in context.required_sections


def test_end_to_end_project_loop_happy_path() -> None:
    raw_spec = "# Product\n## Loop A\n## Loop B\n## Loop C"
    spec = ProductLoop(raw_spec).run()

    code_index = CodeIndex()
    project_loop = ProjectLoop(spec, code_index)
    assert project_loop.run() is True

    tasks = spec.iter_tasks()
    assert all(task.status == TaskStatus.COMPLETED for task in tasks)
    assert all(task.contract is not None for task in tasks)
    assert all(task.ship_evidence is not None for task in tasks)
    assert len(code_index) == len(tasks)


def test_project_loop_blocks_downstream_when_engineering_fails() -> None:
    spec = ProductLoop("# Product\n## A\n## B").run()
    first_task, second_task = spec.iter_tasks()[:2]

    class FailingEngineeringLoop(EngineeringLoop):
        def _proposer(self, task_prompt: str, _context) -> str:
            _ = task_prompt
            return "proposal: incomplete"

    original = ProjectLoop.run

    def forced_run(self: ProjectLoop) -> bool:
        validate_task_dependency_dag(self.spec)
        tasks = self._topological_order(self._flatten_tasks())
        for task in tasks:
            runner = FailingEngineeringLoop(task=task, code_index=self.code_index, max_retries=0)
            if not runner.run():
                self._mark_downstream_blocked(task.task_id)
                return False
        return True

    try:
        ProjectLoop.run = forced_run  # type: ignore[method-assign]
        assert ProjectLoop(spec, CodeIndex()).run() is False
        assert first_task.status == TaskStatus.HALTED
        assert second_task.status == TaskStatus.BLOCKED
    finally:
        ProjectLoop.run = original  # type: ignore[method-assign]
