from dcode_app_factory import (
    CodeIndex,
    ProductLoop,
    ProjectLoop,
    get_agent_config_dir,
    slugify_name,
    to_canonical_json,
)
from dcode_app_factory.models import IOContractSketch, StructuredSpec, TaskStatus
from dcode_app_factory.utils import load_agent_config, validate_task_dependency_dag


def test_slugify_name() -> None:
    assert slugify_name("Hello World!") == "hello-world"
    assert slugify_name("Python_3.12") == "python-3-12"


def test_canonical_json() -> None:
    assert to_canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_io_contract_sketch_validation() -> None:
    sketch = IOContractSketch(
        inputs=["task payload"],
        outputs=["artifact metadata"],
        error_surfaces=["invalid input"],
        effects=["writes artifact"],
        modes=["normal"],
    )
    sketch.validate_complete()


def test_agent_config_files_load() -> None:
    config_paths = []
    for stage in ("product_loop", "project_loop", "engineering_loop"):
        config_paths.extend(get_agent_config_dir(stage).glob("*.json"))
    assert len(config_paths) >= 10
    cfg = load_agent_config(config_paths[0])
    assert cfg.max_context_tokens > 0
    assert cfg.context_policy


def test_loops_execution_and_dag_validation() -> None:
    raw_spec = "# Product\n## Loop A\n## Loop B\n## Loop C"
    spec: StructuredSpec = ProductLoop(raw_spec).run()
    validate_task_dependency_dag(spec)

    code_index = CodeIndex()
    project_loop = ProjectLoop(spec, code_index)
    assert project_loop.run() is True

    tasks = [
        task
        for pillar in spec.pillars
        for epic in pillar.epics
        for story in epic.stories
        for task in story.tasks
    ]
    assert all(task.status == TaskStatus.COMPLETED for task in tasks)
    assert len(code_index) == len(tasks)
