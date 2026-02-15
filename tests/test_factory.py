"""Unit tests for the AI software product factory skeleton."""

from dcode_app_factory import (
    slugify_name,
    to_canonical_json,
    CodeIndex,
    ProductLoop,
    ProjectLoop,
)


def test_slugify_name() -> None:
    assert slugify_name("Hello World!") == "hello-world"
    assert slugify_name("Python_3.12") == "python-3-12"


def test_canonical_json() -> None:
    data = {"b": 2, "a": 1}
    canonical = to_canonical_json(data)
    # Keys should be ordered alphabetically
    assert canonical == "{\"a\":1,\"b\":2}"


def test_code_index_registration_and_lookup() -> None:
    index = CodeIndex()
    from dcode_app_factory.models import MicroModuleContract, InputSpec, OutputSpec

    contract = MicroModuleContract(
        name="Test Module",
        description="A test module",
        inputs=[InputSpec(name="x", type="int")],
        outputs=[OutputSpec(name="y", type="int")],
    )
    slug = index.register(contract)
    assert slug == "test-module"
    found = index.get("Test Module")
    assert found is not None
    assert found.name == contract.name


def test_loops_execution() -> None:
    """Ensure the loops execute end‑to‑end without errors."""
    raw_spec = "Initial spec contents"
    product_loop = ProductLoop(raw_spec)
    spec = product_loop.run()
    code_index = CodeIndex()
    project_loop = ProjectLoop(spec, code_index)
    assert project_loop.run() is True
    # All tasks should be completed
    tasks = [task for pillar in spec.pillars for epic in pillar.epics for story in epic.stories for task in story.tasks]
    assert all(task.status == "completed" for task in tasks)
    # Code index should contain at least one module
    assert len(code_index) >= 1
