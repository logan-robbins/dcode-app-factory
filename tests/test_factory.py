import os
import subprocess
import sys
from pathlib import Path

import pytest

from dcode_app_factory import (
    CodeIndex,
    DEFAULT_MODELS_BY_TIER,
    EngineeringLoop,
    ProductLoop,
    ProjectLoop,
    RuntimeModelSelection,
    TaskStatus,
    get_agent_config_dir,
    slugify_name,
    to_canonical_json,
)
from dcode_app_factory.models import IOContractSketch, StructuredSpec
from dcode_app_factory.settings import RuntimeSettings
from dcode_app_factory.utils import build_context_pack, load_agent_config, validate_task_dependency_dag


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_slugify_name() -> None:
    assert slugify_name("Hello World!") == "hello-world"
    assert slugify_name("Python_3.12") == "python-3-12"


def test_canonical_json() -> None:
    left = {"b": 2, "a": 1, "nested": {"z": 9, "y": [3, 2, 1]}}
    right = {"nested": {"y": [3, 2, 1], "z": 9}, "a": 1, "b": 2}
    assert to_canonical_json(left) == to_canonical_json(right)


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


def test_product_loop_section_limit_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACTORY_MAX_PRODUCT_SECTIONS", "3")
    raw_spec = "\n".join(["# Product"] + [f"## Section {idx}" for idx in range(1, 8)])
    spec = ProductLoop(raw_spec).run()
    assert len(spec.iter_tasks()) == 3


def test_runtime_settings_from_env_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACTORY_CONTEXT_BUDGET_CAP", "7000")
    monkeypatch.setenv("FACTORY_CONTEXT_BUDGET_FLOOR", "3000")
    settings = RuntimeSettings.from_env()
    assert settings.context_budget_cap_tokens == 7000
    assert settings.context_budget_floor_tokens == 3000


def test_runtime_settings_invalid_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACTORY_CONTEXT_BUDGET_CAP", "abc")
    with pytest.raises(ValueError):
        RuntimeSettings.from_env()


def test_context_pack_budget_respects_agent_and_runtime_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACTORY_CONTEXT_BUDGET_FLOOR", "2500")
    monkeypatch.setenv("FACTORY_CONTEXT_BUDGET_CAP", "3500")
    cfg = load_agent_config(get_agent_config_dir("engineering_loop") / "proposer.json")
    context = build_context_pack("TASK-001-a", "test", "engineering_loop", cfg)
    assert 2500 <= context.context_budget_tokens <= 3500
    assert "task_contract" in context.required_sections


def test_default_models_by_tier_are_exposed() -> None:
    assert DEFAULT_MODELS_BY_TIER["frontier"]
    assert DEFAULT_MODELS_BY_TIER["efficient"]


def test_runtime_model_selection_tier_and_role_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACTORY_MODEL_FRONTIER", "openai:gpt-5.2-pro")
    monkeypatch.setenv(
        "FACTORY_MODEL_ROLE_OVERRIDES_JSON",
        '{"engineering_loop.proposer": "anthropic:claude-sonnet-4.5", "arbiter": "openai:gpt-5.2-mini"}',
    )

    selection = RuntimeModelSelection.from_env()
    assert selection.resolve("product_loop", "researcher", "frontier") == "openai:gpt-5.2-pro"
    assert selection.resolve("engineering_loop", "proposer", "frontier") == "anthropic:claude-sonnet-4.5"
    assert selection.resolve("engineering_loop", "arbiter", "efficient") == "openai:gpt-5.2-mini"


def test_runtime_model_selection_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACTORY_MODEL_ROLE_OVERRIDES_JSON", "[]")
    with pytest.raises(ValueError):
        RuntimeModelSelection.from_env()


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


def test_cli_end_to_end_runs_with_explicit_spec(tmp_path: Path) -> None:
    spec_file = tmp_path / "tiny_spec.md"
    spec_file.write_text("# Product\n## A\n## B\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "factory_main.py"),
            "--spec-file",
            str(spec_file),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "project_success=True" in result.stdout
    assert "registered_modules:" in result.stdout


def test_cli_uses_configurable_default_spec_path(tmp_path: Path) -> None:
    spec_file = tmp_path / "custom-default-spec.md"
    spec_file.write_text("# Product\n## A\n", encoding="utf-8")

    env = os.environ.copy()
    env["FACTORY_DEFAULT_SPEC_PATH"] = str(spec_file)

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "factory_main.py")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "project_success=True" in result.stdout
