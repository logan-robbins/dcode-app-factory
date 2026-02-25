from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import json

import pytest
from deepagents.backends import FilesystemBackend

from dcode_app_factory import (
    Adjudication,
    ArtifactEnvelope,
    ArtifactStatus,
    ArtifactType,
    Challenge,
    CodeIndex,
    CodeIndexStatus,
    EngineeringLoop,
    FactoryOrchestrator,
    MicroModuleContract,
    MicroPlan,
    ProductLoop,
    ProjectLoop,
    Proposal,
    ReuseSearchReport,
    TaskStatus,
    validate_spec,
)
from dcode_app_factory.backends import (
    ContextAccessLevel,
    ContextPackBackend,
    ImmutableArtifactBackend,
    OpaqueEnforcementBackend,
    SEALED_ACCESS_ERROR_TEMPLATE,
    seal_module_version,
)
from dcode_app_factory.agent_runtime import RoleAgentRuntime
from dcode_app_factory.debate import DebateGraph, DebateResult
from dcode_app_factory.llm import get_structured_chat_model, normalize_structured_output
from dcode_app_factory.loops import ReleaseLoop
from dcode_app_factory.models import (
    BoundaryLevel,
    ChallengeFailure,
    AdjudicationDecision,
    CompatibilityExpectation,
    ContractCompatibility,
    CompatibilityRule,
    ContractErrorSurface,
    ContractInput,
    ContractModes,
    ContractOutput,
    EffectType,
    InterfaceChangeType,
    ContextPack,
    ContextPermission,
    HumanResolutionAction,
    InterfaceChangeException,
    ReuseDecision,
    ReuseConclusion,
    MicroIoContract,
    MicroPlanModule,
    ProjectState,
    ReuseSearchCandidate,
    ProductRoleReport,
    DependencyManagerDecision,
    DispatchDecision,
    StateAuditDecision,
    MicroPlanReview,
    ShipperDecision,
    ReleaseGateDecision,
    ReleaseManagerDecision,
    ProposedContractDelta,
    RaisedBy,
    RubricAssessment,
    RuntimeBudgets,
    ShipDirective,
    Task,
    Urgency,
)
from dcode_app_factory.state_store import ArtifactStoreService, FactoryStateStore, build_project_state, project_scoped_root
from dcode_app_factory.settings import RuntimeSettings
from dcode_app_factory.utils import (
    apply_canonical_task_ids,
    parse_raw_request_to_product_spec,
    parse_raw_spec_to_product_spec,
    resolve_task_ids,
    slugify_name,
    validate_task_dependency_dag,
)


def _pass_debate_result(target_artifact_id: str) -> DebateResult:
    proposal = Proposal(
        proposal_id="PROP-deadbeef",
        target_artifact_id=target_artifact_id,
        claim="Implements module behavior",
        deliverable_ref="/modules/module",
        acceptance_checks=["returns expected output"],
    )
    challenge = Challenge(
        challenge_id="CHAL-deadbeef",
        target_artifact_id=target_artifact_id,
        verdict="PASS",
        failures=[],
        rubric_assessments=[
            RubricAssessment(criterion=f"R{i}", assessment="MET", evidence="verified")
            for i in range(1, 7)
        ],
    )
    adjudication = Adjudication(
        adjudication_id="ADJ-deadbeef",
        target_artifact_id=target_artifact_id,
        decision=AdjudicationDecision.APPROVE,
        amendments=[],
        rationale="All rubric checks met",
        ship_directive=ShipDirective.SHIP,
    )
    return DebateResult(
        passed=True,
        proposal=proposal,
        challenge=challenge,
        adjudication=adjudication,
        retries_used=0,
    )


def _fail_debate_result(target_artifact_id: str) -> DebateResult:
    proposal = Proposal(
        proposal_id="PROP-feedface",
        target_artifact_id=target_artifact_id,
        claim="Implementation has unresolved issues",
        deliverable_ref="/modules/module",
        acceptance_checks=["returns expected output"],
    )
    challenge = Challenge(
        challenge_id="CHAL-feedface",
        target_artifact_id=target_artifact_id,
        verdict="FAIL",
        failures=[
            ChallengeFailure(
                invariant="contract compliance",
                evidence="detected mismatch",
                required_change="fix output contract",
            )
        ],
        rubric_assessments=[
            RubricAssessment(criterion="R1", assessment="NOT_MET", evidence="failing contract evidence"),
            RubricAssessment(criterion="R2", assessment="MET", evidence="verified"),
            RubricAssessment(criterion="R3", assessment="MET", evidence="verified"),
            RubricAssessment(criterion="R4", assessment="MET", evidence="verified"),
            RubricAssessment(criterion="R5", assessment="MET", evidence="verified"),
            RubricAssessment(criterion="R6", assessment="MET", evidence="verified"),
        ],
    )
    adjudication = Adjudication(
        adjudication_id="ADJ-feedface",
        target_artifact_id=target_artifact_id,
        decision=AdjudicationDecision.REJECT,
        amendments=[],
        rationale="Challenge demonstrated a material violation",
        ship_directive=ShipDirective.NO_SHIP,
    )
    return DebateResult(
        passed=False,
        proposal=proposal,
        challenge=challenge,
        adjudication=adjudication,
        retries_used=2,
    )


@pytest.fixture(autouse=True)
def _deterministic_embeddings_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACTORY_EMBEDDING_MODEL", "deterministic-hash-384")


@pytest.fixture(autouse=True)
def _mock_role_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Adapter:
        def __init__(self, schema: object) -> None:
            self.schema = schema

        def invoke(self, prompt: str):  # noqa: ANN201
            _ = prompt
            if self.schema is Proposal:
                return Proposal(
                    proposal_id="PROP-deadbeef",
                    target_artifact_id="CTR-deadbeef",
                    claim="Implements module behavior",
                    deliverable_ref="/modules/module",
                    acceptance_checks=["returns expected output"],
                )
            if self.schema is Challenge:
                return Challenge(
                    challenge_id="CHAL-deadbeef",
                    target_artifact_id="CTR-deadbeef",
                    verdict="PASS",
                    failures=[],
                    rubric_assessments=[
                        RubricAssessment(criterion=f"R{i}", assessment="MET", evidence="verified")
                        for i in range(1, 7)
                    ],
                )
            if self.schema is Adjudication:
                return Adjudication(
                    adjudication_id="ADJ-deadbeef",
                    target_artifact_id="CTR-deadbeef",
                    decision=AdjudicationDecision.APPROVE,
                    amendments=[],
                    rationale="All checks met",
                    ship_directive=ShipDirective.SHIP,
                )
            raise AssertionError(f"unexpected schema adapter request: {self.schema}")

    def _fake_structured_adapter(self, *, role: str, schema):  # noqa: ANN001,ANN201
        _ = self, role
        return _Adapter(schema)

    def _extract_json_after(prompt: str, key: str) -> dict[str, str]:
        for line in prompt.splitlines():
            if line.startswith(f"{key}="):
                return json.loads(line.split("=", 1)[1])
        return {}

    def _extract_list_after(prompt: str, key: str) -> list[str]:
        for line in prompt.splitlines():
            if line.startswith(f"{key}="):
                return json.loads(line.split("=", 1)[1])
        return []

    def _extract_value_after(prompt: str, key: str) -> str:
        for line in prompt.splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
        return ""

    def _fake_invoke_structured(self, *, role: str, schema, prompt: str):  # noqa: ANN001,ANN201
        _ = self, role
        if schema is DependencyManagerDecision:
            return schema(approved=True, rationale="dependency graph valid", blocking_dependencies=[])
        if schema is DispatchDecision:
            queue = _extract_list_after(prompt, "ReadyQueue")
            selected = queue[0] if queue else None
            return schema(selected_task_id=selected, rationale="selected first ready task")
        if schema is StateAuditDecision:
            return schema(valid=True, rationale="state transition valid", findings=[])
        if schema is MicroPlanReview:
            return schema(approved=True, rationale="micro plan approved", blockers=[], required_revisions=[])
        if schema is ShipperDecision:
            return schema(ship_directive=ShipDirective.SHIP, rationale="ship approved", required_fixes=[])
        if schema is ReleaseGateDecision:
            gates = _extract_json_after(prompt, "ObjectiveGateEvidence")
            return schema(
                dependency_check=gates.get("dependency_check", "PASS"),
                fingerprint_check=gates.get("fingerprint_check", "PASS"),
                deprecation_check=gates.get("deprecation_check", "PASS"),
                code_index_check=gates.get("code_index_check", "PASS"),
                contract_completeness_check=gates.get("contract_completeness_check", "PASS"),
                compatibility_check=gates.get("compatibility_check", "PASS"),
                ownership_check=gates.get("ownership_check", "PASS"),
                context_pack_compliance_check=gates.get("context_pack_compliance_check", "PASS"),
                notes=[],
            )
        if schema is ReleaseManagerDecision:
            gate_overall = _extract_value_after(prompt, "GateOverall")
            overall = "FAIL" if gate_overall == "FAIL" else "PASS"
            return schema(overall_result=overall, rationale="release manager finalized", release_notes=[])
        raise AssertionError(f"unexpected invoke_structured schema: {schema}")

    def _fake_invoke_deepagent_json(
        self,
        *,
        role: str,
        system_prompt: str,
        user_message: str,
        tools: list,
        name: str,
    ) -> dict[str, object]:
        _ = self, role, system_prompt, user_message, tools, name
        return ProductRoleReport(
            approved=True,
            summary="approved",
            warnings=[],
            blocking_issues=[],
            recommended_actions=[],
        ).model_dump(mode="json")

    monkeypatch.setattr("dcode_app_factory.agent_runtime.RoleAgentRuntime.structured_adapter", _fake_structured_adapter)
    monkeypatch.setattr("dcode_app_factory.agent_runtime.RoleAgentRuntime.invoke_structured", _fake_invoke_structured)
    monkeypatch.setattr("dcode_app_factory.agent_runtime.RoleAgentRuntime.invoke_deepagent_json", _fake_invoke_deepagent_json)


class _FakeDebateGraphPass:
    def __init__(
        self,
        *,
        store: FactoryStateStore,
        role_runtime: RoleAgentRuntime,
        use_llm: bool = True,
        propagate_parent_halt: bool = False,
    ) -> None:
        _ = store, role_runtime, use_llm, propagate_parent_halt

    def run(
        self,
        *,
        task_id: str,
        module_id: str,
        target_artifact_id: str,
        contract_summary: dict,
        context_summary: str,
        context_pack_refs: dict[str, str] | None = None,
        max_retries: int = 2,
    ) -> DebateResult:
        _ = task_id, module_id, contract_summary, context_summary, context_pack_refs, max_retries
        return _pass_debate_result(target_artifact_id)


class _FakeDebateGraphFailFirst:
    def __init__(
        self,
        *,
        store: FactoryStateStore,
        role_runtime: RoleAgentRuntime,
        use_llm: bool = True,
        propagate_parent_halt: bool = False,
    ) -> None:
        _ = store, role_runtime, use_llm, propagate_parent_halt

    def run(
        self,
        *,
        task_id: str,
        module_id: str,
        target_artifact_id: str,
        contract_summary: dict,
        context_summary: str,
        context_pack_refs: dict[str, str] | None = None,
        max_retries: int = 2,
    ) -> DebateResult:
        _ = task_id, contract_summary, context_summary, context_pack_refs, max_retries
        if module_id.endswith("-01"):
            return _fail_debate_result(target_artifact_id)
        return _pass_debate_result(target_artifact_id)


def test_slugify_name_has_24_char_cap() -> None:
    slug = slugify_name("This is a very long section name with many words")
    assert len(slug) <= 24
    assert " " not in slug


def test_product_parser_and_validation() -> None:
    spec = parse_raw_spec_to_product_spec("# Product\n## Build API\n## Deploy Pipeline")
    report = validate_spec(spec)
    assert not report.errors
    validate_task_dependency_dag(spec)


def test_request_parser_supports_incremental_feature_mode() -> None:
    spec = parse_raw_request_to_product_spec(
        "Add a user profile settings screen with avatar upload and notification toggles.",
        request_kind="FEATURE",
        target_codebase_root="/tmp/existing-repo",
    )

    tasks = spec.iter_tasks()
    assert spec.pillars[0].name == "Incremental Change Delivery"
    assert len(tasks) == 3
    assert [task.task_id for task in tasks] == ["TSK-001", "TSK-002", "TSK-003"]
    assert tasks[1].depends_on == ["TSK-001"]
    assert tasks[2].depends_on == ["TSK-002"]
    assert "Analyze architecture impact" in tasks[0].name


def test_request_parser_auto_mode_prefers_full_app_with_markdown_sections() -> None:
    spec = parse_raw_request_to_product_spec("# Product\n## A\n## B", request_kind="AUTO")
    tasks = spec.iter_tasks()
    assert spec.pillars[0].name == "Reliable Factory Core"
    assert len(tasks) == 2
    assert tasks[0].name == "Implement A"
    assert tasks[1].depends_on == ["TSK-001"]


def test_canonical_task_ids_are_unique_and_hierarchical() -> None:
    spec = parse_raw_spec_to_product_spec("# Product\n## A\n## B\n## C")
    mapping = resolve_task_ids(spec)
    assert len(mapping.values()) == len(set(mapping.values()))
    apply_canonical_task_ids(spec)
    for task in spec.iter_tasks():
        assert task.task_id.startswith("T-")


def test_context_pack_backend_contract_only_enforcement(tmp_path: Path) -> None:
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    modules_dir = tmp_path / "modules" / "MM-1" / "1.0.0"
    modules_dir.mkdir(parents=True)
    (modules_dir / "contract.json").write_text("{}", encoding="utf-8")
    (modules_dir / "implementation" / "main.py").parent.mkdir(parents=True)
    (modules_dir / "implementation" / "main.py").write_text("print('secret')", encoding="utf-8")

    cp = ContextPack(
        cp_id="CP-1234abcd",
        task_id="T-a-b-c-1",
        role="challenger",
        objective="review contract",
        permissions=[ContextPermission(path="/modules", access_level=ContextAccessLevel.CONTRACT_ONLY)],
        context_budget_tokens=2000,
        required_sections=["contract"],
    )
    wrapped = ContextPackBackend(backend, cp)
    assert "print('secret')" not in wrapped.read("/modules/MM-1/1.0.0/implementation/main.py")
    assert "{}" in wrapped.read("/modules/MM-1/1.0.0/contract.json")


def test_context_pack_backend_contract_ref_scoping(tmp_path: Path) -> None:
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    first = tmp_path / "modules" / "MM-1" / "1.0.0"
    second = tmp_path / "modules" / "MM-2" / "1.0.0"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "contract.json").write_text("{\"id\":\"MM-1\"}", encoding="utf-8")
    (second / "contract.json").write_text("{\"id\":\"MM-2\"}", encoding="utf-8")

    cp = ContextPack(
        cp_id="CP-8765abcd",
        task_id="T-a-b-c-1",
        role="challenger",
        objective="review one module contract",
        permissions=[
            ContextPermission(
                path="/modules",
                access_level=ContextAccessLevel.CONTRACT_ONLY,
                level=BoundaryLevel.L4_COMPONENT,
                contract_ref="MM-1@1.0.0",
            )
        ],
        context_budget_tokens=2000,
        required_sections=["contract"],
    )
    wrapped = ContextPackBackend(backend, cp)
    assert "MM-1" in wrapped.read("/modules/MM-1/1.0.0/contract.json")
    denied = wrapped.read("/modules/MM-2/1.0.0/contract.json")
    assert "Access denied by context pack" in denied


def test_opaque_backend_returns_required_error_format(tmp_path: Path) -> None:
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    version_dir = tmp_path / "modules" / "MM-opaque" / "1.0.0"
    impl_dir = version_dir / "implementation"
    impl_dir.mkdir(parents=True)
    (impl_dir / "secret.py").write_text("print('secret')", encoding="utf-8")
    seal_module_version(version_dir)

    wrapped = OpaqueEnforcementBackend(backend)
    blocked_path = "/modules/MM-opaque/1.0.0/implementation/secret.py"
    message = wrapped.read(blocked_path)
    assert message == SEALED_ACCESS_ERROR_TEMPLATE.format(path=blocked_path)


def test_immutable_backend_blocks_in_place_updates(tmp_path: Path) -> None:
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    artifacts_path = tmp_path / "artifacts" / "SHIP-deadbeef"
    artifacts_path.mkdir(parents=True)
    artifact_file = artifacts_path / "envelope.json"
    artifact_file.write_text("{}", encoding="utf-8")

    wrapped = ImmutableArtifactBackend(backend)
    result = wrapped.edit("/artifacts/SHIP-deadbeef/envelope.json", "{}", "{\"x\":1}")
    assert result.error


def test_artifact_store_status_transitions(tmp_path: Path) -> None:
    store = FactoryStateStore(tmp_path)
    service = ArtifactStoreService(store)
    envelope = ArtifactEnvelope.build(
        artifact_type=ArtifactType.CONTRACT,
        created_by={"role": "proposer", "run_id": "run-1"},
        context_pack_ref="CP-abcdef12",
        payload={"x": 1},
    )
    service.create(envelope)
    service.challenge(envelope.artifact_id)
    service.adjudicate(envelope.artifact_id)
    shipped = service.ship(envelope.artifact_id)
    assert shipped.status == ArtifactStatus.SHIPPED


def test_micro_plan_cycle_detection() -> None:
    with pytest.raises(ValueError):
        MicroPlan(
            micro_plan_id="MP-abcdef12",
            parent_task_ref="T-a-b-c-1",
            modules=[
                MicroPlanModule(
                    module_id="MM-a",
                    name="A",
                    purpose="a",
                    io_contract=MicroIoContract(inputs=["x"], outputs=["y"]),
                    error_cases=[],
                    depends_on=["MM-b"],
                    reuse_candidate_refs=[],
                    reuse_decision=ReuseDecision.CREATE_NEW,
                    reuse_justification="new",
                    reuse_search_report=ReuseSearchReport(
                        query="a",
                        candidates_considered=[],
                        conclusion=ReuseConclusion.CREATE_NEW,
                        justification="new",
                    ),
                ),
                MicroPlanModule(
                    module_id="MM-b",
                    name="B",
                    purpose="b",
                    io_contract=MicroIoContract(inputs=["x"], outputs=["y"]),
                    error_cases=[],
                    depends_on=["MM-a"],
                    reuse_candidate_refs=[],
                    reuse_decision=ReuseDecision.CREATE_NEW,
                    reuse_justification="new",
                    reuse_search_report=ReuseSearchReport(
                        query="b",
                        candidates_considered=[],
                        conclusion=ReuseConclusion.CREATE_NEW,
                        justification="new",
                    ),
                ),
            ],
        )


def test_contract_fingerprint_ignores_metadata_changes() -> None:
    base = dict(
        module_id="MM-1",
        module_version="1.0.0",
        name="mod",
        purpose="purpose",
        tags=["a"],
        examples_ref="/modules/MM-1/1.0.0/examples.md",
        created_by="tester",
        inputs=[ContractInput(name="x", type="str", constraints=["non-empty"])],
        outputs=[ContractOutput(name="y", type="str", invariants=["deterministic"])],
        error_surfaces=[ContractErrorSurface(name="ValidationError", when="bad input", surface="exception")],
        effects=[{"type": EffectType.WRITE, "target": "/tmp", "description": "writes temp"}],
        modes=ContractModes(sync=True, **{"async": False}, notes="sync"),
        error_cases=["bad input"],
        dependencies=[],
        compatibility=CompatibilityRule(backward_compatible_with=[], breaking_change_policy="major"),
        runtime_budgets=RuntimeBudgets(latency_ms_p95=10, memory_mb_max=32),
        status=ArtifactStatus.DRAFT,
    )
    c1 = MicroModuleContract(**base)
    c2 = MicroModuleContract(**{**base, "name": "renamed", "purpose": "different docs", "tags": ["b"]})
    assert c1.interface_fingerprint == c2.interface_fingerprint


def test_code_index_append_only_search_and_reindex(tmp_path: Path) -> None:
    index = CodeIndex(tmp_path)
    contract = MicroModuleContract(
        module_id="MM-search",
        module_version="1.0.0",
        name="Search Module",
        purpose="Searches records by query",
        tags=["search", "query"],
        examples_ref="/modules/MM-search/1.0.0/examples.md",
        created_by="tester",
        inputs=[ContractInput(name="query", type="str", constraints=["non-empty"])],
        outputs=[ContractOutput(name="items", type="list", invariants=["sorted"])],
        error_surfaces=[ContractErrorSurface(name="ValidationError", when="empty query", surface="exception")],
        effects=[{"type": EffectType.CALL, "target": "db", "description": "reads index"}],
        modes=ContractModes(sync=True, **{"async": False}, notes="sync"),
        error_cases=["empty query"],
        dependencies=[],
        compatibility=CompatibilityRule(backward_compatible_with=[], breaking_change_policy="major"),
        runtime_budgets=RuntimeBudgets(latency_ms_p95=20, memory_mb_max=64),
        status=ArtifactStatus.DRAFT,
    )
    index.register(contract)
    results = index.search("module for record query", top_k=5)
    assert results
    assert results[0].entry.module_id == "MM-search"
    assert index.reindex() >= 1


def test_code_index_records_level_owner_and_compatibility(tmp_path: Path) -> None:
    index = CodeIndex(tmp_path)
    contract = MicroModuleContract(
        module_id="MM-owner",
        module_version="1.0.0",
        name="Owner Module",
        purpose="Tracks level-aware metadata",
        tags=["ownership"],
        owner="team-platform",
        compatibility_type=ContractCompatibility.NON_BREAKING,
        examples_ref="/modules/MM-owner/1.0.0/examples.md",
        created_by="tester",
        inputs=[ContractInput(name="request", type="dict", constraints=["non-empty"])],
        outputs=[ContractOutput(name="response", type="dict", invariants=["deterministic"])],
        error_surfaces=[ContractErrorSurface(name="ValidationError", when="bad request", surface="exception")],
        effects=[{"type": EffectType.WRITE, "target": "state_store", "description": "writes artifacts"}],
        modes=ContractModes(sync=True, **{"async": False}, notes="sync"),
        error_cases=["bad request"],
        dependencies=[],
        compatibility=CompatibilityRule(backward_compatible_with=[], breaking_change_policy="major"),
        runtime_budgets=RuntimeBudgets(latency_ms_p95=15, memory_mb_max=64),
        status=ArtifactStatus.DRAFT,
    )
    index.register(contract)
    entry = index.get_entry("MM-owner@1.0.0")
    assert entry is not None
    assert entry.level == BoundaryLevel.L4_COMPONENT
    assert entry.owner == "team-platform"
    assert entry.compatibility_type == ContractCompatibility.NON_BREAKING


def test_code_index_reindexes_when_embedding_model_changes(tmp_path: Path) -> None:
    index = CodeIndex(tmp_path, embedding_model="deterministic-hash-96")
    contract = MicroModuleContract(
        module_id="MM-reindex",
        module_version="1.0.0",
        name="Reindex Module",
        purpose="Validate and normalize request boundary for stock ranking",
        tags=["validation", "stocks"],
        examples_ref="/modules/MM-reindex/1.0.0/examples.md",
        created_by="tester",
        inputs=[ContractInput(name="request", type="dict", constraints=["non-empty"])],
        outputs=[ContractOutput(name="response", type="dict", invariants=["deterministic"])],
        error_surfaces=[ContractErrorSurface(name="ValidationError", when="malformed request", surface="exception")],
        effects=[{"type": EffectType.WRITE, "target": "state_store", "description": "writes artifacts"}],
        modes=ContractModes(sync=True, **{"async": False}, notes="sync"),
        error_cases=["malformed request"],
        dependencies=[],
        compatibility=CompatibilityRule(backward_compatible_with=[], breaking_change_policy="major"),
        runtime_budgets=RuntimeBudgets(latency_ms_p95=15, memory_mb_max=32),
        status=ArtifactStatus.DRAFT,
    )
    index.register(contract)

    CodeIndex(tmp_path, embedding_model="deterministic-hash-64")
    assert (tmp_path / "embedding_model.txt").read_text(encoding="utf-8").strip() == "deterministic-hash-64"
    events = [
        line.strip()
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any("embedding_model_changed" in line for line in events)


def test_interface_change_exception_requires_non_empty_delta() -> None:
    with pytest.raises(ValueError):
        InterfaceChangeException(
            exception_id="ICE-deadbeef",
            type=InterfaceChangeType.INCOMPLETE,
            raised_by=RaisedBy(artifact_ref="prop", role="proposer", run_id="run-1"),
            target_module="MM-1@1.0.0",
            reason="need extra output",
            evidence=["consumer requires field x"],
            proposed_contract_delta=ProposedContractDelta(),
            compatibility_expectation=CompatibilityExpectation.BACKWARD_COMPATIBLE,
            urgency=Urgency.HIGH,
        )


def test_build_project_state_assigns_declaration_order() -> None:
    spec = parse_raw_spec_to_product_spec("# Product\n## A\n## B\n## C")
    apply_canonical_task_ids(spec)
    state = build_project_state(spec)
    orders = [entry.declaration_order for entry in state.tasks.values()]
    assert sorted(orders) == list(range(len(orders)))


def test_runtime_settings_enable_llm_debate_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FACTORY_DEBATE_USE_LLM", raising=False)
    settings = RuntimeSettings.from_env()
    assert settings.debate_use_llm is True


def test_runtime_settings_default_embedding_model_is_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FACTORY_EMBEDDING_MODEL", raising=False)
    settings = RuntimeSettings.from_env()
    assert settings.embedding_model == "text-embedding-3-large"


def test_runtime_settings_default_class_contract_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FACTORY_CLASS_CONTRACT_POLICY", raising=False)
    settings = RuntimeSettings.from_env()
    assert settings.class_contract_policy == "selective_shared"


def test_normalize_structured_output_accepts_include_raw_shape() -> None:
    proposal = Proposal(
        proposal_id="PROP-deadbeef",
        target_artifact_id="CTR-deadbeef",
        claim="Implement module",
        deliverable_ref="/modules/MM-1/1.0.0/implementation/",
        acceptance_checks=["returns deterministic outputs"],
    )
    normalized = normalize_structured_output(
        raw_output={"raw": {"id": "x"}, "parsed": proposal, "parsing_error": None},
        schema=Proposal,
    )
    assert normalized == proposal


def test_normalize_structured_output_raises_on_parsing_error() -> None:
    with pytest.raises(RuntimeError, match="Structured output parsing failed for Proposal"):
        normalize_structured_output(
            raw_output={"raw": {"id": "x"}, "parsed": None, "parsing_error": ValueError("boom")},
            schema=Proposal,
        )


def test_get_structured_chat_model_defaults_to_warning_free_function_calling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeStructuredRunnable:
        def invoke(self, prompt: str) -> Proposal:
            _ = prompt
            return Proposal(
                proposal_id="PROP-deadbeef",
                target_artifact_id="CTR-deadbeef",
                claim="Implement module",
                deliverable_ref="/modules/MM-1/1.0.0/implementation/",
                acceptance_checks=["returns deterministic outputs"],
            )

    class _FakeChatModel:
        def with_structured_output(self, schema, *, method, include_raw, strict):  # noqa: ANN001,ANN201
            captured["schema"] = schema
            captured["method"] = method
            captured["include_raw"] = include_raw
            captured["strict"] = strict
            return _FakeStructuredRunnable()

    monkeypatch.setattr("dcode_app_factory.llm.get_chat_model", lambda **kwargs: _FakeChatModel())

    adapter = get_structured_chat_model(model_name="gpt-4o-mini", schema=Proposal)
    proposal = adapter.invoke("return proposal")
    assert proposal.proposal_id == "PROP-deadbeef"
    assert captured["schema"] is Proposal
    assert captured["method"] == "function_calling"
    assert captured["include_raw"] is False
    assert captured["strict"] is True


def test_get_structured_chat_model_rejects_strict_json_mode() -> None:
    with pytest.raises(ValueError, match="strict=True is not valid for method='json_mode'"):
        get_structured_chat_model(model_name="gpt-4o-mini", schema=Proposal, method="json_mode", strict=True)


def test_debate_graph_llm_path_uses_function_calling_structured_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, object]] = []

    class _FakeStructuredAdapter:
        def __init__(self, schema: object) -> None:
            self.schema = schema

        def invoke(self, prompt: str):  # noqa: ANN201
            _ = prompt
            if self.schema is Proposal:
                return Proposal(
                    proposal_id="PROP-deadbeef",
                    target_artifact_id="CTR-deadbeef",
                    claim="Implement module",
                    deliverable_ref="/modules/MM-1/1.0.0/implementation/",
                    acceptance_checks=["returns deterministic outputs"],
                )
            if self.schema is Challenge:
                return Challenge(
                    challenge_id="CHAL-deadbeef",
                    target_artifact_id="CTR-deadbeef",
                    verdict="PASS",
                    failures=[],
                    rubric_assessments=[
                        RubricAssessment(criterion="R1", assessment="MET", evidence="verified"),
                        RubricAssessment(criterion="R2", assessment="MET", evidence="verified"),
                        RubricAssessment(criterion="R3", assessment="MET", evidence="verified"),
                        RubricAssessment(criterion="R4", assessment="MET", evidence="verified"),
                        RubricAssessment(criterion="R5", assessment="MET", evidence="verified"),
                        RubricAssessment(criterion="R6", assessment="MET", evidence="verified"),
                    ],
                )
            return Adjudication(
                adjudication_id="ADJ-deadbeef",
                target_artifact_id="CTR-deadbeef",
                decision=AdjudicationDecision.APPROVE_WITH_AMENDMENTS,
                amendments=[],
                rationale="All checks met",
                ship_directive=ShipDirective.SHIP,
            )

    def _fake_structured_adapter(self, *, role: str, schema):  # noqa: ANN001,ANN201
        _ = self
        calls.append((role, schema))
        return _FakeStructuredAdapter(schema)

    monkeypatch.setattr("dcode_app_factory.agent_runtime.RoleAgentRuntime.structured_adapter", _fake_structured_adapter)

    store = FactoryStateStore(tmp_path)
    settings = RuntimeSettings(state_store_root=str(tmp_path), project_id="ACME-DEBATE-1", embedding_model="deterministic-hash-384")
    role_runtime = RoleAgentRuntime(stage="engineering_loop", settings=settings, backend_root=store.root)
    debate = DebateGraph(store=store, role_runtime=role_runtime, use_llm=True)
    result = debate.run(
        task_id="T-demo-demo-demo-1",
        module_id="MM-demo-1",
        target_artifact_id="CTR-deadbeef",
        contract_summary={},
        context_summary="reuse_decision=CREATE_NEW; reuse_justification=no index match",
        context_pack_refs={},
        max_retries=1,
    )

    assert result.passed is True
    assert calls == [
        ("proposer", Proposal),
        ("challenger", Challenge),
        ("arbiter", Adjudication),
    ]


def test_project_scoped_root_namespaces_by_project_id(tmp_path: Path) -> None:
    scoped = project_scoped_root(tmp_path, "ACME Project/2026")
    assert scoped == tmp_path / "projects" / "ACME-Project-2026"


def test_debate_graph_halt_returns_without_parent_command(tmp_path: Path) -> None:
    class _AlwaysFailDebate(DebateGraph):
        def _challenge(self, state):  # type: ignore[override]
            proposal = state["proposal"]
            return {
                "challenge": Challenge(
                    challenge_id="CHAL-deadbeef",
                    target_artifact_id=proposal.target_artifact_id,
                    verdict="FAIL",
                    failures=[
                        ChallengeFailure(
                            invariant="contract compliance",
                            evidence="detected mismatch",
                            required_change="fix output contract",
                        )
                    ],
                    rubric_assessments=[
                        RubricAssessment(criterion="R1", assessment="NOT_MET", evidence="failing contract evidence"),
                        RubricAssessment(criterion="R2", assessment="MET", evidence="verified"),
                        RubricAssessment(criterion="R3", assessment="MET", evidence="verified"),
                        RubricAssessment(criterion="R4", assessment="MET", evidence="verified"),
                        RubricAssessment(criterion="R5", assessment="MET", evidence="verified"),
                        RubricAssessment(criterion="R6", assessment="MET", evidence="verified"),
                    ],
                )
            }

        def _adjudicate(self, state):  # type: ignore[override]
            proposal = state["proposal"]
            return {
                "adjudication": Adjudication(
                    adjudication_id="ADJ-deadbeef",
                    target_artifact_id=proposal.target_artifact_id,
                    decision=AdjudicationDecision.REJECT,
                    amendments=[],
                    rationale="Challenge demonstrated a material violation",
                    ship_directive=ShipDirective.NO_SHIP,
                )
            }

    store = FactoryStateStore(tmp_path)
    settings = RuntimeSettings(state_store_root=str(tmp_path), project_id="ACME-DEBATE-2", embedding_model="deterministic-hash-384")
    role_runtime = RoleAgentRuntime(stage="engineering_loop", settings=settings, backend_root=store.root)
    debate = _AlwaysFailDebate(store=store, role_runtime=role_runtime, use_llm=True)
    result = debate.run(
        task_id="T-demo-1",
        module_id="MM-demo-1",
        target_artifact_id="CTR-deadbeef",
        contract_summary={},
        context_summary="demo",
        context_pack_refs={},
        max_retries=2,
    )
    assert result.passed is False
    assert result.retries_used == 2


def test_project_loop_happy_path_with_mocked_debate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("dcode_app_factory.loops.ProductLoop._invoke_deep_agent", lambda self, spec_json_path: None)
    monkeypatch.setattr("dcode_app_factory.loops.DebateGraph", _FakeDebateGraphPass)

    settings = RuntimeSettings(
        state_store_root=str(tmp_path),
        project_id="ACME-001",
        embedding_model="deterministic-hash-384",
    )
    spec = ProductLoop("# Product\n## A\n## B", state_store_root=tmp_path, settings=settings).run()
    loop = ProjectLoop(
        spec,
        CodeIndex(project_scoped_root(tmp_path, settings.project_id) / "code_index"),
        state_store_root=tmp_path,
        settings=settings,
        enable_interrupts=False,
    )
    assert loop.run() is True

    state = loop.state_store.read_project_state()
    assert all(entry.status == TaskStatus.SHIPPED for entry in state.tasks.values())
    assert loop.state_store.root == project_scoped_root(tmp_path, settings.project_id)
    assert state.project_id == settings.project_id


def test_project_loop_emits_levelled_contract_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("dcode_app_factory.loops.ProductLoop._invoke_deep_agent", lambda self, spec_json_path: None)
    monkeypatch.setattr("dcode_app_factory.loops.DebateGraph", _FakeDebateGraphPass)

    settings = RuntimeSettings(
        state_store_root=str(tmp_path),
        project_id="ACME-LVL-001",
        embedding_model="deterministic-hash-384",
        class_contract_policy="selective_shared",
    )
    spec = ProductLoop("# Product\n## A\n## B", state_store_root=tmp_path, settings=settings).run()
    project = ProjectLoop(
        spec,
        CodeIndex(project_scoped_root(tmp_path, settings.project_id) / "code_index"),
        state_store_root=tmp_path,
        settings=settings,
        enable_interrupts=False,
    )
    assert project.run() is True

    assert any(project.state_store.system_contracts_dir.rglob("contract.json"))
    assert any(project.state_store.service_contracts_dir.rglob("contract.json"))
    assert any(project.state_store.class_contracts_dir.rglob("contract.json"))

    state = project.state_store.read_project_state()
    shipped = [entry for entry in state.tasks.values() if entry.status == TaskStatus.SHIPPED]
    assert shipped
    assert all(entry.service_refs for entry in shipped)
    assert all(entry.component_refs for entry in shipped)


def test_project_loop_halt_blocks_downstream(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("dcode_app_factory.loops.ProductLoop._invoke_deep_agent", lambda self, spec_json_path: None)
    monkeypatch.setattr("dcode_app_factory.loops.DebateGraph", _FakeDebateGraphFailFirst)

    settings = RuntimeSettings(
        state_store_root=str(tmp_path),
        project_id="ACME-002",
        embedding_model="deterministic-hash-384",
    )
    spec = ProductLoop("# Product\n## A\n## B", state_store_root=tmp_path, settings=settings).run()
    loop = ProjectLoop(
        spec,
        CodeIndex(project_scoped_root(tmp_path, settings.project_id) / "code_index"),
        state_store_root=tmp_path,
        settings=settings,
        enable_interrupts=False,
    )
    assert loop.run() is False

    state = loop.state_store.read_project_state()
    statuses = {task_id: entry.status for task_id, entry in state.tasks.items()}
    assert TaskStatus.HALTED in statuses.values()
    assert TaskStatus.BLOCKED in statuses.values()


def test_release_loop_flags_deprecated_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("dcode_app_factory.loops.ProductLoop._invoke_deep_agent", lambda self, spec_json_path: None)
    monkeypatch.setattr("dcode_app_factory.loops.DebateGraph", _FakeDebateGraphPass)

    settings = RuntimeSettings(
        state_store_root=str(tmp_path),
        project_id="ACME-003",
        embedding_model="deterministic-hash-384",
    )
    spec = ProductLoop("# Product\n## A\n## B", state_store_root=tmp_path, settings=settings).run()
    code_index = CodeIndex(project_scoped_root(tmp_path, settings.project_id) / "code_index")
    project = ProjectLoop(spec, code_index, state_store_root=tmp_path, settings=settings, enable_interrupts=False)
    assert project.run() is True

    entries = code_index.list_entries()
    assert entries
    code_index.set_status(entries[0].module_ref, status=CodeIndexStatus.DEPRECATED, deprecation_reason="replacement available")

    release = ReleaseLoop(state_store=project.state_store, code_index=code_index, spec=spec)
    result = release.run()
    assert result["integration_gates"]["deprecation_check"] == "FAIL"


def test_release_loop_includes_new_contract_and_context_gates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("dcode_app_factory.loops.ProductLoop._invoke_deep_agent", lambda self, spec_json_path: None)
    monkeypatch.setattr("dcode_app_factory.loops.DebateGraph", _FakeDebateGraphPass)

    settings = RuntimeSettings(
        state_store_root=str(tmp_path),
        project_id="ACME-REL-GATES",
        embedding_model="deterministic-hash-384",
    )
    spec = ProductLoop("# Product\n## A\n## B", state_store_root=tmp_path, settings=settings).run()
    code_index = CodeIndex(project_scoped_root(tmp_path, settings.project_id) / "code_index")
    project = ProjectLoop(spec, code_index, state_store_root=tmp_path, settings=settings, enable_interrupts=False)
    assert project.run() is True

    release = ReleaseLoop(state_store=project.state_store, code_index=code_index, spec=spec)
    result = release.run()
    gates = result["integration_gates"]
    assert "contract_completeness_check" in gates
    assert "compatibility_check" in gates
    assert "ownership_check" in gates
    assert "context_pack_compliance_check" in gates
    assert gates["contract_completeness_check"] == "PASS"
    assert gates["compatibility_check"] == "PASS"
    assert gates["ownership_check"] == "PASS"
    assert gates["context_pack_compliance_check"] == "PASS"


def test_release_loop_expands_transitive_dependencies(tmp_path: Path) -> None:
    settings = RuntimeSettings(
        state_store_root=str(tmp_path),
        project_id="ACME-REL-CLOSURE",
        embedding_model="deterministic-hash-384",
    )
    state_store = FactoryStateStore(tmp_path, project_id=settings.project_id)
    code_index = CodeIndex(state_store.code_index_dir, embedding_model=settings.embedding_model)

    spec = parse_raw_spec_to_product_spec("# Product\n## A")
    apply_canonical_task_ids(spec)
    project_state = build_project_state(spec, project_id=settings.project_id)
    task_id = next(iter(project_state.tasks.keys()))
    project_state.tasks[task_id].status = TaskStatus.SHIPPED
    project_state.tasks[task_id].module_ref = "MM-a@1.0.1"
    project_state.tasks[task_id].module_refs = ["MM-a@1.0.1", "MM-b@1.0.0"]
    project_state.tasks[task_id].shipped_at = datetime.now(UTC)
    state_store.write_project_state(project_state)

    base_contract = dict(
        name="module",
        purpose="purpose",
        tags=["factory"],
        created_by="tester",
        inputs=[ContractInput(name="x", type="str", constraints=["non-empty"])],
        outputs=[ContractOutput(name="y", type="str", invariants=["deterministic"])],
        error_surfaces=[ContractErrorSurface(name="ValidationError", when="bad input", surface="exception")],
        effects=[{"type": EffectType.WRITE, "target": "/tmp", "description": "writes temp"}],
        modes=ContractModes(sync=True, **{"async": False}, notes="sync"),
        error_cases=["bad input"],
        compatibility=CompatibilityRule(backward_compatible_with=[], breaking_change_policy="major"),
        runtime_budgets=RuntimeBudgets(latency_ms_p95=10, memory_mb_max=32),
        status=ArtifactStatus.DRAFT,
    )

    code_index.register(
        MicroModuleContract(
            module_id="MM-a",
            module_version="1.0.0",
            examples_ref="/modules/MM-a/1.0.0/examples.md",
            dependencies=[],
            **base_contract,
        )
    )
    code_index.register(
        MicroModuleContract(
            module_id="MM-a",
            module_version="1.0.1",
            examples_ref="/modules/MM-a/1.0.1/examples.md",
            dependencies=[],
            **base_contract,
        )
    )
    code_index.register(
        MicroModuleContract(
            module_id="MM-b",
            module_version="1.0.0",
            examples_ref="/modules/MM-b/1.0.0/examples.md",
            dependencies=[{"ref": "MM-a@1.0.0", "why": "module dependency"}],
            **base_contract,
        )
    )

    release = ReleaseLoop(state_store=state_store, code_index=code_index, spec=spec)
    result = release.run()
    assert "MM-a@1.0.0" in result["modules"]
    assert result["integration_gates"]["dependency_check"] == "PASS"
    assert result["integration_gates"]["fingerprint_check"] == "PASS"


def test_outer_graph_runs_auto_approve_with_mocked_debate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("dcode_app_factory.loops.ProductLoop._invoke_deep_agent", lambda self, spec_json_path: None)
    monkeypatch.setattr("dcode_app_factory.loops.DebateGraph", _FakeDebateGraphPass)

    settings = RuntimeSettings(
        state_store_root=str(tmp_path),
        project_id="ACME-004",
        embedding_model="deterministic-hash-384",
    )
    orchestrator = FactoryOrchestrator(raw_spec="# Product\n## A\n## B", state_store_root=tmp_path, settings=settings)
    result = orchestrator.run(approval_action="APPROVE")
    assert result["project_success"] is True
    assert "release_result" in result


def test_checkpoint_files_created(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("dcode_app_factory.loops.ProductLoop._invoke_deep_agent", lambda self, spec_json_path: None)
    monkeypatch.setattr("dcode_app_factory.loops.DebateGraph", _FakeDebateGraphPass)

    settings = RuntimeSettings(
        state_store_root=str(tmp_path),
        project_id="ACME-005",
        embedding_model="deterministic-hash-384",
    )
    orchestrator = FactoryOrchestrator(raw_spec="# Product\n## A", state_store_root=tmp_path, settings=settings)
    orchestrator.run(approval_action="APPROVE")

    project_root = project_scoped_root(tmp_path, settings.project_id)
    assert (project_root / "checkpoints" / "outer_graph.sqlite").is_file()
    assert (project_root / "checkpoints" / "project_loop.sqlite").is_file()


def test_micro_plan_decomposes_into_atomic_modules(tmp_path: Path) -> None:
    settings = RuntimeSettings(
        state_store_root=str(tmp_path),
        project_id="ACME-ATOMIC",
        embedding_model="deterministic-hash-384",
    )
    state_store = FactoryStateStore(tmp_path, project_id=settings.project_id)
    loop = EngineeringLoop(
        code_index=CodeIndex(state_store.code_index_dir, embedding_model=settings.embedding_model),
        state_store=state_store,
        settings=settings,
    )

    spec = parse_raw_spec_to_product_spec("# Product\n## Build trader API ranking website")
    apply_canonical_task_ids(spec)
    task = spec.iter_tasks()[0]
    plan = loop._build_micro_plan(task)

    assert len(plan.modules) >= 4
    assert any(module.depends_on for module in plan.modules[1:])
    assert all(module.reuse_search_report.query for module in plan.modules)


def test_micro_plan_can_select_reuse_candidate(tmp_path: Path) -> None:
    settings = RuntimeSettings(
        state_store_root=str(tmp_path),
        project_id="ACME-REUSE",
        embedding_model="deterministic-hash-384",
    )
    state_store = FactoryStateStore(tmp_path, project_id=settings.project_id)
    index = CodeIndex(state_store.code_index_dir, embedding_model=settings.embedding_model)
    existing = MicroModuleContract(
        module_id="MM-existing",
        module_version="1.0.0",
        name="Input Boundary Validator",
        purpose="Validate and normalize request boundary for Build trader API ranking website",
        tags=["validate", "normalize", "boundary"],
        examples_ref="/modules/MM-existing/1.0.0/examples.md",
        created_by="tester",
        inputs=[ContractInput(name="request", type="dict", constraints=["schema-valid"])],
        outputs=[ContractOutput(name="validated", type="dict", invariants=["normalized"])],
        error_surfaces=[ContractErrorSurface(name="ValidationError", when="bad payload", surface="exception")],
        effects=[{"type": EffectType.WRITE, "target": "state_store", "description": "writes validation traces"}],
        modes=ContractModes(sync=True, **{"async": False}, notes="sync"),
        error_cases=["bad payload"],
        dependencies=[],
        compatibility=CompatibilityRule(backward_compatible_with=[], breaking_change_policy="major"),
        runtime_budgets=RuntimeBudgets(latency_ms_p95=12, memory_mb_max=24),
        status=ArtifactStatus.DRAFT,
    )
    index.register(existing)

    loop = EngineeringLoop(
        code_index=index,
        state_store=state_store,
        settings=settings,
    )
    spec = parse_raw_spec_to_product_spec("# Product\n## Build trader API ranking website")
    apply_canonical_task_ids(spec)
    task = spec.iter_tasks()[0]
    plan = loop._build_micro_plan(task)

    assert any(module.reuse_decision == ReuseDecision.REUSE for module in plan.modules)


def test_engineering_loop_resolves_dependency_refs_and_versions_examples_path(tmp_path: Path) -> None:
    settings = RuntimeSettings(
        state_store_root=str(tmp_path),
        project_id="ACME-DEP-REFS",
        embedding_model="deterministic-hash-384",
    )
    state_store = FactoryStateStore(tmp_path, project_id=settings.project_id)
    loop = EngineeringLoop(
        code_index=CodeIndex(state_store.code_index_dir, embedding_model=settings.embedding_model),
        state_store=state_store,
        settings=settings,
    )

    spec = parse_raw_spec_to_product_spec("# Product\n## Build trader API ranking website")
    apply_canonical_task_ids(spec)
    task = spec.iter_tasks()[0]
    module = MicroPlanModule(
        module_id="MM-dep-target",
        name="Dependency target [core-01]",
        purpose="Deterministic core logic",
        io_contract=MicroIoContract(inputs=["validated"], outputs=["computed"]),
        error_cases=["bad input"],
        depends_on=["MM-dep-source"],
        reuse_candidate_refs=[],
        reuse_decision=ReuseDecision.CREATE_NEW,
        reuse_justification="no reusable candidate",
        reuse_search_report=ReuseSearchReport(
            query="dep-target",
            candidates_considered=[
                ReuseSearchCandidate(module_ref="MM-dep-source@1.0.0", why_rejected="similarity below threshold")
            ],
            conclusion=ReuseConclusion.CREATE_NEW,
            justification="no reusable candidate",
        ),
    )

    with pytest.raises(ValueError, match="Missing resolved dependency ref"):
        loop._resolve_dependency_refs(module, {})

    dependency_refs = loop._resolve_dependency_refs(module, {"MM-dep-source": "MM-dep-source@2.4.6"})
    contract = loop._build_contract(
        task,
        module,
        "CP-deadbeef",
        dependency_refs,
    )

    assert dependency_refs == ["MM-dep-source@2.4.6"]
    assert [entry.ref for entry in contract.dependencies] == ["MM-dep-source@2.4.6"]
    assert contract.examples_ref.endswith(f"/{contract.module_version}/examples.md")
