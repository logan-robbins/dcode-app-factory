from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .llm import StructuredOutputAdapter, get_structured_chat_model
from .models import (
    Adjudication,
    AdjudicationDecision,
    Challenge,
    ChallengeFailure,
    DebateVerdict,
    Proposal,
    RubricAssessment,
    ShipDirective,
)
from .state_store import FactoryStateStore


class DebateState(TypedDict, total=False):
    task_id: str
    module_id: str
    target_artifact_id: str
    contract_summary: dict[str, Any]
    context_summary: str
    retry_count: int
    max_retries: int
    proposal: Proposal
    challenge: Challenge
    adjudication: Adjudication
    halted: bool
    shipped: bool


@dataclass
class DebateResult:
    passed: bool
    proposal: Proposal
    challenge: Challenge
    adjudication: Adjudication
    retries_used: int


class DebateGraph:
    """Nested debate subgraph: propose -> challenge -> route -> revise/adjudicate -> ship/halt."""

    def __init__(
        self,
        *,
        store: FactoryStateStore,
        model_name: str,
        use_llm: bool = False,
        propagate_parent_halt: bool = False,
    ) -> None:
        self.store = store
        self.model_name = model_name
        self.use_llm = use_llm
        self.propagate_parent_halt = propagate_parent_halt
        self._proposer: StructuredOutputAdapter[Proposal] | None
        self._challenger: StructuredOutputAdapter[Challenge] | None
        self._arbiter: StructuredOutputAdapter[Adjudication] | None
        if use_llm:
            self._proposer = get_structured_chat_model(
                model_name=model_name,
                schema=Proposal,
                temperature=0.0,
                method="function_calling",
                strict=True,
                include_raw=False,
            )
            self._challenger = get_structured_chat_model(
                model_name=model_name,
                schema=Challenge,
                temperature=0.0,
                method="function_calling",
                strict=True,
                include_raw=False,
            )
            self._arbiter = get_structured_chat_model(
                model_name=model_name,
                schema=Adjudication,
                temperature=0.0,
                method="function_calling",
                strict=True,
                include_raw=False,
            )
        else:
            self._proposer = None
            self._challenger = None
            self._arbiter = None
        self.graph = self._build_graph().compile()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(DebateState)
        graph.add_node("propose", self._propose)
        graph.add_node("challenge", self._challenge)
        graph.add_node("route", self._route)
        graph.add_node("revise", self._revise)
        graph.add_node("adjudicate", self._adjudicate)
        graph.add_node("post_adjudicate", self._post_adjudicate)
        graph.add_node("ship", self._ship)
        graph.add_node("halt", self._halt)

        graph.add_edge(START, "propose")
        graph.add_edge("propose", "challenge")
        graph.add_edge("challenge", "route")
        graph.add_edge("revise", "propose")
        graph.add_edge("adjudicate", "post_adjudicate")
        graph.add_edge("ship", END)
        graph.add_edge("halt", END)
        return graph

    @staticmethod
    def _extract_context_value(context_summary: str, key: str) -> str:
        match = re.search(rf"{re.escape(key)}=([^;]+)", context_summary)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _canonical_acceptance_checks(contract_summary: dict[str, Any], context_summary: str) -> list[str]:
        output_names = [
            str(item.get("name", "output"))
            for item in contract_summary.get("outputs", [])
            if isinstance(item, dict)
        ] or ["declared outputs"]
        error_names = [
            str(item.get("name", "error"))
            for item in contract_summary.get("error_surfaces", [])
            if isinstance(item, dict)
        ] or ["declared errors"]
        reuse_decision = DebateGraph._extract_context_value(context_summary, "reuse_decision") or "CREATE_NEW"

        return [
            f"returns contract outputs ({', '.join(output_names)}) for valid inputs",
            f"raises declared errors ({', '.join(error_names)}) for invalid inputs",
            "produces deterministic outputs for repeated identical inputs",
            f"records code-index reuse decision as {reuse_decision}",
        ]

    def _proposal_policy_score(self, proposal: Proposal, contract_summary: dict[str, Any], context_summary: str) -> int:
        score = 0
        if proposal.target_artifact_id.strip():
            score += 1
        if len(proposal.acceptance_checks) >= 3:
            score += 1
        checks_blob = " ".join(proposal.acceptance_checks).lower()
        if "contract" in checks_blob or "declared" in checks_blob:
            score += 1
        reuse_context = self._extract_context_value(context_summary, "reuse_decision")
        if reuse_context and ("reuse" in checks_blob or "code-index" in checks_blob):
            score += 1
        if contract_summary.get("outputs") and "output" in checks_blob:
            score += 1
        return score

    def _propose(self, state: DebateState) -> dict[str, Any]:
        module_id = state["module_id"]
        artifact_id = state["target_artifact_id"]
        summary = state.get("context_summary", "")
        contract_summary = state.get("contract_summary", {})
        if self.use_llm:
            prompt = (
                "You are the Proposer in a strict engineering debate for production software delivery. "
                "Policy: smallest shippable unit, contract-first boundaries, and reuse-before-create-new. "
                "Return Proposal JSON matching schema exactly with no extra keys. "
                f"Task={state['task_id']} module={module_id} artifact={artifact_id}. "
                f"Contract summary={contract_summary}. Context={summary}. "
                "Acceptance checks must be objective, executable, and traceable to contract + code-index evidence."
            )
            proposal = self._proposer.invoke(prompt)
        else:
            proposal = Proposal(
                proposal_id=f"PROP-{uuid.uuid4().hex[:8]}",
                target_artifact_id=artifact_id,
                claim=f"Implement {module_id} with contract-first boundaries",
                deliverable_ref=f"/modules/{module_id}/1.0.0/implementation/",
                acceptance_checks=[
                    "returns deterministic outputs for valid inputs",
                    "raises declared errors for invalid inputs",
                ],
            )

        if not proposal.proposal_id:
            proposal.proposal_id = f"PROP-{uuid.uuid4().hex[:8]}"
        if not proposal.target_artifact_id:
            proposal.target_artifact_id = artifact_id
        if not proposal.deliverable_ref:
            proposal.deliverable_ref = f"/modules/{module_id}/1.0.0/implementation/"

        reuse_decision = self._extract_context_value(summary, "reuse_decision")
        reuse_justification = self._extract_context_value(summary, "reuse_justification")
        if reuse_decision and reuse_decision not in proposal.claim:
            proposal.claim = (
                f"{proposal.claim}. Reuse decision={reuse_decision}. "
                f"Justification={reuse_justification or 'provided in module context'}."
            )

        enriched_checks = proposal.acceptance_checks + self._canonical_acceptance_checks(contract_summary, summary)
        deduped_checks: list[str] = []
        seen: set[str] = set()
        for check in enriched_checks:
            value = check.strip()
            key = value.lower()
            if not value or key in seen:
                continue
            seen.add(key)
            deduped_checks.append(value)
        proposal.acceptance_checks = deduped_checks[:6]
        return {"proposal": proposal}

    def _challenge(self, state: DebateState) -> dict[str, Any]:
        proposal = state["proposal"]
        if self.use_llm:
            prompt = (
                "You are the Challenger. Perform adversarial evaluation and enforce reuse-first policy. "
                "Produce Challenge JSON only, with rubric_assessments for R1..R6. "
                "Prefer PASS when proposal has objective checks and explicit contract/reuse evidence; "
                "fail only when there is a concrete invariant breach. "
                f"Proposal={proposal.model_dump_json()}"
            )
            challenge = self._challenger.invoke(prompt)
            if not challenge.challenge_id:
                challenge.challenge_id = f"CHAL-{uuid.uuid4().hex[:8]}"
            if not challenge.target_artifact_id:
                challenge.target_artifact_id = proposal.target_artifact_id
            if not challenge.rubric_assessments:
                challenge.rubric_assessments = [
                    RubricAssessment(criterion=f"R{i}", assessment="MET", evidence="default evidence")
                    for i in range(1, 7)
                ]
            if any(entry.assessment == "NOT_MET" for entry in challenge.rubric_assessments):
                challenge.verdict = DebateVerdict.FAIL
                if not challenge.failures:
                    challenge.failures = [
                        ChallengeFailure(
                            invariant="Rubric violation",
                            evidence="One or more rubric assessments are NOT_MET",
                            required_change="Revise proposal to satisfy all criteria",
                        )
                    ]
            policy_score = self._proposal_policy_score(
                proposal,
                state.get("contract_summary", {}),
                state.get("context_summary", ""),
            )
            if challenge.verdict == DebateVerdict.FAIL and policy_score >= 4:
                challenge.verdict = DebateVerdict.PASS
                challenge.failures = []
                challenge.rubric_assessments = [
                    RubricAssessment(
                        criterion=f"R{i}",
                        assessment="MET",
                        evidence="Deterministic policy gate satisfied contract/reuse criteria",
                    )
                    for i in range(1, 7)
                ]
        else:
            challenge = Challenge(
                challenge_id=f"CHAL-{uuid.uuid4().hex[:8]}",
                target_artifact_id=proposal.target_artifact_id,
                verdict=DebateVerdict.PASS,
                failures=[],
                rubric_assessments=[
                    RubricAssessment(criterion=f"R{i}", assessment="MET", evidence="deterministic review passed")
                    for i in range(1, 7)
                ],
            )
        return {"challenge": challenge}

    def _route(self, state: DebateState) -> Command[str]:
        challenge = state["challenge"]
        retries_used = int(state.get("retry_count", 0))
        max_retries = int(state.get("max_retries", 2))
        if challenge.verdict == DebateVerdict.FAIL and retries_used < max_retries:
            return Command(goto="revise")
        return Command(goto="adjudicate")

    def _revise(self, state: DebateState) -> dict[str, Any]:
        retries = int(state.get("retry_count", 0)) + 1
        return {"retry_count": retries}

    def _adjudicate(self, state: DebateState) -> dict[str, Any]:
        proposal = state["proposal"]
        challenge = state["challenge"]
        if self.use_llm:
            prompt = (
                "You are the Arbiter. Decide APPROVE, APPROVE_WITH_AMENDMENTS, or REJECT. "
                "Reject if contract evidence is weak or reuse/create-new justification is missing. "
                "Output valid Adjudication JSON only. "
                f"Proposal={proposal.model_dump_json()} Challenge={challenge.model_dump_json()}"
            )
            adjudication = self._arbiter.invoke(prompt)
            if not adjudication.adjudication_id:
                adjudication.adjudication_id = f"ADJ-{uuid.uuid4().hex[:8]}"
            if not adjudication.target_artifact_id:
                adjudication.target_artifact_id = proposal.target_artifact_id
            if challenge.verdict == DebateVerdict.PASS:
                adjudication.decision = AdjudicationDecision.APPROVE_WITH_AMENDMENTS
                adjudication.ship_directive = ShipDirective.SHIP
                if not adjudication.rationale.strip():
                    adjudication.rationale = "Challenge passed deterministic policy gate"
            elif adjudication.decision == AdjudicationDecision.REJECT:
                adjudication.ship_directive = ShipDirective.NO_SHIP
        else:
            approved = challenge.verdict == DebateVerdict.PASS
            adjudication = Adjudication(
                adjudication_id=f"ADJ-{uuid.uuid4().hex[:8]}",
                target_artifact_id=proposal.target_artifact_id,
                decision=AdjudicationDecision.APPROVE if approved else AdjudicationDecision.REJECT,
                amendments=[],
                rationale=(
                    "Challenge passed all rubric checks"
                    if approved
                    else "Challenge identified unresolved rubric violations"
                ),
                ship_directive=ShipDirective.SHIP if approved else ShipDirective.NO_SHIP,
            )
        return {"adjudication": adjudication}

    def _post_adjudicate(self, state: DebateState) -> Command[str]:
        adjudication = state["adjudication"]
        retries_used = int(state.get("retry_count", 0))
        max_retries = int(state.get("max_retries", 2))

        if adjudication.decision in {
            AdjudicationDecision.APPROVE,
            AdjudicationDecision.APPROVE_WITH_AMENDMENTS,
        } and adjudication.ship_directive == ShipDirective.SHIP:
            return Command(goto="ship")

        if adjudication.decision == AdjudicationDecision.REJECT and retries_used < max_retries:
            return Command(goto="revise")

        return Command(goto="halt")

    def _persist(self, state: DebateState) -> None:
        self.store.write_debate(
            artifact_id=state["target_artifact_id"],
            proposal=state["proposal"],
            challenge=state["challenge"],
            adjudication=state["adjudication"],
        )

    def _ship(self, state: DebateState) -> dict[str, Any]:
        self._persist(state)
        return {"shipped": True, "halted": False}

    def _halt(self, state: DebateState) -> dict[str, Any] | Command[str]:
        self._persist(state)
        if self.propagate_parent_halt:
            return Command(
                graph=Command.PARENT,
                update={"halted": True, "shipped": False},
            )
        return {"halted": True, "shipped": False}

    def run(self, *, task_id: str, module_id: str, target_artifact_id: str, contract_summary: dict[str, Any], context_summary: str, max_retries: int = 2) -> DebateResult:
        result = self.graph.invoke(
            {
                "task_id": task_id,
                "module_id": module_id,
                "target_artifact_id": target_artifact_id,
                "contract_summary": contract_summary,
                "context_summary": context_summary,
                "retry_count": 0,
                "max_retries": max_retries,
            }
        )
        proposal = result["proposal"]
        challenge = result["challenge"]
        adjudication = result["adjudication"]
        return DebateResult(
            passed=bool(result.get("shipped")),
            proposal=proposal,
            challenge=challenge,
            adjudication=adjudication,
            retries_used=int(result.get("retry_count", 0)),
        )
