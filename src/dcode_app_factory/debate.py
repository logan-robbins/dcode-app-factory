from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .agent_runtime import RoleAgentRuntime
from .llm import StructuredOutputAdapter
from .models import (
    Adjudication,
    AdjudicationDecision,
    Challenge,
    DebateVerdict,
    Proposal,
    ShipDirective,
)
from .state_store import FactoryStateStore


class DebateState(TypedDict, total=False):
    task_id: str
    module_id: str
    target_artifact_id: str
    contract_summary: dict[str, Any]
    context_summary: str
    context_pack_refs: dict[str, str]
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
        role_runtime: RoleAgentRuntime,
        propagate_parent_halt: bool = False,
    ) -> None:
        self.store = store
        self.role_runtime = role_runtime
        self.propagate_parent_halt = propagate_parent_halt
        self.role_runtime.require_roles(["proposer", "challenger", "arbiter"])
        self._proposer: StructuredOutputAdapter[Proposal] = self.role_runtime.structured_adapter(
            role="proposer",
            schema=Proposal,
        )
        self._challenger: StructuredOutputAdapter[Challenge] = self.role_runtime.structured_adapter(
            role="challenger",
            schema=Challenge,
        )
        self._arbiter: StructuredOutputAdapter[Adjudication] = self.role_runtime.structured_adapter(
            role="arbiter",
            schema=Adjudication,
        )
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

    def _context_pack_payload(self, state: DebateState, role: str) -> dict[str, Any]:
        refs = state.get("context_pack_refs", {})
        if not refs:
            return {}
        cp_id = refs.get(role)
        if not cp_id:
            return {}
        context_pack = self.store.read_context_pack(cp_id)
        return context_pack.model_dump(mode="json")

    def _propose(self, state: DebateState) -> dict[str, Any]:
        module_id = state["module_id"]
        artifact_id = state["target_artifact_id"]
        proposal = self._proposer.invoke(
            (
                "You are the proposer role for production engineering debate. "
                f"{self.role_runtime.role_context_line('proposer')} "
                "Return Proposal JSON only. Acceptance checks must be objective and executable. "
                "Do not propose mock/stub/fake/placeholder deliverables.\n"
                f"TaskID={state['task_id']}\n"
                f"ModuleID={module_id}\n"
                f"TargetArtifactID={artifact_id}\n"
                f"ContractSummary={state.get('contract_summary', {})}\n"
                f"ContextSummary={state.get('context_summary', '')}\n"
                f"ContextPack={self._context_pack_payload(state, 'proposer')}"
            )
        )
        if proposal.target_artifact_id != artifact_id:
            raise ValueError(
                f"proposer returned target artifact mismatch: expected={artifact_id} got={proposal.target_artifact_id}"
            )
        return {"proposal": proposal}

    def _challenge(self, state: DebateState) -> dict[str, Any]:
        proposal = state["proposal"]
        challenge = self._challenger.invoke(
            (
                "You are the challenger role for adversarial contract evaluation. "
                f"{self.role_runtime.role_context_line('challenger')} "
                "Return Challenge JSON only and enforce rubric R1..R6. "
                "Reject any proposal that relies on mock/stub/fake/placeholder behavior.\n"
                f"Proposal={proposal.model_dump_json()}\n"
                f"ContractSummary={state.get('contract_summary', {})}\n"
                f"ContextSummary={state.get('context_summary', '')}\n"
                f"ContextPack={self._context_pack_payload(state, 'challenger')}"
            )
        )
        if challenge.target_artifact_id != proposal.target_artifact_id:
            raise ValueError(
                "challenger returned target artifact mismatch: "
                f"expected={proposal.target_artifact_id} got={challenge.target_artifact_id}"
            )
        if challenge.verdict == DebateVerdict.FAIL and not challenge.failures:
            raise ValueError("challenger returned FAIL verdict without concrete failures")
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
        adjudication = self._arbiter.invoke(
            (
                "You are the arbiter role and must decide APPROVE, APPROVE_WITH_AMENDMENTS, or REJECT. "
                f"{self.role_runtime.role_context_line('arbiter')} "
                "Return Adjudication JSON only. Do not approve proposals with mock/stub/fake/placeholder behavior.\n"
                f"Proposal={proposal.model_dump_json()}\n"
                f"Challenge={challenge.model_dump_json()}\n"
                f"ContractSummary={state.get('contract_summary', {})}\n"
                f"ContextSummary={state.get('context_summary', '')}\n"
                f"ContextPack={self._context_pack_payload(state, 'arbiter')}"
            )
        )
        if adjudication.target_artifact_id != proposal.target_artifact_id:
            raise ValueError(
                "arbiter returned target artifact mismatch: "
                f"expected={proposal.target_artifact_id} got={adjudication.target_artifact_id}"
            )
        if adjudication.decision == AdjudicationDecision.REJECT and adjudication.ship_directive != ShipDirective.NO_SHIP:
            raise ValueError("arbiter returned REJECT with ship_directive != NO_SHIP")
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

    def run(
        self,
        *,
        task_id: str,
        module_id: str,
        target_artifact_id: str,
        contract_summary: dict[str, Any],
        context_summary: str,
        context_pack_refs: dict[str, str] | None = None,
        max_retries: int = 2,
    ) -> DebateResult:
        result = self.graph.invoke(
            {
                "task_id": task_id,
                "module_id": module_id,
                "target_artifact_id": target_artifact_id,
                "contract_summary": contract_summary,
                "context_summary": context_summary,
                "context_pack_refs": context_pack_refs or {},
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
