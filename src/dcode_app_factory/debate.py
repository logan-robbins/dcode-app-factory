from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Substantive role prompts -- each role receives a detailed system instruction
# that defines its responsibilities, evaluation criteria, and output contract.
# ---------------------------------------------------------------------------

_PROPOSER_SYSTEM_PROMPT = """\
You are the PROPOSER in a production engineering debate. Your responsibility is to
produce a concrete, shippable proposal for a specific micro-module artifact.

PROPOSAL REQUIREMENTS:
- proposal_id: A unique identifier in the format PROP-<8 hex digits>.
- target_artifact_id: Must exactly match the TargetArtifactID provided below.
- claim: A clear, specific statement of what this proposal delivers. The claim must
  describe the concrete behavior or capability being implemented, not a vague intent.
- deliverable_ref: The filesystem path where the implementation artifact will reside.
  This must be a concrete path under /modules/<module_id>/<version>/.
- acceptance_checks: A list of objective, executable acceptance criteria. Each check
  must be verifiable by automated tooling or deterministic inspection. Avoid subjective
  language such as "should work well" or "is performant." Every check must specify a
  concrete pass/fail condition.

PRODUCTION QUALITY STANDARDS:
- No mocks, stubs, fakes, placeholders, or TODO markers. Every deliverable must be
  a complete, production-grade implementation.
- Acceptance criteria must be deterministic: given identical inputs, the same verdict
  must always be produced.
- Implementation steps must be specific and feasible. Do not propose steps that defer
  decisions or require further design.
- The proposal must be fully consistent with the ContractSummary: all declared inputs,
  outputs, error surfaces, and effects must be accounted for.
- Reference the ContextSummary to ensure the proposal accounts for reuse decisions,
  dependency ordering, and any prior context from the engineering loop.

REVISION CONTEXT:
If a PriorChallengeFailures section is provided below, this is a revision pass. You
MUST address every failure listed. Do not repeat the same proposal that was previously
rejected. Explain in your claim how each prior failure has been resolved.

Return Proposal JSON only. No markdown, no commentary.\
"""

_CHALLENGER_SYSTEM_PROMPT = """\
You are the CHALLENGER in a production engineering debate. Your responsibility is
adversarial evaluation of the proposal against a strict six-rubric framework. You must
be rigorous, specific, and evidence-based in every assessment.

RUBRIC DEFINITIONS -- evaluate the proposal against ALL six criteria:

R1 - FUNCTIONAL COMPLETENESS:
  The proposal covers every input, output, error surface, and effect declared in the
  contract summary. No contract dimension is omitted or only partially addressed. All
  declared modes (sync/async) are handled. Acceptance checks collectively cover the
  full behavioral surface.
  MET: Every contract dimension is addressed with specific acceptance checks.
  NOT_MET: Any contract dimension is missing or only vaguely referenced.

R2 - NO MOCK/STUB/PLACEHOLDER BEHAVIOR:
  The proposal does not rely on mock objects, stub implementations, fake data, placeholder
  logic, TODO markers, or deferred decisions. Every deliverable is a complete, production-
  grade artifact. The deliverable_ref points to a concrete implementation path.
  MET: All deliverables are concrete and production-ready.
  NOT_MET: Any deliverable or acceptance check references mocks, stubs, fakes, or placeholders.

R3 - CONTRACT COMPLIANCE:
  The proposal is fully consistent with the provided contract summary. Input types,
  output invariants, error handling behavior, and side effects match the contract
  specification exactly. No undeclared inputs are consumed and no undeclared outputs
  are produced. The target_artifact_id matches.
  MET: Proposal aligns exactly with the contract surface.
  NOT_MET: Any mismatch between proposal and contract is present.

R4 - ACCEPTANCE CRITERIA OBJECTIVITY:
  Every acceptance check in the proposal is objective, executable, and deterministic.
  Each check specifies a concrete pass/fail condition that can be evaluated by automated
  tooling. No subjective, vague, or unmeasurable criteria are present.
  MET: All acceptance checks are deterministic and executable.
  NOT_MET: Any acceptance check is subjective, vague, or not mechanically verifiable.

R5 - IMPLEMENTATION APPROACH CONCRETENESS:
  The deliverable_ref points to a specific artifact path. The claim describes a concrete
  implementation strategy, not an abstract intent. The approach is technically feasible
  given the declared dependencies and contract constraints.
  MET: Implementation approach is specific, feasible, and fully described.
  NOT_MET: Approach is abstract, hand-wavy, or technically infeasible.

R6 - NO CIRCULAR DEPENDENCIES OR MISSING PREREQUISITES:
  The proposal does not introduce circular dependency chains. All referenced dependencies
  exist and are available. No prerequisite modules are missing from the dependency graph.
  The proposal can be executed in the declared dependency order.
  MET: Dependency graph is acyclic and all prerequisites are available.
  NOT_MET: Circular dependency detected or prerequisite is missing.

EVALUATION PROCEDURE:
1. Assess each rubric R1 through R6 independently. For each, provide a MET or NOT_MET
   assessment with specific evidence from the proposal.
2. If ANY rubric is NOT_MET, the overall verdict MUST be FAIL.
3. For each NOT_MET rubric, add a ChallengeFailure entry to the failures list with:
   - invariant: The specific contract invariant or rubric requirement violated.
   - evidence: Concrete evidence from the proposal demonstrating the violation.
   - required_change: The specific change needed to resolve the failure.
4. If all rubrics are MET, the verdict is PASS and failures must be empty.

Return Challenge JSON only. No markdown, no commentary.\
"""

_ARBITER_SYSTEM_PROMPT = """\
You are the ARBITER in a production engineering debate. You receive a Proposal and a
Challenge and must render a binding decision: APPROVE, APPROVE_WITH_AMENDMENTS, or REJECT.

DECISION CRITERIA:

APPROVE:
  Use APPROVE only when ALL of the following are true:
  - The challenger's verdict is PASS, OR the challenger raised issues that are
    demonstrably incorrect or not applicable to this proposal.
  - All six rubrics (R1-R6) are satisfied based on your independent assessment.
  - The proposal is production-ready with no remaining issues.
  - Set ship_directive to SHIP.
  - Provide non-empty rationale explaining why every rubric is satisfied.

APPROVE_WITH_AMENDMENTS:
  Use APPROVE_WITH_AMENDMENTS only when ALL of the following are true:
  - The proposal is fundamentally sound and addresses the core contract.
  - The challenger raised minor issues that are real but do not constitute fundamental
    flaws. Examples: cosmetic naming, minor documentation gaps, non-blocking edge cases.
  - You can specify exactly what amendments are needed via the amendments list.
  - Each amendment must have a concrete action (MODIFY, ADD, or REMOVE), a specific
    target, and sufficient detail to implement without further design.
  - Set ship_directive to SHIP (the amendments will be applied post-ship).
  - Provide non-empty rationale explaining the assessment and why amendments are minor.

REJECT:
  Use REJECT when ANY of the following are true:
  - Any rubric R1-R6 is NOT_MET and the challenger's evidence is valid.
  - The proposal contains mocks, stubs, fakes, or placeholder behavior.
  - The proposal fundamentally misaligns with the contract summary.
  - Acceptance criteria are not objective or executable.
  - The implementation approach is infeasible or incomplete.
  - Set ship_directive to NO_SHIP.
  - Provide non-empty rationale identifying every unresolved rubric failure.

EVALUATION PROCEDURE:
1. Review the proposal claim, acceptance checks, and deliverable reference.
2. Review every rubric assessment in the challenge (R1-R6).
3. For each NOT_MET assessment, independently verify whether the challenger's evidence
   is valid. Do not rubber-stamp the challenge -- apply your own judgment.
4. Weigh the totality of evidence. If any fundamental issue remains unresolved, REJECT.
5. If only minor, well-specified issues remain, APPROVE_WITH_AMENDMENTS.
6. If everything is clean, APPROVE.

Return Adjudication JSON only. No markdown, no commentary.\
"""


class DebateState(TypedDict, total=False):
    """Typed state for the debate graph.

    All keys are optional (total=False) because the graph accumulates state
    across nodes. Keys are populated as the graph progresses through propose,
    challenge, adjudicate, and terminal nodes.
    """

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
    prior_challenge_summary: str
    halted: bool
    shipped: bool


@dataclass
class DebateResult:
    """Result payload returned by DebateGraph.run().

    Attributes:
        passed: True if the debate concluded with a SHIP directive.
        proposal: The final proposal (may reflect revisions).
        challenge: The final challenge assessment.
        adjudication: The final arbiter decision.
        retries_used: Number of revision cycles consumed.
    """

    passed: bool
    proposal: Proposal
    challenge: Challenge
    adjudication: Adjudication
    retries_used: int


class DebateGraph:
    """Nested debate subgraph: propose -> challenge -> route -> revise/adjudicate -> ship/halt.

    The debate graph implements a propose-challenge-adjudicate loop with bounded retries.
    On each iteration the proposer generates a Proposal, the challenger evaluates it against
    rubrics R1-R6, and if the challenge fails, the graph either revises (if retries remain)
    or sends the pair to the arbiter for a binding decision.
    """

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
        """Construct the LangGraph state graph for the debate loop."""
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
        """Load and serialize the context pack for the given role, if available."""
        refs = state.get("context_pack_refs", {})
        if not refs:
            return {}
        cp_id = refs.get(role)
        if not cp_id:
            return {}
        context_pack = self.store.read_context_pack(cp_id)
        return context_pack.model_dump(mode="json")

    def _propose(self, state: DebateState) -> dict[str, Any]:
        """Generate a Proposal via the proposer LLM.

        On revision passes, includes the prior challenge failure summary so the
        proposer can address specific failures from the previous iteration.
        """
        module_id = state["module_id"]
        artifact_id = state["target_artifact_id"]
        retry_count = int(state.get("retry_count", 0))

        # Build the prior challenge context block for revision passes
        prior_challenge_block = ""
        prior_summary = state.get("prior_challenge_summary", "")
        if retry_count > 0 and prior_summary:
            prior_challenge_block = (
                f"\nPriorChallengeFailures (revision {retry_count}, address ALL items below):\n"
                f"{prior_summary}\n"
            )

        prompt = (
            f"{_PROPOSER_SYSTEM_PROMPT}\n"
            f"{self.role_runtime.role_context_line('proposer')}\n"
            f"TaskID={state['task_id']}\n"
            f"ModuleID={module_id}\n"
            f"TargetArtifactID={artifact_id}\n"
            f"RetryCount={retry_count}\n"
            f"ContractSummary={state.get('contract_summary', {})}\n"
            f"ContextSummary={state.get('context_summary', '')}\n"
            f"ContextPack={self._context_pack_payload(state, 'proposer')}"
            f"{prior_challenge_block}"
        )

        proposal = self._proposer.invoke(prompt)

        # Validate target artifact alignment
        if proposal.target_artifact_id != artifact_id:
            raise ValueError(
                f"proposer returned target artifact mismatch: expected={artifact_id} got={proposal.target_artifact_id}"
            )
        # Validate proposal has substantive content
        if not proposal.claim or not proposal.claim.strip():
            raise ValueError("proposer returned proposal with empty claim")
        if not proposal.acceptance_checks:
            raise ValueError("proposer returned proposal with no acceptance_checks")

        logger.debug(
            "Proposer generated proposal %s for artifact %s (retry=%d)",
            proposal.proposal_id,
            artifact_id,
            retry_count,
        )
        return {"proposal": proposal}

    def _challenge(self, state: DebateState) -> dict[str, Any]:
        """Evaluate the proposal via the challenger LLM against rubrics R1-R6."""
        proposal = state["proposal"]

        prompt = (
            f"{_CHALLENGER_SYSTEM_PROMPT}\n"
            f"{self.role_runtime.role_context_line('challenger')}\n"
            f"Proposal={proposal.model_dump_json()}\n"
            f"ContractSummary={state.get('contract_summary', {})}\n"
            f"ContextSummary={state.get('context_summary', '')}\n"
            f"ContextPack={self._context_pack_payload(state, 'challenger')}"
        )

        challenge = self._challenger.invoke(prompt)

        # Validate target artifact alignment
        if challenge.target_artifact_id != proposal.target_artifact_id:
            raise ValueError(
                "challenger returned target artifact mismatch: "
                f"expected={proposal.target_artifact_id} got={challenge.target_artifact_id}"
            )
        # FAIL verdict must include concrete failures
        if challenge.verdict == DebateVerdict.FAIL and not challenge.failures:
            raise ValueError("challenger returned FAIL verdict without concrete failures")
        # PASS verdict must have zero failures (no contradictory state)
        if challenge.verdict == DebateVerdict.PASS and challenge.failures:
            raise ValueError(
                f"challenger returned PASS verdict with {len(challenge.failures)} failures; "
                "PASS requires an empty failures list"
            )

        logger.debug(
            "Challenger verdict=%s for proposal %s (%d failures)",
            challenge.verdict.value,
            proposal.proposal_id,
            len(challenge.failures),
        )
        return {"challenge": challenge}

    def _route(self, state: DebateState) -> Command[str]:
        """Route from challenge to either revise (if FAIL with retries) or adjudicate."""
        challenge = state["challenge"]
        retries_used = int(state.get("retry_count", 0))
        max_retries = int(state.get("max_retries", 2))
        if challenge.verdict == DebateVerdict.FAIL and retries_used < max_retries:
            return Command(goto="revise")
        return Command(goto="adjudicate")

    def _revise(self, state: DebateState) -> dict[str, Any]:
        """Prepare state for a revision pass.

        Increments the retry counter and builds a structured summary of the
        challenge failures so the proposer can address each specific issue
        in the next iteration.
        """
        retries = int(state.get("retry_count", 0)) + 1
        challenge = state.get("challenge")

        # Build a structured summary of challenge failures for the proposer
        failure_lines: list[str] = []
        if challenge is not None and challenge.failures:
            for idx, failure in enumerate(challenge.failures, start=1):
                failure_lines.append(
                    f"  [{idx}] Invariant: {failure.invariant}\n"
                    f"      Evidence: {failure.evidence}\n"
                    f"      Required change: {failure.required_change}"
                )
        if challenge is not None and challenge.rubric_assessments:
            not_met = [ra for ra in challenge.rubric_assessments if ra.assessment == "NOT_MET"]
            if not_met:
                failure_lines.append("  Rubrics NOT_MET:")
                for ra in not_met:
                    failure_lines.append(f"    {ra.criterion}: {ra.evidence}")

        prior_challenge_summary = "\n".join(failure_lines) if failure_lines else ""

        logger.debug("Revise pass %d with %d failure items", retries, len(failure_lines))
        return {
            "retry_count": retries,
            "prior_challenge_summary": prior_challenge_summary,
        }

    def _adjudicate(self, state: DebateState) -> dict[str, Any]:
        """Render a binding decision via the arbiter LLM."""
        proposal = state["proposal"]
        challenge = state["challenge"]

        prompt = (
            f"{_ARBITER_SYSTEM_PROMPT}\n"
            f"{self.role_runtime.role_context_line('arbiter')}\n"
            f"Proposal={proposal.model_dump_json()}\n"
            f"Challenge={challenge.model_dump_json()}\n"
            f"ContractSummary={state.get('contract_summary', {})}\n"
            f"ContextSummary={state.get('context_summary', '')}\n"
            f"ContextPack={self._context_pack_payload(state, 'arbiter')}"
        )

        adjudication = self._arbiter.invoke(prompt)

        # Validate target artifact alignment
        if adjudication.target_artifact_id != proposal.target_artifact_id:
            raise ValueError(
                "arbiter returned target artifact mismatch: "
                f"expected={proposal.target_artifact_id} got={adjudication.target_artifact_id}"
            )
        # REJECT must have NO_SHIP
        if adjudication.decision == AdjudicationDecision.REJECT and adjudication.ship_directive != ShipDirective.NO_SHIP:
            raise ValueError("arbiter returned REJECT with ship_directive != NO_SHIP")
        # APPROVE and APPROVE_WITH_AMENDMENTS must have SHIP
        if adjudication.decision in {
            AdjudicationDecision.APPROVE,
            AdjudicationDecision.APPROVE_WITH_AMENDMENTS,
        } and adjudication.ship_directive != ShipDirective.SHIP:
            raise ValueError(
                f"arbiter returned {adjudication.decision.value} with ship_directive != SHIP"
            )
        # All decisions must have non-empty rationale
        if not adjudication.rationale or not adjudication.rationale.strip():
            raise ValueError("arbiter returned adjudication with empty rationale")
        # APPROVE_WITH_AMENDMENTS must specify at least one amendment
        if adjudication.decision == AdjudicationDecision.APPROVE_WITH_AMENDMENTS and not adjudication.amendments:
            raise ValueError("arbiter returned APPROVE_WITH_AMENDMENTS with no amendments")

        logger.debug(
            "Arbiter decision=%s ship=%s for artifact %s",
            adjudication.decision.value,
            adjudication.ship_directive.value,
            proposal.target_artifact_id,
        )
        return {"adjudication": adjudication}

    def _post_adjudicate(self, state: DebateState) -> Command[str]:
        """Route from adjudication to ship, revise, or halt.

        APPROVE/APPROVE_WITH_AMENDMENTS with SHIP -> ship.
        REJECT with retries remaining -> revise for another attempt.
        REJECT with no retries remaining -> halt (final).
        """
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
        """Write all three debate artifacts (proposal, challenge, adjudication) to the store."""
        self.store.write_debate(
            artifact_id=state["target_artifact_id"],
            proposal=state["proposal"],
            challenge=state["challenge"],
            adjudication=state["adjudication"],
        )

    def _ship(self, state: DebateState) -> dict[str, Any]:
        """Terminal node: persist artifacts and mark as shipped."""
        self._persist(state)
        logger.info("Debate SHIPPED for artifact %s", state["target_artifact_id"])
        return {"shipped": True, "halted": False}

    def _halt(self, state: DebateState) -> dict[str, Any] | Command[str]:
        """Terminal node: persist artifacts and mark as halted.

        If propagate_parent_halt is set, sends a Command to the parent graph
        to propagate the halt signal upward.
        """
        self._persist(state)
        logger.warning("Debate HALTED for artifact %s", state["target_artifact_id"])
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
        """Execute the full debate loop and return the result.

        Args:
            task_id: The parent task identifier.
            module_id: The micro-module being debated.
            target_artifact_id: The contract artifact the proposal targets.
            contract_summary: Serialized contract for context injection.
            context_summary: Human-readable context from the engineering loop.
            context_pack_refs: Optional per-role context pack identifiers.
            max_retries: Maximum revision cycles before final adjudication.

        Returns:
            DebateResult with the outcome, artifacts, and retry count.
        """
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
                "prior_challenge_summary": "",
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
