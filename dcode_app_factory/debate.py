"""Minimal debate mechanism for engineering tasks.

The debate mechanism implements a three‑phase protocol inspired by the
specification: proposal, challenge, and adjudication. Each phase is
represented by a callable that returns a boolean indicating success.
Failure at any phase halts the debate and returns a FAIL result.

The debate is intentionally simplified. Real implementations would
invoke distinct LLMs or agents to propose, challenge, and adjudicate
solutions. Here, we operate on callables provided to the debate
engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Any


class DebateResult(Enum):
    """Enumeration of possible debate outcomes."""

    PASS = "pass"
    FAIL = "fail"


@dataclass
class Debate:
    """Orchestrates a simple debate sequence.

    Args:
        proposer: Function that generates a proposal. It should return
            ``True`` on success or ``False`` on failure.
        challenger: Function that challenges the proposal. It receives
            the proposal output and should return ``True`` if the
            proposal is acceptable.
        adjudicator: Function that adjudicates the challenged proposal.
            It receives the proposal output and challenge result and
            returns ``True`` for a final PASS or ``False`` for FAIL.
    """

    proposer: Callable[[], Any]
    challenger: Callable[[Any], bool]
    adjudicator: Callable[[Any, bool], bool]

    def run(self) -> DebateResult:
        """Execute the debate process.

        Returns:
            ``DebateResult.PASS`` if all phases succeed, otherwise
            ``DebateResult.FAIL``.
        """
        try:
            proposal = self.proposer()
        except Exception:
            return DebateResult.FAIL

        # Challenger evaluates the proposal
        try:
            challenge_ok = self.challenger(proposal)
        except Exception:
            return DebateResult.FAIL

        if not challenge_ok:
            return DebateResult.FAIL

        # Adjudicator makes the final call
        try:
            adjudication_ok = self.adjudicator(proposal, challenge_ok)
        except Exception:
            return DebateResult.FAIL

        return DebateResult.PASS if adjudication_ok else DebateResult.FAIL
