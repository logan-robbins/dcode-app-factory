from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .models import ContextPack, DebateTrace


class DebateResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class DebateAgent(Protocol):
    def __call__(self, task_prompt: str, context: ContextPack) -> str: ...


@dataclass
class Debate:
    proposer: DebateAgent
    challenger: DebateAgent
    arbiter: DebateAgent

    def run(self, task_prompt: str, context: ContextPack) -> tuple[DebateResult, DebateTrace]:
        trace = DebateTrace()
        proposal = self.proposer(task_prompt, context)
        trace.proposal = proposal
        challenge = self.challenger(proposal, context)
        trace.challenge = challenge
        adjudication = self.arbiter(f"proposal={proposal}\nchallenge={challenge}", context)
        trace.adjudication = adjudication
        passed = "pass" in adjudication.lower() and "fail" not in adjudication.lower().split()
        trace.passed = passed
        return (DebateResult.PASS if passed else DebateResult.FAIL, trace)
