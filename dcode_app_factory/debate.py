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
        trace.proposal = self.proposer(task_prompt, context)
        trace.challenge = self.challenger(trace.proposal, context)
        trace.adjudication = self.arbiter(f"proposal={trace.proposal}\nchallenge={trace.challenge}", context).strip().upper()
        trace.passed = trace.adjudication == "PASS"
        return (DebateResult.PASS if trace.passed else DebateResult.FAIL, trace)
