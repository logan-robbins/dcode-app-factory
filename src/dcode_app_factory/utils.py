from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from .models import AgentConfig, ContextPack, StructuredSpec


def get_agent_config_dir(stage: str) -> Path:
    """Return the package-relative path to agent configs for a given stage."""
    return Path(__file__).resolve().parent / "agent_configs" / stage


def slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def to_canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_agent_config(path: Path) -> AgentConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return AgentConfig(**payload)


def validate_task_dependency_dag(spec: StructuredSpec) -> None:
    tasks = {}
    for pillar in spec.pillars:
        for epic in pillar.epics:
            for story in epic.stories:
                for task in story.tasks:
                    if task.task_id in tasks:
                        raise ValueError(f"Duplicate task_id: {task.task_id}")
                    tasks[task.task_id] = task

    indegree = {task_id: 0 for task_id in tasks}
    outgoing = defaultdict(list)
    for task_id, task in tasks.items():
        for dep in task.depends_on:
            if dep not in tasks:
                raise ValueError(f"Task {task_id} depends on unknown task {dep}")
            outgoing[dep].append(task_id)
            indegree[task_id] += 1

    queue = deque([task_id for task_id, degree in indegree.items() if degree == 0])
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for neighbor in outgoing[current]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    if visited != len(tasks):
        raise ValueError("Task dependency graph contains a cycle")


def build_context_pack(task_id: str, objective: str, stage: str) -> ContextPack:
    base_interfaces = [
        "MicroModuleContract",
        "CodeIndex.register(contract)",
        "Debate: propose->challenge->adjudicate",
    ]
    return ContextPack(
        task_id=task_id,
        objective=objective,
        interfaces=base_interfaces,
        allowed_files=[f"state_store/tasks/{task_id}.md", f"state_store/context/{stage}/{task_id}.json"],
        denied_files=["state_store/code_index/*.json", "**/*.py"],
    )
