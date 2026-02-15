"""Implementation of the Product, Project, and Engineering loops.

These classes orchestrate the sequential flow described in the
specification. Each loop focuses on a distinct phase of the factory:
researching and structuring the spec, breaking it down into tasks,
and implementing micro modules via a debate mechanism.

The loops are intentionally lightweight and deterministic for this
exercise. They use the data models defined in :mod:`dcode_app_factory.models`
and the registry in :mod:`dcode_app_factory.registry`. The engineering
loop integrates the debate mechanism from :mod:`dcode_app_factory.debate`.
"""

from __future__ import annotations

import logging
from typing import Optional, List

from .models import StructuredSpec, Pillar, Epic, Story, Task, MicroModuleContract, InputSpec, OutputSpec
from .registry import CodeIndex
from .debate import Debate, DebateResult

logger = logging.getLogger(__name__)


class ProductLoop:
    """Refines a human‑authored spec into a structured specification.

    In this simplified implementation the ProductLoop generates a
    placeholder structured spec rather than performing true research
    and decomposition. In a real system, this loop would invoke an
    agent equipped with web search and summarisation tools to create
    pillars, epics, stories, and tasks. The human would then approve
    the result before moving on to the Project Loop.
    """

    def __init__(self, raw_spec: str) -> None:
        self.raw_spec = raw_spec

    def run(self) -> StructuredSpec:
        """Generate a structured specification from the raw spec.

        Returns:
            A :class:`StructuredSpec` containing placeholder data.
        """
        logger.info("Starting ProductLoop to structure the spec")
        # Create a very simple structured spec with a single task. This
        # fulfils the requirement to produce pillars→epics→stories→tasks.
        task = Task(
            name="implement skeleton module",
            description="Implement a placeholder micro module as part of the factory scaffold.",
            contract=MicroModuleContract(
                name="placeholder_module",
                description="A micro module created by the ProductLoop skeleton.",
                inputs=[InputSpec(name="input", type="Any", description="Generic input")],
                outputs=[OutputSpec(name="output", type="Any", description="Generic output")],
                errors=[],
            ),
        )
        story = Story(name="Placeholder story", description="A single story in the skeleton", tasks=[task])
        epic = Epic(name="Placeholder epic", description="A single epic in the skeleton", stories=[story])
        pillar = Pillar(name="Placeholder pillar", description="A single pillar in the skeleton", epics=[epic])
        spec = StructuredSpec(pillars=[pillar])
        logger.info("ProductLoop produced structured spec with %s pillars", len(spec.pillars))
        return spec


class ProjectLoop:
    """Decomposes the structured spec into tasks and manages execution.

    The ProjectLoop iterates through tasks defined in the structured
    spec, dispatching each one to the EngineeringLoop in sequence. It
    halts on the first failure, reflecting the spec's requirement to
    stop execution on errors.
    """

    def __init__(self, spec: StructuredSpec, code_index: Optional[CodeIndex] = None) -> None:
        self.spec = spec
        # Always honour an explicitly supplied code index, even if it is
        # currently empty. CodeIndex implements ``__len__``, so using it in
        # a boolean context could incorrectly select a new CodeIndex when
        # ``len(code_index) == 0``. Avoid this by checking for None explicitly.
        self.code_index = code_index if code_index is not None else CodeIndex()

    def _flatten_tasks(self) -> List[Task]:
        """Flatten all tasks from the structured spec into a list."""
        tasks: List[Task] = []
        for pillar in self.spec.pillars:
            for epic in pillar.epics:
                for story in epic.stories:
                    tasks.extend(story.tasks)
        return tasks

    def run(self) -> bool:
        """Execute all tasks through the Engineering Loop.

        Returns:
            ``True`` if all tasks pass; ``False`` if any task fails.
        """
        logger.info("Starting ProjectLoop with %s tasks", len(self._flatten_tasks()))
        for task in self._flatten_tasks():
            engineering = EngineeringLoop(task, self.code_index)
            success = engineering.run()
            if not success:
                logger.error("Task %s failed during EngineeringLoop", task.name)
                return False
        logger.info("All tasks completed successfully in ProjectLoop")
        return True


class EngineeringLoop:
    """Implements a single micro task via a debate.

    The EngineeringLoop wraps a simple debate mechanism around a
    placeholder implementation. In a real system this loop would
    iterate through micro tasks, performing a micro plan and then
    triggering the debate subgraph for each proposed module.
    """

    def __init__(self, task: Task, code_index: CodeIndex) -> None:
        self.task = task
        self.code_index = code_index

    def _implement(self) -> bool:
        """Placeholder implementation for a micro module.

        This method represents the proposal phase. It registers the
        task's contract with the code index and returns ``True`` to
        indicate success. In a real system this would perform the
        actual engineering work (writing code, tests, etc.).
        """
        contract = self.task.contract
        if not contract:
            # No contract defined; cannot implement
            logger.error("Task '%s' lacks a contract and cannot be implemented", self.task.name)
            return False
        try:
            slug = self.code_index.register(contract)
            logger.debug("Registered module '%s' as '%s'", contract.name, slug)
        except ValueError as exc:
            logger.error(str(exc))
            return False
        # Placeholder: always succeed in implementation
        return True

    def _challenge(self, result: bool) -> bool:
        """Placeholder challenger.

        For demonstration purposes the challenger simply returns the
        incoming result. A more sophisticated challenger would analyse
        the proposal, verify invariants, and attempt to falsify the
        work.
        """
        return bool(result)

    def _adjudicate(self, result: bool, challenge_ok: bool) -> bool:
        """Placeholder adjudication.

        Always returns the challenger result. In a real system the
        adjudicator could weigh evidence or apply additional logic.
        """
        return bool(challenge_ok)

    def run(self) -> bool:
        """Execute the debate and update task status.

        Returns:
            ``True`` if the debate passes and the task is marked
            completed; ``False`` otherwise.
        """
        logger.info("EngineeringLoop starting for task '%s'", self.task.name)
        debate = Debate(self._implement, self._challenge, self._adjudicate)
        outcome = debate.run()
        if outcome is DebateResult.PASS:
            self.task.status = "completed"
            logger.info("Task '%s' completed successfully", self.task.name)
            return True
        else:
            self.task.status = "failed"
            logger.warning("Task '%s' failed in EngineeringLoop", self.task.name)
            return False
