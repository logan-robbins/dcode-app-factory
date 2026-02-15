"""Dataclass models for the AI software product factory.

This module provides plain Python dataclasses that mirror the shape of
the Pydantic models described in the specification. Dataclasses are
used instead of Pydantic to avoid reliance on external packages that
may not be available in the execution environment. The dataclasses
include simple default values and type hints but no runtime
validation beyond Python's built‑in type system.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class InputSpec:
    """Describes a single input parameter for a micro module."""
    name: str
    type: str
    description: Optional[str] = None


@dataclass(frozen=True)
class OutputSpec:
    """Describes a single output value for a micro module."""
    name: str
    type: str
    description: Optional[str] = None


@dataclass(frozen=True)
class ErrorSpec:
    """Describes an error condition that a micro module may raise."""
    code: str
    message: str


@dataclass(frozen=True)
class MicroModuleContract:
    """A contract describing a micro module's interface."""
    name: str
    description: str
    version: str = "0.1.0"
    inputs: List[InputSpec] = field(default_factory=list)
    outputs: List[OutputSpec] = field(default_factory=list)
    errors: List[ErrorSpec] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)


@dataclass
class Task:
    """Represents a single implementable task derived from the structured spec."""
    name: str
    description: str
    contract: Optional[MicroModuleContract] = None
    status: str = "pending"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Story:
    """User‑facing behaviour grouped under an epic."""
    name: str
    description: str
    tasks: List[Task] = field(default_factory=list)


@dataclass
class Epic:
    """Major capability within a pillar."""
    name: str
    description: str
    stories: List[Story] = field(default_factory=list)


@dataclass
class Pillar:
    """Strategic theme in the structured specification."""
    name: str
    description: str
    epics: List[Epic] = field(default_factory=list)


@dataclass
class StructuredSpec:
    """Top‑level structured specification produced by the Product Loop."""
    pillars: List[Pillar] = field(default_factory=list)
    version: str = "0.1.0"
