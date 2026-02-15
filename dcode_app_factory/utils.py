"""Utility functions for the AI software product factory.

This module collects helper functions used throughout the package,
including slugifying names for safe file and identifier generation and
producing canonical JSON strings. These utilities wrap third‑party
libraries to provide a consistent API.
"""

from __future__ import annotations

from typing import Any

import re
import json


def slugify_name(name: str) -> str:
    """Return a URL‑ and filename‑safe slug for the given name.

    The slug is lower‑cased, uses hyphens as separators, and strips
    characters that are unsafe for file systems. Consecutive
    separators are collapsed into a single hyphen. Leading and
    trailing hyphens are removed.

    This implementation avoids external dependencies by using a
    regular expression to replace non‑alphanumeric characters with
    hyphens.

    Args:
        name: Arbitrary input string.

    Returns:
        A slugified version of the input.
    """
    # Lowercase and replace any sequence of non‑alphanumeric characters with a hyphen
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower())
    # Strip leading and trailing hyphens
    slug = slug.strip("-")
    return slug


def to_canonical_json(value: Any) -> str:
    """Serialize a value to a deterministic JSON string.

    Keys are sorted alphabetically and all unnecessary whitespace is
    removed. This is a pragmatic substitute for full RFC 8785
    canonicalisation and suffices for hashing or comparison in the
    absence of an external dependency.

    Args:
        value: Any JSON‑serialisable object (dict, list, int, str, etc.).

    Returns:
        A string containing the canonical JSON representation of the input.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
