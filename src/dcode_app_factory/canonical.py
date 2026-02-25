from __future__ import annotations

import logging
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

import rfc8785
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# JSON-primitive types that rfc8785 can serialize directly.
_PASSTHROUGH_TYPES = (bool, int, float, str, type(None))


def _normalize_for_jcs(value: Any) -> bool | int | float | str | None | list[Any] | dict[str, Any]:
    """Recursively convert Python/Pydantic types into JSON-primitive types.

    rfc8785.dumps only accepts: bool, int, float, str, None, list/tuple, dict.
    This function converts Pydantic models, datetime, UUID, Decimal, and Enum
    values into their JSON-compatible representations before serialization.

    Args:
        value: Any Python value to normalize for JCS serialization.

    Returns:
        A JSON-primitive structure suitable for rfc8785.dumps.

    Raises:
        TypeError: If value contains a type that cannot be converted to JSON.
    """
    if isinstance(value, _PASSTHROUGH_TYPES):
        return value

    if isinstance(value, BaseModel):
        return _normalize_for_jcs(value.model_dump(mode="json"))

    if isinstance(value, dict):
        return {str(k): _normalize_for_jcs(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_normalize_for_jcs(item) for item in value]

    if isinstance(value, Enum):
        return _normalize_for_jcs(value.value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, time):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, Decimal):
        # Decimal -> float for JCS. Finite decimals only.
        if not value.is_finite():
            raise TypeError(f"Cannot serialize non-finite Decimal to JSON: {value!r}")
        return float(value)

    if isinstance(value, bytes):
        raise TypeError(
            f"Cannot serialize bytes to canonical JSON. "
            f"Encode to base64 or hex string first: {value!r:.64}"
        )

    raise TypeError(
        f"Cannot serialize type {type(value).__name__} to canonical JSON. "
        f"Convert to a JSON-compatible type first."
    )


def to_canonical_json(value: Any) -> str:
    """Serialize a value to deterministic, byte-for-byte reproducible JSON per RFC 8785.

    Handles Pydantic models, datetime, UUID, Decimal, Enum, and nested structures
    by normalizing them into JSON-primitive types before passing to rfc8785.dumps.

    Args:
        value: Any Python value including Pydantic models and special types.

    Returns:
        A UTF-8 string containing the canonicalized JSON representation.

    Raises:
        TypeError: If value contains an unsupported type.
        rfc8785.CanonicalizationError: If rfc8785 rejects the normalized value.
    """
    normalized = _normalize_for_jcs(value)
    return rfc8785.dumps(normalized).decode("utf-8")
