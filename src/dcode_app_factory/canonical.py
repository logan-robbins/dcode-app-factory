from __future__ import annotations

from typing import Any

import rfc8785


def to_canonical_json(value: Any) -> str:
    return rfc8785.dumps(value).decode("utf-8")
