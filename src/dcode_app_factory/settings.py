from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeSettings:
    """Runtime-tunable settings loaded from environment variables."""

    max_product_sections: int = 8
    context_budget_floor_tokens: int = 2_000
    context_budget_cap_tokens: int = 16_000
    default_spec_path: str = "SPEC.md"

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            max_product_sections=_get_env_int("FACTORY_MAX_PRODUCT_SECTIONS", default=8, minimum=1),
            context_budget_floor_tokens=_get_env_int("FACTORY_CONTEXT_BUDGET_FLOOR", default=2_000, minimum=256),
            context_budget_cap_tokens=_get_env_int("FACTORY_CONTEXT_BUDGET_CAP", default=16_000, minimum=512),
            default_spec_path=os.getenv("FACTORY_DEFAULT_SPEC_PATH", "SPEC.md"),
        ).normalized()

    def normalized(self) -> "RuntimeSettings":
        cap = max(self.context_budget_cap_tokens, self.context_budget_floor_tokens)
        return RuntimeSettings(
            max_product_sections=self.max_product_sections,
            context_budget_floor_tokens=self.context_budget_floor_tokens,
            context_budget_cap_tokens=cap,
            default_spec_path=self.default_spec_path,
        )

    def default_spec_file(self, repo_root: Path) -> Path:
        path = Path(self.default_spec_path)
        if path.is_absolute():
            return path
        return repo_root / path


def _get_env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {raw!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got: {parsed}")
    return parsed
