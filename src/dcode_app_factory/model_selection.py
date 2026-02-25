from __future__ import annotations

import os
from dataclasses import dataclass

from .models import AgentConfig


DEFAULT_MODELS_BY_TIER: dict[str, str] = {
    "frontier": "openai:gpt-4o",
    "efficient": "openai:gpt-4o-mini",
    "economy": "openai:gpt-4o-mini",
}


@dataclass(frozen=True)
class RuntimeModelSelection:
    """Runtime model routing by tier."""

    by_tier: dict[str, str]

    @classmethod
    def from_env(cls) -> "RuntimeModelSelection":
        by_tier = dict(DEFAULT_MODELS_BY_TIER)
        for tier, env_key in {
            "frontier": "FACTORY_MODEL_FRONTIER",
            "efficient": "FACTORY_MODEL_EFFICIENT",
            "economy": "FACTORY_MODEL_ECONOMY",
        }.items():
            configured_model = os.getenv(env_key)
            if configured_model:
                by_tier[tier] = configured_model.strip()

        return cls(by_tier=by_tier)

    def resolve(self, stage: str, role: str, model_tier: str) -> str:
        _ = stage, role
        if model_tier in self.by_tier:
            return self.by_tier[model_tier]
        raise ValueError(f"No default model configured for tier: {model_tier}")


def resolve_agent_models(configs: dict[str, AgentConfig], model_selection: RuntimeModelSelection) -> dict[str, str]:
    return {
        role: model_selection.resolve(cfg.stage, cfg.role, cfg.model_tier)
        for role, cfg in sorted(configs.items())
    }
