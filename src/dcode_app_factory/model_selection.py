from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .models import AgentConfig


DEFAULT_MODELS_BY_TIER: dict[str, str] = {
    "frontier": "openai:gpt-4o",
    "efficient": "openai:gpt-4o-mini",
    "economy": "openai:gpt-4o-mini",
}


@dataclass(frozen=True)
class RuntimeModelSelection:
    """Runtime model routing for agent roles.

    - by_tier maps model tiers (e.g. frontier/efficient) to default model IDs.
    - by_role supports optional explicit overrides by `stage.role` or `role`.
    """

    by_tier: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODELS_BY_TIER))
    by_role: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "RuntimeModelSelection":
        by_tier = dict(DEFAULT_MODELS_BY_TIER)
        for tier, env_key in {
            "frontier": "FACTORY_MODEL_FRONTIER",
            "efficient": "FACTORY_MODEL_EFFICIENT",
            "economy": "FACTORY_MODEL_ECONOMY",
        }.items():
            override = os.getenv(env_key)
            if override:
                by_tier[tier] = override.strip()

        role_overrides_json = os.getenv("FACTORY_MODEL_ROLE_OVERRIDES_JSON", "").strip()
        by_role: dict[str, str] = {}
        if role_overrides_json:
            parsed = json.loads(role_overrides_json)
            if not isinstance(parsed, dict):
                raise ValueError("FACTORY_MODEL_ROLE_OVERRIDES_JSON must be a JSON object")
            for key, value in parsed.items():
                if not isinstance(key, str) or not isinstance(value, str) or not value.strip():
                    raise ValueError("FACTORY_MODEL_ROLE_OVERRIDES_JSON must map string keys to non-empty strings")
                by_role[key.strip()] = value.strip()

        return cls(by_tier=by_tier, by_role=by_role)

    def resolve(self, stage: str, role: str, model_tier: str) -> str:
        stage_role_key = f"{stage}.{role}"
        if stage_role_key in self.by_role:
            return self.by_role[stage_role_key]
        if role in self.by_role:
            return self.by_role[role]
        if model_tier in self.by_tier:
            return self.by_tier[model_tier]
        raise ValueError(f"No default model configured for tier: {model_tier}")


def resolve_agent_models(configs: dict[str, AgentConfig], model_selection: RuntimeModelSelection) -> dict[str, str]:
    return {
        role: model_selection.resolve(cfg.stage, cfg.role, cfg.model_tier)
        for role, cfg in sorted(configs.items())
    }
