from __future__ import annotations

import os
from dataclasses import dataclass

from .models import AgentConfig


VALID_TIERS: frozenset[str] = frozenset({"frontier", "efficient", "economy"})

DEFAULT_MODELS_BY_TIER: dict[str, str] = {
    "frontier": "gpt-4o",
    "efficient": "gpt-4o-mini",
    "economy": "gpt-4o-mini",
}


@dataclass(frozen=True)
class RuntimeModelSelection:
    """Maps model tier names to concrete model identifiers for agent-role routing.

    Each agent config declares a ``model_tier`` (frontier, efficient, economy).
    This dataclass holds the concrete model names for each tier so that
    ``resolve`` can translate tier to model name at runtime.
    """

    by_tier: dict[str, str]

    def __post_init__(self) -> None:
        """Validate that all required tiers are present and no tier maps to an empty model name."""
        missing = VALID_TIERS - set(self.by_tier)
        if missing:
            raise ValueError(
                f"RuntimeModelSelection missing required tiers: {', '.join(sorted(missing))}. "
                f"All of {sorted(VALID_TIERS)} must be configured."
            )
        for tier, model_name in self.by_tier.items():
            if not model_name or not model_name.strip():
                raise ValueError(f"RuntimeModelSelection tier '{tier}' has empty model name")

    @classmethod
    def from_env(cls) -> "RuntimeModelSelection":
        """Build model selection from environment variables with validated defaults.

        Environment variables:
            FACTORY_MODEL_FRONTIER: Model name for frontier tier.
            FACTORY_MODEL_EFFICIENT: Model name for efficient tier.
            FACTORY_MODEL_ECONOMY: Model name for economy tier.

        Returns:
            RuntimeModelSelection with all three tiers populated.
        """
        by_tier = dict(DEFAULT_MODELS_BY_TIER)
        for tier, env_key in {
            "frontier": "FACTORY_MODEL_FRONTIER",
            "efficient": "FACTORY_MODEL_EFFICIENT",
            "economy": "FACTORY_MODEL_ECONOMY",
        }.items():
            configured_model = os.getenv(env_key)
            if configured_model and configured_model.strip():
                by_tier[tier] = configured_model.strip()

        return cls(by_tier=by_tier)

    def resolve(self, stage: str, role: str, model_tier: str) -> str:
        """Resolve a model tier to a concrete model name.

        Args:
            stage: Pipeline stage name (for diagnostics).
            role: Agent role name (for diagnostics).
            model_tier: One of 'frontier', 'efficient', 'economy'.

        Returns:
            The concrete model name string.

        Raises:
            ValueError: If model_tier is not a recognized tier.
        """
        if model_tier not in self.by_tier:
            available = ", ".join(sorted(self.by_tier))
            raise ValueError(
                f"Unknown model tier '{model_tier}' for {stage}/{role}. "
                f"Valid tiers: {available}"
            )
        return self.by_tier[model_tier]


def resolve_agent_models(
    configs: dict[str, AgentConfig],
    model_selection: RuntimeModelSelection,
) -> dict[str, str]:
    """Resolve all agent configs to concrete model names via the model selection mapping.

    Args:
        configs: Mapping of role name to AgentConfig.
        model_selection: Tier-to-model mapping.

    Returns:
        Mapping of role name to concrete model name string.

    Raises:
        ValueError: If any config references an unknown model tier.
    """
    return {
        role: model_selection.resolve(cfg.stage, cfg.role, cfg.model_tier)
        for role, cfg in sorted(configs.items())
    }
