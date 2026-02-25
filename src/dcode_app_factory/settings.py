from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeSettings:
    """Runtime settings loaded from environment with fail-fast validation."""

    max_product_sections: int = 8
    context_budget_floor_tokens: int = 2_000
    context_budget_cap_tokens: int = 16_000
    default_request_path: str = "SPEC.md"
    state_store_root: str = "state_store"
    project_id: str = "PROJECT-001"
    model_frontier: str = "gpt-4o"
    model_efficient: str = "gpt-4o-mini"
    model_economy: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    recursion_limit: int = 1_000
    checkpoint_db: str = "state_store/checkpoints/langgraph.sqlite"
    class_contract_policy: str = "selective_shared"
    workspace_root: str = ""

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            max_product_sections=_get_env_int("FACTORY_MAX_PRODUCT_SECTIONS", default=8, minimum=1),
            context_budget_floor_tokens=_get_env_int("FACTORY_CONTEXT_BUDGET_FLOOR", default=2_000, minimum=256),
            context_budget_cap_tokens=_get_env_int("FACTORY_CONTEXT_BUDGET_CAP", default=16_000, minimum=512),
            default_request_path=os.getenv("FACTORY_DEFAULT_REQUEST_PATH", "SPEC.md"),
            state_store_root=os.getenv("FACTORY_STATE_STORE_ROOT", "state_store"),
            project_id=os.getenv("FACTORY_PROJECT_ID", "PROJECT-001"),
            model_frontier=os.getenv("FACTORY_MODEL_FRONTIER", "gpt-4o"),
            model_efficient=os.getenv("FACTORY_MODEL_EFFICIENT", "gpt-4o-mini"),
            model_economy=os.getenv("FACTORY_MODEL_ECONOMY", "gpt-4o-mini"),
            embedding_model=os.getenv("FACTORY_EMBEDDING_MODEL", "text-embedding-3-large"),
            recursion_limit=_get_env_int("FACTORY_RECURSION_LIMIT", default=1_000, minimum=100),
            checkpoint_db=os.getenv("FACTORY_CHECKPOINT_DB", "state_store/checkpoints/langgraph.sqlite"),
            class_contract_policy=os.getenv("FACTORY_CLASS_CONTRACT_POLICY", "selective_shared"),
            workspace_root=os.getenv("FACTORY_WORKSPACE_ROOT", ""),
        ).normalized()

    @property
    def workspace_root_path(self) -> Path:
        """Return the workspace root as a Path, defaulting to cwd if unset."""
        return Path(self.workspace_root) if self.workspace_root else Path.cwd()

    def normalized(self) -> "RuntimeSettings":
        """Validate and normalize all fields. Raises ValueError on invalid configuration."""
        cap = max(self.context_budget_cap_tokens, self.context_budget_floor_tokens)

        # -- Model name validation --
        model_frontier = self.model_frontier.strip()
        if not model_frontier:
            raise ValueError("FACTORY_MODEL_FRONTIER must be non-empty")
        model_efficient = self.model_efficient.strip()
        if not model_efficient:
            raise ValueError("FACTORY_MODEL_EFFICIENT must be non-empty")
        model_economy = self.model_economy.strip()
        if not model_economy:
            raise ValueError("FACTORY_MODEL_ECONOMY must be non-empty")
        embedding_model = self.embedding_model.strip()
        if not embedding_model:
            raise ValueError("FACTORY_EMBEDDING_MODEL must be non-empty")

        # -- Numeric bounds validation --
        if self.max_product_sections > 100:
            raise ValueError(
                f"FACTORY_MAX_PRODUCT_SECTIONS must be <= 100, got: {self.max_product_sections}"
            )
        if self.recursion_limit > 100_000:
            raise ValueError(
                f"FACTORY_RECURSION_LIMIT must be <= 100000, got: {self.recursion_limit}"
            )

        # -- String field validation --
        if not self.project_id.strip():
            raise ValueError("FACTORY_PROJECT_ID must be non-empty")
        if not self.default_request_path.strip():
            raise ValueError("FACTORY_DEFAULT_REQUEST_PATH must be non-empty")
        if not self.state_store_root.strip():
            raise ValueError("FACTORY_STATE_STORE_ROOT must be non-empty")
        if not self.checkpoint_db.strip():
            raise ValueError("FACTORY_CHECKPOINT_DB must be non-empty")

        # -- Policy validation --
        class_contract_policy = self.class_contract_policy.strip().lower()
        if class_contract_policy not in {"selective_shared", "universal_public", "service_only"}:
            raise ValueError(
                "FACTORY_CLASS_CONTRACT_POLICY must be one of: selective_shared, universal_public, service_only"
            )
        return RuntimeSettings(
            max_product_sections=self.max_product_sections,
            context_budget_floor_tokens=self.context_budget_floor_tokens,
            context_budget_cap_tokens=cap,
            default_request_path=self.default_request_path,
            state_store_root=self.state_store_root,
            project_id=self.project_id,
            model_frontier=model_frontier,
            model_efficient=model_efficient,
            model_economy=model_economy,
            embedding_model=embedding_model,
            recursion_limit=self.recursion_limit,
            checkpoint_db=self.checkpoint_db,
            class_contract_policy=class_contract_policy,
            workspace_root=self.workspace_root,
        )

    def default_request_file(self, repo_root: Path) -> Path:
        path = Path(self.default_request_path)
        return path if path.is_absolute() else repo_root / path

    def state_store_path(self, repo_root: Path) -> Path:
        path = Path(self.state_store_root)
        return path if path.is_absolute() else repo_root / path

    def checkpoint_path(self, repo_root: Path) -> Path:
        path = Path(self.checkpoint_db)
        return path if path.is_absolute() else repo_root / path


def _get_env_int(name: str, default: int, minimum: int, maximum: int = 10_000_000) -> int:
    """Parse an integer from an environment variable with bounds checking.

    Args:
        name: Environment variable name.
        default: Value to return if the variable is unset.
        minimum: Inclusive lower bound.
        maximum: Inclusive upper bound (default 10M, prevents absurd values).

    Returns:
        The parsed integer, guaranteed to be within [minimum, maximum].

    Raises:
        ValueError: If the value is not an integer or is outside bounds.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {raw!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got: {parsed}")
    if parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got: {parsed}")
    return parsed
