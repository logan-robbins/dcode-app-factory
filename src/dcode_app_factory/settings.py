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
    default_spec_path: str = "SPEC.md"
    state_store_root: str = "state_store"
    project_id: str = "PROJECT-001"
    model_frontier: str = "gpt-4o"
    model_efficient: str = "gpt-4o-mini"
    model_economy: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    recursion_limit: int = 1_000
    checkpoint_db: str = "state_store/checkpoints/langgraph.sqlite"
    class_contract_policy: str = "selective_shared"

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            max_product_sections=_get_env_int("FACTORY_MAX_PRODUCT_SECTIONS", default=8, minimum=1),
            context_budget_floor_tokens=_get_env_int("FACTORY_CONTEXT_BUDGET_FLOOR", default=2_000, minimum=256),
            context_budget_cap_tokens=_get_env_int("FACTORY_CONTEXT_BUDGET_CAP", default=16_000, minimum=512),
            default_spec_path=os.getenv("FACTORY_DEFAULT_REQUEST_PATH", os.getenv("FACTORY_DEFAULT_SPEC_PATH", "SPEC.md")),
            state_store_root=os.getenv("FACTORY_STATE_STORE_ROOT", "state_store"),
            project_id=os.getenv("FACTORY_PROJECT_ID", "PROJECT-001"),
            model_frontier=os.getenv("FACTORY_MODEL_FRONTIER", "gpt-4o"),
            model_efficient=os.getenv("FACTORY_MODEL_EFFICIENT", "gpt-4o-mini"),
            model_economy=os.getenv("FACTORY_MODEL_ECONOMY", "gpt-4o-mini"),
            embedding_model=os.getenv("FACTORY_EMBEDDING_MODEL", "text-embedding-3-large"),
            recursion_limit=_get_env_int("FACTORY_RECURSION_LIMIT", default=1_000, minimum=100),
            checkpoint_db=os.getenv("FACTORY_CHECKPOINT_DB", "state_store/checkpoints/langgraph.sqlite"),
            class_contract_policy=os.getenv("FACTORY_CLASS_CONTRACT_POLICY", "selective_shared"),
        ).normalized()

    def normalized(self) -> "RuntimeSettings":
        cap = max(self.context_budget_cap_tokens, self.context_budget_floor_tokens)
        embedding_model = self.embedding_model.strip()
        if not embedding_model:
            raise ValueError("FACTORY_EMBEDDING_MODEL must be non-empty")
        class_contract_policy = self.class_contract_policy.strip().lower()
        if class_contract_policy not in {"selective_shared", "universal_public", "service_only"}:
            raise ValueError(
                "FACTORY_CLASS_CONTRACT_POLICY must be one of: selective_shared, universal_public, service_only"
            )
        return RuntimeSettings(
            max_product_sections=self.max_product_sections,
            context_budget_floor_tokens=self.context_budget_floor_tokens,
            context_budget_cap_tokens=cap,
            default_spec_path=self.default_spec_path,
            state_store_root=self.state_store_root,
            project_id=self.project_id,
            model_frontier=self.model_frontier,
            model_efficient=self.model_efficient,
            model_economy=self.model_economy,
            embedding_model=embedding_model,
            recursion_limit=self.recursion_limit,
            checkpoint_db=self.checkpoint_db,
            class_contract_policy=class_contract_policy,
        )

    def default_request_file(self, repo_root: Path) -> Path:
        path = Path(self.default_spec_path)
        return path if path.is_absolute() else repo_root / path

    def default_spec_file(self, repo_root: Path) -> Path:
        return self.default_request_file(repo_root)

    def state_store_path(self, repo_root: Path) -> Path:
        path = Path(self.state_store_root)
        return path if path.is_absolute() else repo_root / path

    def checkpoint_path(self, repo_root: Path) -> Path:
        path = Path(self.checkpoint_db)
        return path if path.is_absolute() else repo_root / path


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
