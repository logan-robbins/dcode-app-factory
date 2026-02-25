from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from deepagents import create_deep_agent
from pydantic import BaseModel

from .backends import build_factory_backend
from .llm import StructuredOutputAdapter, get_chat_model, get_structured_chat_model
from .model_selection import RuntimeModelSelection, resolve_agent_models
from .models import AgentConfig
from .settings import RuntimeSettings
from .utils import get_agent_config_dir, load_agent_config

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class RoleBinding:
    """Binds an agent config to a resolved model name for a specific role."""

    config: AgentConfig
    model_name: str


def load_stage_role_bindings(*, stage: str, settings: RuntimeSettings) -> dict[str, RoleBinding]:
    """Load all agent config JSONs for a stage and resolve each to a RoleBinding.

    Scans the agent_configs/{stage}/ directory for JSON files, validates each
    config against the stage, checks for duplicates, and resolves model tiers
    to concrete model names via RuntimeModelSelection.

    Args:
        stage: Pipeline stage name (e.g. 'engineering_loop', 'project_loop').
        settings: Runtime settings containing model tier-to-name mappings.

    Returns:
        Mapping of role name to RoleBinding.

    Raises:
        FileNotFoundError: If the agent config directory for the stage does not exist.
        ValueError: If configs are empty, have stage mismatches, or duplicate roles.
    """
    config_dir = get_agent_config_dir(stage)
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Agent config directory missing for stage '{stage}': {config_dir}")

    configs: dict[str, AgentConfig] = {}
    for path in sorted(config_dir.glob("*.json")):
        cfg = load_agent_config(path)
        if cfg.stage != stage:
            raise ValueError(f"Agent config stage mismatch: expected '{stage}', got '{cfg.stage}' in {path}")
        if cfg.role in configs:
            raise ValueError(f"Duplicate role config for stage '{stage}': {cfg.role}")
        configs[cfg.role] = cfg

    if not configs:
        raise ValueError(f"No agent configs found for stage '{stage}' under {config_dir}")

    model_selection = RuntimeModelSelection(
        by_tier={
            "frontier": settings.model_frontier,
            "efficient": settings.model_efficient,
            "economy": settings.model_economy,
        }
    )
    resolved_models = resolve_agent_models(configs, model_selection)
    return {role: RoleBinding(config=cfg, model_name=resolved_models[role]) for role, cfg in sorted(configs.items())}


def _content_to_text(content: Any) -> str:
    """Recursively extract plain text from heterogeneous LLM response content.

    Handles strings, lists of text/dict items, and nested content structures
    produced by various LLM response formats.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    chunks.append(text_value)
                    continue
                nested = item.get("content")
                if nested is not None:
                    chunks.append(_content_to_text(nested))
                    continue
                chunks.append(json.dumps(item, sort_keys=True))
                continue
            chunks.append(str(item))
        return "\n".join(chunk for chunk in chunks if chunk.strip())
    if isinstance(content, dict):
        if "content" in content:
            return _content_to_text(content["content"])
        return json.dumps(content, sort_keys=True)
    return str(content)


def extract_agent_text(response: Any) -> str:
    """Extract the final text content from an agent response.

    Navigates through various response shapes (dict with messages, output,
    content; objects with content attribute) to produce a single text string.

    Args:
        response: Raw agent response in any supported format.

    Returns:
        Extracted text string.
    """
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        if "messages" in response and isinstance(response["messages"], list) and response["messages"]:
            return extract_agent_text(response["messages"][-1])
        if "output" in response:
            return extract_agent_text(response["output"])
        if "content" in response:
            return _content_to_text(response["content"])
    content = getattr(response, "content", None)
    if content is not None:
        return _content_to_text(content)
    return _content_to_text(response)


def extract_json_payload(text: str) -> dict[str, Any]:
    """Extract a JSON object from agent text output.

    Attempts parsing in order: direct JSON, fenced code block, first/last brace extraction.

    Args:
        text: Raw text output from an agent.

    Returns:
        Parsed JSON dict.

    Raises:
        RuntimeError: If no valid JSON object can be extracted from the text.
    """
    body = text.strip()
    if not body:
        raise RuntimeError("Agent returned empty output; expected JSON object")

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, flags=re.DOTALL)
    if fenced is not None:
        try:
            payload = json.loads(fenced.group(1))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive parsing branch
            raise RuntimeError(f"Failed to parse fenced JSON payload: {exc}") from exc
        if isinstance(payload, dict):
            return payload

    start = body.find("{")
    end = body.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = body[start : end + 1]
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive parsing branch
            raise RuntimeError(f"Failed to parse extracted JSON payload: {exc}") from exc
        if isinstance(payload, dict):
            return payload

    preview = body[:220].replace("\n", " ")
    raise RuntimeError(f"Agent output did not contain a JSON object: {preview}")


class RoleAgentRuntime:
    """Runtime that binds agent roles to LLM models and provides invocation methods.

    Loads agent configs for a pipeline stage, resolves model tiers to concrete
    model names, and provides structured, text, and deep-agent invocation
    methods that always call the real LLM -- no shortcuts, no fallbacks.
    """

    def __init__(self, *, stage: str, settings: RuntimeSettings, backend_root: Path) -> None:
        self.stage = stage
        self.settings = settings
        self.backend_root = backend_root
        self.bindings = load_stage_role_bindings(stage=stage, settings=settings)

    def require_roles(self, required_roles: list[str]) -> None:
        """Validate that all required roles have configs loaded for this stage.

        Args:
            required_roles: List of role names that must be present.

        Raises:
            ValueError: If any required role is missing from the loaded bindings.
        """
        missing = sorted(role for role in required_roles if role not in self.bindings)
        if missing:
            raise ValueError(f"Stage '{self.stage}' missing required role configs: {', '.join(missing)}")

    def binding_for(self, role: str) -> RoleBinding:
        """Return the RoleBinding for the given role, or raise if unknown.

        Args:
            role: Agent role name.

        Returns:
            The RoleBinding containing the AgentConfig and resolved model name.

        Raises:
            ValueError: If the role is not configured for this stage.
        """
        if role not in self.bindings:
            available = ", ".join(sorted(self.bindings))
            raise ValueError(f"Unknown role '{role}' for stage '{self.stage}'. Available: {available}")
        return self.bindings[role]

    def role_context_line(self, role: str) -> str:
        """Return a structured metadata string describing the role's context policy.

        Includes role name, stage, context policy, allowed sections, and token budget.

        Args:
            role: Agent role name.

        Returns:
            Formatted context metadata string for prompt injection.
        """
        binding = self.binding_for(role)
        allowed = ", ".join(binding.config.allowed_context_sections) or "none"
        return (
            f"Role={binding.config.role}; stage={binding.config.stage}; context_policy={binding.config.context_policy}; "
            f"allowed_context_sections=[{allowed}]; max_context_tokens={binding.config.max_context_tokens}."
        )

    def structured_adapter(self, *, role: str, schema: type[ModelT]) -> StructuredOutputAdapter[ModelT]:
        """Build a StructuredOutputAdapter for the given role and Pydantic schema.

        Configures the LLM with function_calling structured output, strict mode,
        and the role's configured temperature and token limits.

        Args:
            role: Agent role name.
            schema: Pydantic model class for structured output.

        Returns:
            Configured StructuredOutputAdapter.
        """
        binding = self.binding_for(role)
        return get_structured_chat_model(
            model_name=binding.model_name,
            schema=schema,
            temperature=binding.config.temperature,
            max_completion_tokens=binding.config.max_context_tokens,
            method="function_calling",
            strict=True,
            include_raw=False,
        )

    def invoke_structured(self, *, role: str, schema: type[ModelT], prompt: str) -> ModelT:
        """Invoke the LLM for the given role and return a validated Pydantic object.

        This always calls the real LLM. There are no shortcut paths.

        Args:
            role: Agent role name.
            schema: Pydantic model class for structured output.
            prompt: The prompt to send to the LLM.

        Returns:
            A validated instance of the schema type.

        Raises:
            RuntimeError: If the LLM returns invalid or unparseable output.
            ValueError: If the role is not configured.
        """
        adapter = self.structured_adapter(role=role, schema=schema)
        return adapter.invoke(prompt)

    def invoke_text(self, *, role: str, prompt: str) -> str:
        """Invoke the LLM for the given role and return plain text.

        This always calls the real LLM. There are no shortcut paths.

        Args:
            role: Agent role name.
            prompt: The prompt to send to the LLM.

        Returns:
            The extracted text response, stripped of whitespace.

        Raises:
            ValueError: If the role is not configured.
        """
        binding = self.binding_for(role)
        model = get_chat_model(
            model_name=binding.model_name,
            temperature=binding.config.temperature,
            max_completion_tokens=binding.config.max_context_tokens,
        )
        response = model.invoke(prompt)
        return extract_agent_text(response).strip()

    def invoke_deepagent_json(
        self,
        *,
        role: str,
        system_prompt: str,
        user_message: str,
        tools: list[Any],
        name: str,
    ) -> dict[str, Any]:
        """Invoke a deep agent (tool-using agent) and extract a JSON payload from its output.

        Creates a deep agent via ``deepagents.create_deep_agent`` with the role's
        model, the provided tools, and a filesystem backend rooted at ``backend_root``.

        This always calls the real LLM. There are no shortcut paths.

        Args:
            role: Agent role name.
            system_prompt: System prompt for the deep agent.
            user_message: User message to send.
            tools: List of LangChain-compatible tools for the agent.
            name: Name identifier for the deep agent instance.

        Returns:
            Parsed JSON dict from the agent's final text output.

        Raises:
            RuntimeError: If the agent output contains no valid JSON.
            ValueError: If the role is not configured.
        """
        binding = self.binding_for(role)
        model = get_chat_model(
            model_name=binding.model_name,
            temperature=binding.config.temperature,
            max_completion_tokens=binding.config.max_context_tokens,
        )
        backend = build_factory_backend(self.backend_root)
        agent = create_deep_agent(
            model=model,
            tools=tools,
            backend=backend,
            system_prompt=system_prompt,
            name=name,
        )
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config={"configurable": {"thread_id": f"{self.stage}-{role}-{uuid.uuid4().hex[:8]}"}},
        )
        text = extract_agent_text(response)
        return extract_json_payload(text)
