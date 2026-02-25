from __future__ import annotations

import json
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

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class RoleBinding:
    config: AgentConfig
    model_name: str


def load_stage_role_bindings(*, stage: str, settings: RuntimeSettings) -> dict[str, RoleBinding]:
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
    def __init__(self, *, stage: str, settings: RuntimeSettings, backend_root: Path) -> None:
        self.stage = stage
        self.settings = settings
        self.backend_root = backend_root
        self.bindings = load_stage_role_bindings(stage=stage, settings=settings)

    def require_roles(self, required_roles: list[str]) -> None:
        missing = sorted(role for role in required_roles if role not in self.bindings)
        if missing:
            raise ValueError(f"Stage '{self.stage}' missing required role configs: {', '.join(missing)}")

    def binding_for(self, role: str) -> RoleBinding:
        if role not in self.bindings:
            available = ", ".join(sorted(self.bindings))
            raise ValueError(f"Unknown role '{role}' for stage '{self.stage}'. Available: {available}")
        return self.bindings[role]

    def role_context_line(self, role: str) -> str:
        binding = self.binding_for(role)
        allowed = ", ".join(binding.config.allowed_context_sections) or "none"
        return (
            f"Role={binding.config.role}; stage={binding.config.stage}; context_policy={binding.config.context_policy}; "
            f"allowed_context_sections=[{allowed}]; max_context_tokens={binding.config.max_context_tokens}."
        )

    def structured_adapter(self, *, role: str, schema: type[ModelT]) -> StructuredOutputAdapter[ModelT]:
        binding = self.binding_for(role)
        return get_structured_chat_model(
            model_name=binding.model_name,
            schema=schema,
            temperature=binding.config.temperature,
            method="function_calling",
            strict=True,
            include_raw=False,
        )

    def invoke_structured(self, *, role: str, schema: type[ModelT], prompt: str) -> ModelT:
        adapter = self.structured_adapter(role=role, schema=schema)
        return adapter.invoke(prompt)

    def invoke_text(self, *, role: str, prompt: str) -> str:
        binding = self.binding_for(role)
        model = get_chat_model(model_name=binding.model_name, temperature=binding.config.temperature)
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
        binding = self.binding_for(role)
        model = get_chat_model(model_name=binding.model_name, temperature=binding.config.temperature)
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
