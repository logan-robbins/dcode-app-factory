from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)
StructuredOutputMethod = Literal["function_calling", "json_mode", "json_schema"]


class SupportsInvoke(Protocol):
    def invoke(self, input: Any) -> Any:  # noqa: ANN401 - external runnable protocol.
        ...


@dataclass(slots=True)
class StructuredOutputAdapter(Generic[ModelT]):
    schema: type[ModelT]
    runnable: SupportsInvoke

    def invoke(self, prompt: str) -> ModelT:
        raw_output = self.runnable.invoke(prompt)
        return normalize_structured_output(raw_output=raw_output, schema=self.schema)


def ensure_openai_api_key(repo_root: Path | None = None) -> str:
    """Load OPENAI_API_KEY from environment or .env and return it.

    Raises:
        RuntimeError: if OPENAI_API_KEY is unavailable.
    """

    repo = repo_root if repo_root is not None else Path.cwd()
    env_path = repo / ".env"
    if env_path.is_file():
        load_dotenv(env_path)

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for agent runtime execution")
    return key


def get_chat_model(
    *,
    model_name: str,
    temperature: float = 0.0,
    timeout: int = 60,
    repo_root: Path | None = None,
) -> ChatOpenAI:
    ensure_openai_api_key(repo_root=repo_root)
    return ChatOpenAI(model=model_name, temperature=temperature, timeout=timeout)


def normalize_structured_output(*, raw_output: Any, schema: type[ModelT]) -> ModelT:
    payload = raw_output
    if isinstance(payload, dict) and "parsed" in payload and "parsing_error" in payload:
        parsing_error = payload.get("parsing_error")
        if parsing_error is not None:
            raise RuntimeError(
                f"Structured output parsing failed for {schema.__name__}: {parsing_error!r}"
            ) from parsing_error
        payload = payload.get("parsed")
        if payload is None:
            raise RuntimeError(
                f"Structured output returned no parsed payload for {schema.__name__}"
            )

    if isinstance(payload, schema):
        return payload

    if isinstance(payload, BaseModel):
        candidate = payload.model_dump(mode="json")
    elif isinstance(payload, dict):
        candidate = payload
    else:
        raise RuntimeError(
            f"Structured output for {schema.__name__} returned unsupported payload type "
            f"{type(payload).__name__}"
        )

    try:
        return schema.model_validate(candidate)
    except ValidationError as exc:
        raise RuntimeError(
            f"Structured output validation failed for {schema.__name__}: {exc}"
        ) from exc


def get_structured_chat_model(
    *,
    model_name: str,
    schema: type[ModelT],
    temperature: float = 0.0,
    timeout: int = 60,
    method: StructuredOutputMethod = "function_calling",
    strict: bool = True,
    include_raw: bool = False,
    repo_root: Path | None = None,
) -> StructuredOutputAdapter[ModelT]:
    if method == "json_mode" and strict:
        raise ValueError("strict=True is not valid for method='json_mode'")

    model = get_chat_model(
        model_name=model_name,
        temperature=temperature,
        timeout=timeout,
        repo_root=repo_root,
    )
    runnable = model.with_structured_output(
        schema,
        method=method,
        include_raw=include_raw,
        strict=strict if method != "json_mode" else None,
    )
    return StructuredOutputAdapter(schema=schema, runnable=runnable)
