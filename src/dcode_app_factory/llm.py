from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)
StructuredOutputMethod = Literal["function_calling", "json_mode", "json_schema"]

_DEFAULT_TIMEOUT: int = 120
_DEFAULT_MAX_RETRIES: int = 3


class SupportsInvoke(Protocol):
    """Protocol for any LangChain-compatible runnable that supports invoke."""

    def invoke(self, input: Any) -> Any:  # noqa: ANN401 - external runnable protocol.
        ...


@dataclass(slots=True)
class StructuredOutputAdapter(Generic[ModelT]):
    """Adapter that wraps a structured-output runnable and validates the response.

    Calls the underlying LLM runnable and normalizes the raw output into the
    declared Pydantic schema, handling both direct schema instances and
    ``include_raw=True`` envelope shapes.
    """

    schema: type[ModelT]
    runnable: SupportsInvoke

    def invoke(self, prompt: str) -> ModelT:
        """Invoke the LLM and return a validated Pydantic model instance.

        Args:
            prompt: The user prompt to send to the LLM.

        Returns:
            An instance of the declared schema type.

        Raises:
            RuntimeError: If the LLM returns unparseable or invalid output.
        """
        raw_output = self.runnable.invoke(prompt)
        return normalize_structured_output(raw_output=raw_output, schema=self.schema)


def ensure_openai_api_key(repo_root: Path | None = None) -> str:
    """Load OPENAI_API_KEY from environment or .env and return it.

    Searches the environment first, then falls back to a ``.env`` file at the
    given ``repo_root`` (or cwd if not specified).

    Args:
        repo_root: Optional repo root path to search for .env file.

    Returns:
        The API key string.

    Raises:
        RuntimeError: If OPENAI_API_KEY is unavailable after all sources are checked.
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
    timeout: int = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    max_completion_tokens: int | None = None,
    repo_root: Path | None = None,
) -> ChatOpenAI:
    """Construct a ChatOpenAI instance with validated API key and production defaults.

    Args:
        model_name: OpenAI model identifier (e.g. 'gpt-4o', 'gpt-4o-mini').
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts on transient failures.
        max_completion_tokens: Maximum tokens for the completion response.
            When None, the model default is used.
        repo_root: Optional repo root for .env file resolution.

    Returns:
        Configured ChatOpenAI instance.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not available.
    """
    if not model_name or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    ensure_openai_api_key(repo_root=repo_root)
    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens
    return ChatOpenAI(**kwargs)


def normalize_structured_output(*, raw_output: Any, schema: type[ModelT]) -> ModelT:
    """Normalize raw LLM structured output into a validated Pydantic model instance.

    Handles three input shapes:
    1. ``include_raw=True`` envelope: ``{"parsed": ..., "parsing_error": ..., "raw": ...}``
    2. Direct Pydantic BaseModel instance (same or different schema)
    3. Plain dict

    Args:
        raw_output: The raw output from the LLM or structured runnable.
        schema: The target Pydantic model class.

    Returns:
        A validated instance of ``schema``.

    Raises:
        RuntimeError: If the output cannot be parsed or validated against the schema.
    """
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
    timeout: int = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    max_completion_tokens: int | None = None,
    method: StructuredOutputMethod = "function_calling",
    strict: bool = True,
    include_raw: bool = False,
    repo_root: Path | None = None,
) -> StructuredOutputAdapter[ModelT]:
    """Build a StructuredOutputAdapter that invokes the LLM with schema-constrained output.

    Uses ``ChatOpenAI.with_structured_output`` to bind the schema to the model,
    ensuring responses conform to the Pydantic schema.

    Args:
        model_name: OpenAI model identifier.
        schema: Pydantic model class for structured output.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts on transient failures.
        max_completion_tokens: Maximum tokens for the completion response.
        method: Structured output method ('function_calling', 'json_mode', 'json_schema').
        strict: Enable strict schema enforcement (not valid for json_mode).
        include_raw: Whether to include raw LLM output alongside parsed result.
        repo_root: Optional repo root for .env file resolution.

    Returns:
        StructuredOutputAdapter configured to return validated schema instances.

    Raises:
        ValueError: If strict=True with method='json_mode'.
        RuntimeError: If OPENAI_API_KEY is not available.
    """
    if method == "json_mode" and strict:
        raise ValueError("strict=True is not valid for method='json_mode'")

    model = get_chat_model(
        model_name=model_name,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        max_completion_tokens=max_completion_tokens,
        repo_root=repo_root,
    )
    runnable = model.with_structured_output(
        schema,
        method=method,
        include_raw=include_raw,
        strict=strict if method != "json_mode" else None,
    )
    return StructuredOutputAdapter(schema=schema, runnable=runnable)
