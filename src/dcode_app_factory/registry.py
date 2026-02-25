from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions

from .llm import ensure_openai_api_key
from .models import (
    BoundaryLevel,
    CodeIndexEntry,
    CodeIndexIoSummary,
    CodeIndexStatus,
    MicroModuleContract,
)
from .utils import slugify_name


DETERMINISTIC_MODEL_RE = re.compile(r"^deterministic-hash(?:-(\d+))?$")


def _normalize_embedding_model_name(value: str) -> tuple[str, str]:
    model = value.strip()
    if not model:
        raise ValueError("embedding_model must be non-empty")

    deterministic_match = DETERMINISTIC_MODEL_RE.match(model)
    if deterministic_match:
        dimensions = int(deterministic_match.group(1) or 384)
        if dimensions <= 0:
            raise ValueError(f"deterministic embedding dimensions must be > 0, got {dimensions}")
        return f"deterministic-hash-{dimensions}", "deterministic"

    if model.startswith("openai:"):
        model = model.split(":", 1)[1].strip()
    if not model:
        raise ValueError("OpenAI embedding model must be non-empty")
    return model, "openai"


def _build_embedding_function(*, model_name: str, provider: str) -> EmbeddingFunction[Documents]:
    if provider == "deterministic":
        match = DETERMINISTIC_MODEL_RE.match(model_name)
        if match is None:
            raise ValueError(f"Invalid deterministic embedding model: {model_name}")
        dimensions = int(match.group(1) or 384)
        return DeterministicEmbeddingFunction(dimensions=dimensions)

    api_key = ensure_openai_api_key()
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=model_name,
    )


class DeterministicEmbeddingFunction(EmbeddingFunction[Documents]):
    """Dense deterministic embedding function to avoid runtime model dependencies."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    @staticmethod
    def name() -> str:
        return "deterministic-hash"

    def get_config(self) -> dict[str, int]:
        return {"dimensions": self.dimensions}

    @staticmethod
    def build_from_config(config: dict[str, int]) -> "DeterministicEmbeddingFunction":
        return DeterministicEmbeddingFunction(dimensions=int(config.get("dimensions", 384)))

    def __call__(self, input: Documents) -> Embeddings:
        vectors: list[list[float]] = []
        for text in input:
            vec = [0.0] * self.dimensions
            for token in text.lower().split():
                index = hash(token) % self.dimensions
                vec[index] += 1.0
            norm = math.sqrt(sum(value * value for value in vec))
            if norm > 0:
                vec = [value / norm for value in vec]
            vectors.append(vec)
        return vectors


@dataclass(frozen=True)
class CodeIndexSearchResult:
    entry: CodeIndexEntry
    similarity_score: float


class CodeIndex:
    """Append-only Chroma-backed code index with semantic and metadata search."""

    def __init__(self, root: Path, *, embedding_model: str = "text-embedding-3-large") -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        normalized_model, provider = _normalize_embedding_model_name(embedding_model)
        self.embedding_model = normalized_model
        self.embedding_model_provider = provider
        self.embedding_model_path = self.root / "embedding_model.txt"
        self.embedding_function = _build_embedding_function(
            model_name=self.embedding_model,
            provider=self.embedding_model_provider,
        )
        self.client = chromadb.PersistentClient(path=str(self.root))
        self.collection = self.client.get_or_create_collection(
            name="code_index",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model},
        )
        self._sync_embedding_model()

    def _sync_embedding_model(self) -> None:
        previous_model: str | None = None
        if self.embedding_model_path.is_file():
            previous_model = self.embedding_model_path.read_text(encoding="utf-8").strip() or None
        self.embedding_model_path.write_text(f"{self.embedding_model}\n", encoding="utf-8")
        if previous_model and previous_model != self.embedding_model:
            reindexed = self._rebuild_collection()
            self._log_event(
                {
                    "event": "embedding_model_changed",
                    "from": previous_model,
                    "to": self.embedding_model,
                    "reindexed": reindexed,
                }
            )

    def _rebuild_collection(self) -> int:
        payload = self.collection.get(include=["metadatas"])
        metadatas = payload.get("metadatas", [])
        entries = [CodeIndexEntry.model_validate_json(str(metadata["raw_entry"])) for metadata in metadatas]

        self.client.delete_collection(name="code_index")
        self.collection = self.client.get_or_create_collection(
            name="code_index",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model},
        )

        if not entries:
            return 0

        self.collection.add(
            ids=[entry.module_ref for entry in entries],
            documents=[self._embedding_document(entry) for entry in entries],
            metadatas=[self._entry_metadata(entry) for entry in entries],
        )
        return len(entries)

    @staticmethod
    def _embedding_document(entry: CodeIndexEntry) -> str:
        io_json = json.dumps(entry.io_summary.model_dump(mode="json"), separators=(",", ":"), sort_keys=True)
        return f"{entry.purpose}\n{', '.join(entry.tags)}\n{io_json}"

    @staticmethod
    def _entry_metadata(entry: CodeIndexEntry) -> dict[str, str | int | float | bool]:
        return {
            "module_id": entry.module_id,
            "version": entry.version,
            "level": entry.level.value,
            "status": entry.status.value,
            "name": entry.name,
            "purpose": entry.purpose,
            "owner": entry.owner,
            "compatibility_type": entry.compatibility_type.value,
            "tags": json.dumps(entry.tags),
            "dependencies": json.dumps(entry.dependencies),
            "io_inputs": json.dumps(entry.io_summary.inputs),
            "io_outputs": json.dumps(entry.io_summary.outputs),
            "raw_entry": entry.model_dump_json(),
        }

    def _log_event(self, event: dict[str, object]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def add_entry(self, entry: CodeIndexEntry) -> str:
        existing = self.collection.get(ids=[entry.module_ref])
        if existing.get("ids"):
            raise ValueError(f"CodeIndex entry already exists: {entry.module_ref}")

        self.collection.add(
            ids=[entry.module_ref],
            documents=[self._embedding_document(entry)],
            metadatas=[self._entry_metadata(entry)],
        )
        self._log_event(
            {
                "event": "add",
                "module_ref": entry.module_ref,
                "status": entry.status.value,
                "embedding_model": self.embedding_model,
            }
        )
        return entry.module_ref

    def get_entry(self, module_ref: str) -> CodeIndexEntry | None:
        payload = self.collection.get(ids=[module_ref], include=["metadatas"])
        ids = payload.get("ids", [])
        if not ids:
            return None
        metadata = payload["metadatas"][0]
        return CodeIndexEntry.model_validate_json(str(metadata["raw_entry"]))

    def _matches_filters(
        self,
        entry: CodeIndexEntry,
        *,
        tags: list[str] | None,
        input_types: list[str] | None,
        output_types: list[str] | None,
        level: BoundaryLevel | None,
        include_inactive: bool,
    ) -> bool:
        if not include_inactive and entry.status in {CodeIndexStatus.DEPRECATED, CodeIndexStatus.SUPERSEDED}:
            return False
        if level is not None and entry.level != level:
            return False
        if tags and not set(tags).intersection(set(entry.tags)):
            return False
        if input_types and not set(input_types).intersection(set(entry.io_summary.inputs)):
            return False
        if output_types and not set(output_types).intersection(set(entry.io_summary.outputs)):
            return False
        return True

    def search(
        self,
        query: str,
        *,
        level: BoundaryLevel | None = None,
        tags: list[str] | None = None,
        input_types: list[str] | None = None,
        output_types: list[str] | None = None,
        include_inactive: bool = False,
        top_k: int = 10,
    ) -> list[CodeIndexSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        n_results = max(top_k * 5, 25)
        result = self.collection.query(query_texts=[query], n_results=n_results, include=["metadatas", "distances"])
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        ranked: list[CodeIndexSearchResult] = []
        for metadata, distance in zip(metadatas, distances):
            entry = CodeIndexEntry.model_validate_json(str(metadata["raw_entry"]))
            if not self._matches_filters(
                entry,
                level=level,
                tags=tags,
                input_types=input_types,
                output_types=output_types,
                include_inactive=include_inactive,
            ):
                continue
            # Chroma cosine distance: lower is better. Convert to bounded score.
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            ranked.append(CodeIndexSearchResult(entry=entry, similarity_score=score))

        ranked.sort(key=lambda item: item.similarity_score, reverse=True)
        return ranked[:top_k]

    def _assert_one_way_status_transition(self, old: CodeIndexStatus, new: CodeIndexStatus) -> None:
        allowed: dict[CodeIndexStatus, set[CodeIndexStatus]] = {
            CodeIndexStatus.CURRENT: {CodeIndexStatus.DEPRECATED, CodeIndexStatus.SUPERSEDED},
            CodeIndexStatus.DEPRECATED: set(),
            CodeIndexStatus.SUPERSEDED: set(),
        }
        if new not in allowed[old]:
            raise ValueError(f"Illegal code-index status transition: {old.value} -> {new.value}")

    def set_status(
        self,
        module_ref: str,
        *,
        status: CodeIndexStatus,
        superseded_by: str | None = None,
        deprecation_reason: str | None = None,
    ) -> CodeIndexEntry:
        entry = self.get_entry(module_ref)
        if entry is None:
            raise ValueError(f"Unknown module_ref: {module_ref}")
        old_status = entry.status
        self._assert_one_way_status_transition(old_status, status)

        entry.status = status
        entry.superseded_by = superseded_by
        entry.deprecation_reason = deprecation_reason
        self.collection.update(
            ids=[module_ref],
            documents=[self._embedding_document(entry)],
            metadatas=[self._entry_metadata(entry)],
        )
        self._log_event(
            {
                "event": "status_transition",
                "module_ref": module_ref,
                "from": old_status.value,
                "to": status.value,
                "superseded_by": superseded_by,
                "deprecation_reason": deprecation_reason,
            }
        )
        return entry

    def reindex(self) -> int:
        data = self.collection.get(include=["metadatas", "documents"])
        ids: list[str] = data.get("ids", [])
        metadatas: list[dict[str, str]] = data.get("metadatas", [])
        if not ids:
            return 0

        documents: list[str] = []
        new_metadatas: list[dict[str, str | int | float | bool]] = []
        for metadata in metadatas:
            entry = CodeIndexEntry.model_validate_json(str(metadata["raw_entry"]))
            documents.append(self._embedding_document(entry))
            new_metadatas.append(self._entry_metadata(entry))

        self.collection.update(ids=ids, documents=documents, metadatas=new_metadatas)
        self._log_event({"event": "reindex", "count": len(ids), "embedding_model": self.embedding_model})
        return len(ids)

    def list_entries(self, *, include_inactive: bool = True) -> list[CodeIndexEntry]:
        payload = self.collection.get(include=["metadatas"])
        entries: list[CodeIndexEntry] = []
        for metadata in payload.get("metadatas", []):
            entry = CodeIndexEntry.model_validate_json(str(metadata["raw_entry"]))
            if not include_inactive and entry.status in {CodeIndexStatus.DEPRECATED, CodeIndexStatus.SUPERSEDED}:
                continue
            entries.append(entry)
        entries.sort(key=lambda entry: entry.module_ref)
        return entries

    def register(self, contract: MicroModuleContract) -> str:
        module_ref = f"{contract.module_id}@{contract.module_version}"
        entry = CodeIndexEntry(
            module_ref=module_ref,
            module_id=contract.module_id,
            version=contract.module_version,
            level=BoundaryLevel.L4_COMPONENT,
            name=contract.name,
            purpose=contract.purpose,
            owner=contract.owner,
            tags=list(contract.tags),
            contract_ref=f"/modules/{contract.module_id}/{contract.module_version}/contract.json",
            examples_ref=contract.examples_ref,
            ship_ref=f"/modules/{contract.module_id}/{contract.module_version}/ship.json",
            io_summary=CodeIndexIoSummary(
                inputs=[f"{item.name}:{item.type}" for item in contract.inputs],
                outputs=[f"{item.name}:{item.type}" for item in contract.outputs],
                error_surfaces=[f"{item.name}:{item.surface}" for item in contract.error_surfaces],
                effects=[f"{item.type}:{item.target}" for item in contract.effects],
                modes=[
                    "sync" if contract.modes.sync else "",
                    "async" if contract.modes.async_mode else "",
                ],
            ),
            dependencies=[item.ref for item in contract.dependencies],
            compatibility_type=contract.compatibility_type,
            status=CodeIndexStatus.CURRENT,
            notes="",
        )
        self.add_entry(entry)
        return slugify_name(contract.name)

    def get(self, name_or_slug: str) -> MicroModuleContract | None:
        _ = name_or_slug
        return None

    def items(self) -> list[tuple[str, MicroModuleContract]]:
        return []

    def __len__(self) -> int:
        return len(self.list_entries())

    def iter_refs(self) -> Iterable[str]:
        for entry in self.list_entries():
            yield entry.module_ref
