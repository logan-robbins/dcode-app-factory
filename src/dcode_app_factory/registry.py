from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .models import MicroModuleContract
from .utils import slugify_name


@dataclass(frozen=True)
class CodeIndexEntry:
    module_slug: str
    contract_fingerprint: str
    contract: MicroModuleContract


class CodeIndex:
    """Append-only in-memory index keyed by module slug."""

    def __init__(self) -> None:
        self._entries: Dict[str, CodeIndexEntry] = {}

    def register(self, contract: MicroModuleContract) -> str:
        slug = slugify_name(contract.name)
        if slug in self._entries:
            raise ValueError(f"Contract slug already exists: {slug}")
        self._entries[slug] = CodeIndexEntry(
            module_slug=slug,
            contract_fingerprint=contract.fingerprint,
            contract=contract,
        )
        return slug

    def get(self, name_or_slug: str) -> MicroModuleContract | None:
        slug = slugify_name(name_or_slug)
        entry = self._entries.get(slug)
        return entry.contract if entry else None

    def list_entries(self) -> list[CodeIndexEntry]:
        return list(self._entries.values())

    def items(self) -> list[tuple[str, MicroModuleContract]]:
        """Return (slug, contract) pairs for iteration."""
        return [(e.module_slug, e.contract) for e in self._entries.values()]

    def __len__(self) -> int:
        return len(self._entries)
