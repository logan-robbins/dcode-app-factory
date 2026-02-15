"""Code index for micro modules.

The Code Index serves as an institutional memory of all micro modules
produced by the factory. It provides registration and lookup
capabilities based on module names. The registry is in‑memory for
simplification but could be persisted to disk or a database in a
complete implementation.
"""

from __future__ import annotations

from typing import Dict, Optional

from .models import MicroModuleContract
from .utils import slugify_name


class CodeIndex:
    """A simple registry for micro module contracts.

    The CodeIndex exposes methods to register a contract and query
    contracts by name or slug. It maintains a mapping of slugs to
    contracts and ensures that each slug is unique.
    """

    def __init__(self) -> None:
        self._contracts: Dict[str, MicroModuleContract] = {}

    def register(self, contract: MicroModuleContract) -> str:
        """Register a micro module contract and return its slug.

        Args:
            contract: The contract to register.

        Returns:
            The slugified identifier for the contract.

        Raises:
            ValueError: If a contract with the same slug already exists.
        """
        slug = slugify_name(contract.name)
        if slug in self._contracts:
            existing = self._contracts[slug]
            raise ValueError(
                f"Contract with slug '{slug}' already exists for module '{existing.name}'"
            )
        self._contracts[slug] = contract
        return slug

    def get(self, name_or_slug: str) -> Optional[MicroModuleContract]:
        """Retrieve a contract by its name or slug.

        Args:
            name_or_slug: The module name or slug.

        Returns:
            The corresponding contract if found, else ``None``.
        """
        slug = slugify_name(name_or_slug)
        return self._contracts.get(slug)

    def __len__(self) -> int:
        return len(self._contracts)

    def __contains__(self, name_or_slug: str) -> bool:
        return self.get(name_or_slug) is not None

    def items(self):
        """Return iterator over (slug, contract) pairs."""
        return self._contracts.items()
