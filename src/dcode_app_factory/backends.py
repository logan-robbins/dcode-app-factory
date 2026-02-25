"""Storage-level enforcement backends for the factory pipeline.

These backends form the security boundary of the agent harness. They wrap the
raw ``FilesystemBackend`` to enforce three invariants:

1. **Immutable artifacts** -- Once a module version is shipped (marked with a
   ``.immutable`` sentinel), no further writes are permitted.  Artifact
   directories (``/artifacts/``) are write-once by design.
2. **Opaque sealed modules** -- After a module version is sealed (marked with a
   ``.sealed`` sentinel), implementation and test directories are hidden from
   all reads, greps, downloads, and directory listings.
3. **Context-pack scoped access** -- Each agent receives a ``ContextPack``
   that restricts which files are visible based on role-specific access levels
   (FULL, CONTRACT_ONLY, SUMMARY_ONLY, METADATA_ONLY).

Composition order (innermost to outermost)::

    FilesystemBackend -> OpaqueEnforcementBackend -> ImmutableArtifactBackend -> [ContextPackBackend] -> CompositeBackend
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from deepagents.backends import CompositeBackend, FilesystemBackend
from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)

from .models import BoundaryLevel, ContextAccessLevel, ContextPack

logger = logging.getLogger(__name__)


SEALED_ACCESS_ERROR_TEMPLATE = (
    "ACCESS DENIED: '{path}' is a sealed implementation directory. "
    "This module is a closed-source service — integrate via contract.json, examples.md, and ship.json only. "
    "See §12 (Opaque Implementation Rule) and §6.8 (Black-Box Integration) for rationale. "
    "If the contract is insufficient, raise an Interface Change Exception (§13)."
)


class BlockedAccessError(RuntimeError):
    """Raised when storage-level enforcement blocks a path access.

    Available for callers that need to programmatically distinguish blocked
    access from other error conditions.  The backend methods themselves return
    error strings/results per the ``BackendProtocol`` contract rather than
    raising this exception.
    """


@dataclass(frozen=True)
class BackendRoutes:
    """Canonical route prefixes for the factory state store directory layout.

    Each prefix maps to a ``CompositeBackend`` route that dispatches file
    operations to the enforcement-wrapped backend.
    """

    root: str = "/"
    product: str = "/product/"
    project: str = "/project/"
    tasks: str = "/tasks/"
    artifacts: str = "/artifacts/"
    modules: str = "/modules/"
    debates: str = "/debates/"
    context_packs: str = "/context_packs/"
    exceptions: str = "/exceptions/"
    escalations: str = "/escalations/"
    code_index: str = "/code_index/"
    release: str = "/release/"


def _module_version_root(path: Path) -> Path | None:
    """Extract the module version root directory from a resolved path.

    Given a path like ``/root/modules/MM-foo/1.0.0/implementation/main.py``,
    returns ``/root/modules/MM-foo/1.0.0``.

    Args:
        path: Resolved filesystem path that may be inside a module version
            directory.

    Returns:
        The module version root path, or ``None`` if the path is not inside a
        ``modules/{module_id}/{version}/`` directory tree.
    """
    parts = list(path.parts)
    if "modules" not in parts:
        return None
    idx = parts.index("modules")
    if len(parts) <= idx + 2:
        return None
    # /.../modules/{module_id}/{version}/...
    return Path(*parts[: idx + 3])


def _is_sealed_implementation_path(path: Path) -> bool:
    """Return True if *path* points inside a sealed module's implementation or test tree.

    A module version is sealed when a ``.sealed`` sentinel file exists at the
    version root (e.g. ``modules/MM-foo/1.0.0/.sealed``).  Once sealed,
    ``/implementation/`` and ``/tests/`` subtrees are opaque.

    Args:
        path: Resolved filesystem path to check.

    Returns:
        ``True`` when the path is inside a sealed module's private
        implementation or test directory; ``False`` otherwise.
    """
    normalized = str(path)
    if "/modules/" not in normalized:
        return False
    if "/implementation/" not in normalized and "/tests/" not in normalized:
        return False
    version_root = _module_version_root(path)
    if version_root is None:
        return False
    return (version_root / ".sealed").is_file()


def _is_immutable_module_path(path: Path) -> bool:
    """Return True if *path* is inside a module version marked immutable.

    A module version becomes immutable when a ``.immutable`` sentinel file
    exists at the version root (created by ``mark_module_immutable`` at ship
    time).

    Args:
        path: Resolved filesystem path to check.

    Returns:
        ``True`` when the path is inside an immutable module version
        directory; ``False`` otherwise.
    """
    version_root = _module_version_root(path)
    if version_root is None:
        return False
    return (version_root / ".immutable").is_file()


def _is_allowed_public_surface(path: Path) -> bool:
    """Return True if *path* names a file in the public API surface of a module.

    These files are always visible even when the module is sealed, because
    downstream agents need them to integrate via contracts.
    """
    return path.name in {"contract.json", "examples.md", "ship.json", "envelope.json"}


def _contract_scope_for_path(file_path: str) -> tuple[BoundaryLevel | None, str | None]:
    """Extract the boundary level and contract reference from a file path.

    Inspects the path components to determine which contract hierarchy level
    the file belongs to and constructs a ``module_id@version`` reference.

    Hierarchy (checked in order of specificity):

    * ``system_contracts/{id}/{version}/...`` -> ``L2_SYSTEM``
    * ``service_contracts/{id}/{version}/...`` -> ``L3_SERVICE``
    * ``modules/{id}/{version}/...`` -> ``L4_COMPONENT``
    * ``class_contracts/{id}/{version}/...`` -> ``L5_CLASS``

    Args:
        file_path: Virtual path to inspect.

    Returns:
        A tuple of ``(BoundaryLevel, contract_ref)`` where ``contract_ref``
        is formatted as ``{id}@{version}``, or ``(None, None)`` if the path
        does not match any known contract hierarchy.
    """
    parts = [part for part in Path(file_path).parts if part not in {"/", ""}]
    if "system_contracts" in parts:
        idx = parts.index("system_contracts")
        if len(parts) > idx + 2:
            return BoundaryLevel.L2_SYSTEM, f"{parts[idx + 1]}@{parts[idx + 2]}"
    if "service_contracts" in parts:
        idx = parts.index("service_contracts")
        if len(parts) > idx + 2:
            return BoundaryLevel.L3_SERVICE, f"{parts[idx + 1]}@{parts[idx + 2]}"
    if "modules" in parts:
        idx = parts.index("modules")
        if len(parts) > idx + 2:
            return BoundaryLevel.L4_COMPONENT, f"{parts[idx + 1]}@{parts[idx + 2]}"
    if "class_contracts" in parts:
        idx = parts.index("class_contracts")
        if len(parts) > idx + 2:
            return BoundaryLevel.L5_CLASS, f"{parts[idx + 1]}@{parts[idx + 2]}"
    return None, None


class ImmutableArtifactBackend(BackendProtocol):
    """Blocks in-place modifications for shipped artifacts and immutable modules.

    Enforcement rules:

    * **Artifact paths** (``/artifacts/...``): Write-once.  Any existing file
      under an artifact directory is immutable -- artifacts are append-only
      records of the debate and ship pipeline.
    * **Module paths** (``/modules/...``): Mutable during development.
      Becomes immutable only after shipping, indicated by the presence of a
      ``.immutable`` sentinel file at the module version root (created by
      ``mark_module_immutable``).

    Read operations (``read``, ``ls_info``, ``grep_raw``, ``glob_info``,
    ``download_files``) are always permitted -- immutability only constrains
    writes.
    """

    _IMMUTABLE_ERROR = "Cannot modify immutable shipped artifact: {path}"

    def __init__(self, backend: BackendProtocol) -> None:
        self._backend = backend

    def _resolve(self, file_path: str) -> Path:
        """Resolve a virtual path to an absolute filesystem path.

        Delegates to the inner backend's ``_resolve_path`` when available,
        falling back to a plain ``Path`` construction.
        """
        resolver = getattr(self._backend, "_resolve_path", None)
        if callable(resolver):
            return resolver(file_path)
        return Path(file_path)

    def _is_immutable_target(self, path: str) -> bool:
        """Determine whether *path* refers to an immutable file.

        * Under ``/artifacts/``: immutable if the resolved file already exists
          (artifacts are write-once by design).
        * Under ``/modules/``: immutable only if a ``.immutable`` sentinel
          exists at the module version root (set at ship time).
        * All other paths: always mutable through this backend.

        Args:
            path: Virtual path to check.

        Returns:
            ``True`` if the path is protected from writes.
        """
        resolved = self._resolve(path)
        if "/artifacts/" in path:
            return resolved.exists()
        if "/modules/" in path:
            return _is_immutable_module_path(resolved)
        return False

    def ls_info(self, path: str) -> list[FileInfo]:
        """Delegate to inner backend -- reads are unrestricted."""
        return self._backend.ls_info(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Delegate to inner backend -- reads are unrestricted."""
        return self._backend.read(file_path, offset=offset, limit=limit)

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch] | str:
        """Delegate to inner backend -- reads are unrestricted."""
        return self._backend.grep_raw(pattern, path=path, glob=glob)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Delegate to inner backend -- reads are unrestricted."""
        return self._backend.glob_info(pattern, path=path)

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write a file, blocking if the target is immutable.

        Returns:
            ``WriteResult`` with an error message if the target is immutable,
            or the result from the inner backend on success.
        """
        if self._is_immutable_target(file_path):
            msg = self._IMMUTABLE_ERROR.format(path=file_path)
            logger.warning(msg)
            return WriteResult(error=msg)
        return self._backend.write(file_path, content)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        """Edit a file, blocking if the target is immutable.

        Returns:
            ``EditResult`` with an error message if the target is immutable,
            or the result from the inner backend on success.
        """
        if self._is_immutable_target(file_path):
            msg = self._IMMUTABLE_ERROR.format(path=file_path)
            logger.warning(msg)
            return EditResult(error=msg)
        return self._backend.edit(file_path, old_string, new_string, replace_all=replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files, blocking any that target immutable paths.

        If *any* file in the batch targets an immutable path, the entire batch
        is rejected to maintain atomicity.

        Returns:
            List of ``FileUploadResponse`` -- one per input file.
        """
        blocked = [path for path, _ in files if self._is_immutable_target(path)]
        if blocked:
            return [
                FileUploadResponse(path=path, error=self._IMMUTABLE_ERROR.format(path=path))
                for path in blocked
            ]
        return self._backend.upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Delegate to inner backend -- reads are unrestricted."""
        return self._backend.download_files(paths)


class OpaqueEnforcementBackend(BackendProtocol):
    """Blocks reads of sealed implementation/test directories post-ship.

    Once a module version is sealed (indicated by a ``.sealed`` sentinel at the
    module version root), the ``/implementation/`` and ``/tests/`` subtrees
    become opaque:

    * ``read()`` returns a structured denial message instead of file content.
    * ``grep_raw()`` and ``glob_info()`` strip sealed paths from results.
    * ``ls_info()`` hides sealed entries while preserving public surface files.
    * ``write()``, ``edit()``, ``upload_files()`` return errors for sealed paths.
    * ``download_files()`` returns errors for sealed paths.

    Public surface files (``contract.json``, ``examples.md``, ``ship.json``,
    ``envelope.json``) remain accessible even when the module is sealed,
    because they are at the module version root (not under ``implementation/``
    or ``tests/``).
    """

    def __init__(self, backend: BackendProtocol) -> None:
        self._backend = backend

    def _resolve(self, file_path: str) -> Path:
        """Resolve a virtual path to an absolute filesystem path."""
        resolver = getattr(self._backend, "_resolve_path", None)
        if callable(resolver):
            return resolver(file_path)
        return Path(file_path)

    def _blocked_message(self, path: str) -> str:
        """Format the denial message for a sealed path."""
        return SEALED_ACCESS_ERROR_TEMPLATE.format(path=path)

    def _block_if_sealed(self, file_path: str) -> str | None:
        """Check if *file_path* is inside a sealed implementation/test tree.

        Returns:
            A denial message string if the path is blocked, ``None`` if access
            is permitted.
        """
        resolved = self._resolve(file_path)
        if _is_sealed_implementation_path(resolved):
            logger.info("Blocked access to sealed path: %s", file_path)
            return self._blocked_message(file_path)
        return None

    def ls_info(self, path: str) -> list[FileInfo]:
        """List directory contents, filtering out sealed implementation entries.

        Public surface files (``contract.json``, etc.) are preserved even when
        adjacent to sealed directories.
        """
        infos = self._backend.ls_info(path)
        filtered: list[FileInfo] = []
        for info in infos:
            info_path = str(info.get("path", ""))
            if not info_path:
                continue
            resolved = self._resolve(info_path)
            if _is_sealed_implementation_path(resolved) and not _is_allowed_public_surface(resolved):
                continue
            filtered.append(info)
        return filtered

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read file content, returning a denial message for sealed paths.

        The denial message is returned as the read content (not raised as an
        exception) to conform to the ``BackendProtocol.read`` contract which
        returns error strings.
        """
        blocked = self._block_if_sealed(file_path)
        if blocked:
            return blocked
        return self._backend.read(file_path, offset=offset, limit=limit)

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch] | str:
        """Search files, excluding any matches inside sealed directories."""
        matches = self._backend.grep_raw(pattern, path=path, glob=glob)
        if isinstance(matches, str):
            return matches
        filtered: list[GrepMatch] = []
        for match in matches:
            match_path = str(match.get("path", ""))
            resolved = self._resolve(match_path)
            if _is_sealed_implementation_path(resolved):
                continue
            filtered.append(match)
        return filtered

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Find files matching a glob, excluding sealed implementation entries.

        Public surface files within sealed modules are still returned.
        """
        infos = self._backend.glob_info(pattern, path=path)
        filtered: list[FileInfo] = []
        for info in infos:
            info_path = str(info.get("path", ""))
            resolved = self._resolve(info_path)
            if _is_sealed_implementation_path(resolved) and not _is_allowed_public_surface(resolved):
                continue
            filtered.append(info)
        return filtered

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write a file, blocking writes to sealed paths."""
        blocked = self._block_if_sealed(file_path)
        if blocked:
            return WriteResult(error=blocked)
        return self._backend.write(file_path, content)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        """Edit a file, blocking edits to sealed paths."""
        blocked = self._block_if_sealed(file_path)
        if blocked:
            return EditResult(error=blocked)
        return self._backend.edit(file_path, old_string, new_string, replace_all=replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files, blocking any that target sealed paths.

        If any file in the batch targets a sealed path, the entire batch is
        rejected.
        """
        blocked_paths = [path for path, _ in files if self._block_if_sealed(path)]
        if blocked_paths:
            return [
                FileUploadResponse(path=path, error=self._blocked_message(path))
                for path in blocked_paths
            ]
        return self._backend.upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files, blocking downloads from sealed paths."""
        blocked = [path for path in paths if self._block_if_sealed(path)]
        if blocked:
            return [
                FileDownloadResponse(path=path, content=None, error=self._blocked_message(path))
                for path in blocked
            ]
        return self._backend.download_files(paths)


class ContextPackBackend(BackendProtocol):
    """Enforces role-specific access levels at the storage API boundary.

    Each agent in the factory pipeline receives a ``ContextPack`` that defines
    which files it may access and at what level.  This backend wraps the inner
    backend and enforces those permissions on every operation.

    Access levels (most to least permissive):

    * **FULL** -- unrestricted read and write access.
    * **CONTRACT_ONLY** -- read access limited to ``contract.json``,
      ``examples.md``, ``ship.json``, ``envelope.json``.
    * **SUMMARY_ONLY** -- read access limited to ``summary.md``,
      ``ship.json``, ``envelope.json``, ``contract.json``.
    * **METADATA_ONLY** -- read access limited to ``envelope.json``,
      ``metadata.json``.

    Write operations (``write``, ``edit``, ``upload_files``) require FULL
    access.  Read operations respect the filename allowlists defined by each
    access level.  Unknown access levels fail closed (deny).
    """

    # Filename allowlists per access level (most restrictive first).
    _READ_ALLOWLISTS: dict[ContextAccessLevel, frozenset[str]] = {
        ContextAccessLevel.METADATA_ONLY: frozenset({"envelope.json", "metadata.json"}),
        ContextAccessLevel.SUMMARY_ONLY: frozenset({"summary.md", "ship.json", "envelope.json", "contract.json"}),
        ContextAccessLevel.CONTRACT_ONLY: frozenset({"contract.json", "examples.md", "ship.json", "envelope.json"}),
    }

    def __init__(self, backend: BackendProtocol, context_pack: ContextPack) -> None:
        self._backend = backend
        self._context_pack = context_pack

    def _effective_access_level(self, file_path: str) -> ContextAccessLevel | None:
        """Compute the effective access level for *file_path*.

        Uses the full scoped-permission resolution path (boundary level +
        contract ref matching) to determine the most specific access level.

        Returns:
            The effective ``ContextAccessLevel``, or ``None`` if access is
            denied due to unresolved scoped permissions (level-aware
            permissions exist but none matched).
        """
        boundary_level, contract_ref = _contract_scope_for_path(file_path)
        path_permissions = self._context_pack.matching_permissions_for_path(file_path)
        scoped_permissions = self._context_pack.matching_permissions_for_ref(
            level=boundary_level,
            contract_ref=contract_ref,
            path=file_path,
        )
        if scoped_permissions:
            return scoped_permissions[0].access_level
        if any(permission.level is not None for permission in path_permissions):
            # Level-aware permissions are strict; no scoped match means deny.
            return None
        return self._context_pack.access_for_path(file_path)

    def _allowed(self, file_path: str) -> bool:
        """Determine whether *file_path* is readable under the context pack.

        Args:
            file_path: Virtual path to check.

        Returns:
            ``True`` if the file is accessible for reading at the current
            access level, ``False`` otherwise.
        """
        level = self._effective_access_level(file_path)
        if level is None:
            return False
        if level == ContextAccessLevel.FULL:
            return True
        allowlist = self._READ_ALLOWLISTS.get(level)
        if allowlist is None:
            # Unknown access level -- fail closed.
            logger.warning(
                "Unknown ContextAccessLevel %r for path %s; denying access",
                level,
                file_path,
            )
            return False
        return Path(file_path).name in allowlist

    def _has_write_access(self, file_path: str) -> bool:
        """Determine whether *file_path* is writable under the context pack.

        Write operations require the effective access level to be FULL.
        Uses the same scoped-permission resolution as ``_allowed`` to prevent
        privilege escalation through unscoped path matching.

        Args:
            file_path: Virtual path to check.

        Returns:
            ``True`` if the file is writable, ``False`` otherwise.
        """
        return self._effective_access_level(file_path) == ContextAccessLevel.FULL

    def _deny_message(self, file_path: str) -> str:
        """Format a structured denial message for *file_path*."""
        effective = self._effective_access_level(file_path)
        access = effective.value if effective is not None else "DENIED"
        boundary_level, contract_ref = _contract_scope_for_path(file_path)
        boundary_note = boundary_level.value if boundary_level is not None else "UNSCOPED"
        ref_note = contract_ref if contract_ref is not None else "N/A"
        return (
            f"Access denied by context pack {self._context_pack.cp_id}: {file_path} "
            f"(level={boundary_note}, contract_ref={ref_note}) requires higher than {access}"
        )

    def ls_info(self, path: str) -> list[FileInfo]:
        """List directory contents, filtering to only accessible files."""
        infos = self._backend.ls_info(path)
        return [info for info in infos if self._allowed(str(info.get("path", "")))]

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read file content, returning a denial message if access is restricted."""
        if not self._allowed(file_path):
            return self._deny_message(file_path)
        return self._backend.read(file_path, offset=offset, limit=limit)

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch] | str:
        """Search files, filtering results to only accessible paths."""
        matches = self._backend.grep_raw(pattern, path=path, glob=glob)
        if isinstance(matches, str):
            return matches
        return [match for match in matches if self._allowed(str(match.get("path", "")))]

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Find files matching a glob, filtered to accessible paths."""
        infos = self._backend.glob_info(pattern, path=path)
        return [info for info in infos if self._allowed(str(info.get("path", "")))]

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write a file, requiring FULL access level.

        Uses scoped permission resolution to prevent privilege escalation
        from unscoped path-level FULL access to scoped contract-ref paths.
        """
        if not self._has_write_access(file_path):
            msg = self._deny_message(file_path)
            logger.warning(msg)
            return WriteResult(error=msg)
        return self._backend.write(file_path, content)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        """Edit a file, requiring FULL access level."""
        if not self._has_write_access(file_path):
            msg = self._deny_message(file_path)
            logger.warning(msg)
            return EditResult(error=msg)
        return self._backend.edit(file_path, old_string, new_string, replace_all=replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files, requiring FULL access for every target path.

        If any file in the batch lacks FULL access, the entire batch is
        rejected.
        """
        blocked = [path for path, _ in files if not self._has_write_access(path)]
        if blocked:
            return [FileUploadResponse(path=path, error=self._deny_message(path)) for path in blocked]
        return self._backend.upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files, blocking any that the context pack does not allow."""
        blocked = [path for path in paths if not self._allowed(path)]
        if blocked:
            return [FileDownloadResponse(path=path, content=None, error=self._deny_message(path)) for path in blocked]
        return self._backend.download_files(paths)


def build_factory_backend(root_dir: Path, context_pack: ContextPack | None = None) -> BackendProtocol:
    """Construct the fully-composed factory backend with all enforcement layers.

    Composition order (innermost to outermost)::

        FilesystemBackend(root_dir, virtual_mode=True)
            -> OpaqueEnforcementBackend  (blocks reads of sealed impl/tests)
            -> ImmutableArtifactBackend  (blocks writes to shipped artifacts)
            -> ContextPackBackend        (role-scoped access control, if context_pack provided)
            -> CompositeBackend          (route-based dispatch)

    All three enforcement backends are always active.  The ``ContextPackBackend``
    is added only when a ``context_pack`` is provided (agent-scoped operations).

    Args:
        root_dir: Filesystem root for the factory state store.  All virtual
            paths are resolved relative to this directory.
        context_pack: Optional role-scoped access pack.  When provided,
            the ``ContextPackBackend`` is added as the outermost enforcement
            layer.

    Returns:
        A ``CompositeBackend`` wrapping the enforcement chain, with routes
        for all factory directory prefixes.
    """
    base = FilesystemBackend(root_dir=root_dir, virtual_mode=True)
    opaque = OpaqueEnforcementBackend(base)
    immutable = ImmutableArtifactBackend(opaque)
    backend: BackendProtocol = immutable
    if context_pack is not None:
        backend = ContextPackBackend(backend, context_pack)

    routes = BackendRoutes()
    route_map = {
        routes.product: backend,
        routes.project: backend,
        routes.tasks: backend,
        routes.artifacts: backend,
        routes.modules: backend,
        routes.debates: backend,
        routes.context_packs: backend,
        routes.exceptions: backend,
        routes.escalations: backend,
        routes.code_index: backend,
        routes.release: backend,
    }
    return CompositeBackend(default=backend, routes=route_map)


def seal_module_version(module_version_dir: Path) -> None:
    """Seal a module version, making its implementation and tests opaque.

    Creates a ``.sealed`` sentinel file at the module version root.  Once
    sealed, the ``OpaqueEnforcementBackend`` will block all reads, greps, and
    directory listings of the ``/implementation/`` and ``/tests/`` subtrees.

    Public surface files (``contract.json``, ``examples.md``, ``ship.json``,
    ``envelope.json``) remain accessible.

    Args:
        module_version_dir: Path to the module version directory
            (e.g. ``modules/MM-foo/1.0.0/``).
    """
    module_version_dir.mkdir(parents=True, exist_ok=True)
    sentinel = module_version_dir / ".sealed"
    sentinel.write_text("sealed=true\n", encoding="utf-8")
    logger.info("Sealed module version: %s", module_version_dir)


def mark_module_immutable(module_version_dir: Path) -> None:
    """Mark a module version as immutable, preventing further writes.

    Creates a ``.immutable`` sentinel file at the module version root.  Once
    marked, the ``ImmutableArtifactBackend`` will block all write, edit, and
    upload operations targeting any file within this module version directory.

    This is called at ship time, after verification and sealing.

    Args:
        module_version_dir: Path to the module version directory
            (e.g. ``modules/MM-foo/1.0.0/``).
    """
    module_version_dir.mkdir(parents=True, exist_ok=True)
    sentinel = module_version_dir / ".immutable"
    sentinel.write_text("immutable=true\n", encoding="utf-8")
    logger.info("Marked module version immutable: %s", module_version_dir)
