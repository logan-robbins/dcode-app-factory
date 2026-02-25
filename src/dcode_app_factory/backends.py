from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


SEALED_ACCESS_ERROR_TEMPLATE = (
    "ACCESS DENIED: '{path}' is a sealed implementation directory. "
    "This module is a closed-source service — integrate via contract.json, examples.md, and ship.json only. "
    "See §12 (Opaque Implementation Rule) and §6.8 (Black-Box Integration) for rationale. "
    "If the contract is insufficient, raise an Interface Change Exception (§13)."
)


class BlockedAccessError(RuntimeError):
    """Raised when storage-level enforcement blocks a path read."""


@dataclass(frozen=True)
class BackendRoutes:
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
    parts = list(path.parts)
    if "modules" not in parts:
        return None
    idx = parts.index("modules")
    if len(parts) <= idx + 2:
        return None
    # /.../modules/{module_id}/{version}/...
    return Path(*parts[: idx + 3])


def _is_sealed_implementation_path(path: Path) -> bool:
    normalized = str(path)
    if "/modules/" not in normalized:
        return False
    if "/implementation/" not in normalized and "/tests/" not in normalized:
        return False
    version_root = _module_version_root(path)
    if version_root is None:
        return False
    return (version_root / ".sealed").is_file()


def _is_allowed_public_surface(path: Path) -> bool:
    return path.name in {"contract.json", "examples.md", "ship.json", "envelope.json"}


def _contract_scope_for_path(file_path: str) -> tuple[BoundaryLevel | None, str | None]:
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
    """Blocks in-place modifications for artifact/module paths once created."""

    def __init__(self, backend: BackendProtocol) -> None:
        self._backend = backend

    def _resolve(self, file_path: str) -> Path:
        resolver = getattr(self._backend, "_resolve_path", None)
        if callable(resolver):
            return resolver(file_path)
        return Path(file_path)

    def _is_immutable_target(self, path: str) -> bool:
        pure = self._resolve(path)
        if "/artifacts/" in path or "/modules/" in path:
            return pure.exists()
        return False

    def ls_info(self, path: str) -> list[FileInfo]:
        return self._backend.ls_info(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return self._backend.read(file_path, offset=offset, limit=limit)

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch] | str:
        return self._backend.grep_raw(pattern, path=path, glob=glob)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return self._backend.glob_info(pattern, path=path)

    def write(self, file_path: str, content: str) -> WriteResult:
        if self._is_immutable_target(file_path):
            return WriteResult(error=f"Cannot modify immutable shipped artifact: {file_path}")
        return self._backend.write(file_path, content)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        if self._is_immutable_target(file_path):
            return EditResult(error=f"Cannot modify immutable shipped artifact: {file_path}")
        return self._backend.edit(file_path, old_string, new_string, replace_all=replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        blocked = [path for path, _ in files if self._is_immutable_target(path)]
        if blocked:
            return [
                FileUploadResponse(path=path, error=f"Cannot modify immutable shipped artifact: {path}")
                for path in blocked
            ]
        return self._backend.upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return self._backend.download_files(paths)


class OpaqueEnforcementBackend(BackendProtocol):
    """Blocks reads of sealed implementation/test directories post-ship."""

    def __init__(self, backend: BackendProtocol) -> None:
        self._backend = backend

    def _blocked_message(self, path: str) -> str:
        return SEALED_ACCESS_ERROR_TEMPLATE.format(path=path)

    def _block_if_sealed(self, file_path: str) -> str | None:
        resolver = getattr(self._backend, "_resolve_path", None)
        resolved = resolver(file_path) if callable(resolver) else Path(file_path)
        if _is_sealed_implementation_path(resolved):
            return self._blocked_message(file_path)
        return None

    def ls_info(self, path: str) -> list[FileInfo]:
        infos = self._backend.ls_info(path)
        filtered: list[FileInfo] = []
        resolver = getattr(self._backend, "_resolve_path", None)
        for info in infos:
            info_path = str(info.get("path", ""))
            if not info_path:
                continue
            resolved = resolver(info_path) if callable(resolver) else Path(info_path)
            if _is_sealed_implementation_path(resolved) and not _is_allowed_public_surface(resolved):
                continue
            filtered.append(info)
        return filtered

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        blocked = self._block_if_sealed(file_path)
        if blocked:
            return blocked
        return self._backend.read(file_path, offset=offset, limit=limit)

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch] | str:
        matches = self._backend.grep_raw(pattern, path=path, glob=glob)
        if isinstance(matches, str):
            return matches
        filtered: list[GrepMatch] = []
        resolver = getattr(self._backend, "_resolve_path", None)
        for match in matches:
            match_path = str(match.get("path", ""))
            resolved = resolver(match_path) if callable(resolver) else Path(match_path)
            if _is_sealed_implementation_path(resolved):
                continue
            filtered.append(match)
        return filtered

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        infos = self._backend.glob_info(pattern, path=path)
        resolver = getattr(self._backend, "_resolve_path", None)
        return [
            info
            for info in infos
            if not _is_sealed_implementation_path(
                resolver(str(info.get("path", ""))) if callable(resolver) else Path(str(info.get("path", "")))
            )
            or _is_allowed_public_surface(
                resolver(str(info.get("path", ""))) if callable(resolver) else Path(str(info.get("path", "")))
            )
        ]

    def write(self, file_path: str, content: str) -> WriteResult:
        blocked = self._block_if_sealed(file_path)
        if blocked:
            return WriteResult(error=blocked)
        return self._backend.write(file_path, content)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        blocked = self._block_if_sealed(file_path)
        if blocked:
            return EditResult(error=blocked)
        return self._backend.edit(file_path, old_string, new_string, replace_all=replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        blocked_paths = [path for path, _ in files if self._block_if_sealed(path)]
        if blocked_paths:
            return [
                FileUploadResponse(path=path, error=self._blocked_message(path))
                for path in blocked_paths
            ]
        return self._backend.upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        blocked = [path for path in paths if self._block_if_sealed(path)]
        if blocked:
            return [
                FileDownloadResponse(path=path, content=None, error=self._blocked_message(path))
                for path in blocked
            ]
        return self._backend.download_files(paths)


class ContextPackBackend(BackendProtocol):
    """Enforces role-specific access levels at storage API boundary."""

    def __init__(self, backend: BackendProtocol, context_pack: ContextPack) -> None:
        self._backend = backend
        self._context_pack = context_pack

    def _allowed(self, file_path: str) -> bool:
        boundary_level, contract_ref = _contract_scope_for_path(file_path)
        path_permissions = self._context_pack.matching_permissions_for_path(file_path)
        scoped_permissions = self._context_pack.matching_permissions_for_ref(
            level=boundary_level,
            contract_ref=contract_ref,
            path=file_path,
        )
        if scoped_permissions:
            level = scoped_permissions[0].access_level
        elif any(permission.level is not None for permission in path_permissions):
            # Level-aware permissions are strict; no scoped match means deny.
            return False
        else:
            level = self._context_pack.access_for_path(file_path)
        path = Path(file_path)
        if level == ContextAccessLevel.FULL:
            return True
        if level == ContextAccessLevel.CONTRACT_ONLY:
            return path.name in {"contract.json", "examples.md", "ship.json", "envelope.json"}
        if level == ContextAccessLevel.SUMMARY_ONLY:
            return path.name in {"summary.md", "ship.json", "envelope.json", "contract.json"}
        if level == ContextAccessLevel.METADATA_ONLY:
            return path.name in {"envelope.json", "metadata.json"}
        return False

    def _deny_message(self, file_path: str) -> str:
        access = self._context_pack.access_for_path(file_path).value
        boundary_level, contract_ref = _contract_scope_for_path(file_path)
        boundary_note = boundary_level.value if boundary_level is not None else "UNSCOPED"
        ref_note = contract_ref if contract_ref is not None else "N/A"
        return (
            f"Access denied by context pack {self._context_pack.cp_id}: {file_path} "
            f"(level={boundary_note}, contract_ref={ref_note}) requires higher than {access}"
        )

    def ls_info(self, path: str) -> list[FileInfo]:
        infos = self._backend.ls_info(path)
        return [info for info in infos if self._allowed(str(info.get("path", "")))]

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        if not self._allowed(file_path):
            return self._deny_message(file_path)
        return self._backend.read(file_path, offset=offset, limit=limit)

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch] | str:
        matches = self._backend.grep_raw(pattern, path=path, glob=glob)
        if isinstance(matches, str):
            return matches
        return [match for match in matches if self._allowed(str(match.get("path", "")))]

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        infos = self._backend.glob_info(pattern, path=path)
        return [info for info in infos if self._allowed(str(info.get("path", "")))]

    def write(self, file_path: str, content: str) -> WriteResult:
        if self._context_pack.access_for_path(file_path) != ContextAccessLevel.FULL:
            return WriteResult(error=self._deny_message(file_path))
        return self._backend.write(file_path, content)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        if self._context_pack.access_for_path(file_path) != ContextAccessLevel.FULL:
            return EditResult(error=self._deny_message(file_path))
        return self._backend.edit(file_path, old_string, new_string, replace_all=replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        blocked = [path for path, _ in files if self._context_pack.access_for_path(path) != ContextAccessLevel.FULL]
        if blocked:
            return [FileUploadResponse(path=path, error=self._deny_message(path)) for path in blocked]
        return self._backend.upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        blocked = [path for path in paths if not self._allowed(path)]
        if blocked:
            return [FileDownloadResponse(path=path, content=None, error=self._deny_message(path)) for path in blocked]
        return self._backend.download_files(paths)


def build_factory_backend(root_dir: Path, context_pack: ContextPack | None = None) -> BackendProtocol:
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
    module_version_dir.mkdir(parents=True, exist_ok=True)
    (module_version_dir / ".sealed").write_text("sealed=true\n", encoding="utf-8")


def mark_module_immutable(module_version_dir: Path) -> None:
    module_version_dir.mkdir(parents=True, exist_ok=True)
    (module_version_dir / ".immutable").write_text("immutable=true\n", encoding="utf-8")
