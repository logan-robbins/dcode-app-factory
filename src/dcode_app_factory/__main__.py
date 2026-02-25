"""Entry point for `python -m dcode_app_factory` and the `dcode` CLI script."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dcode_app_factory import FactoryOrchestrator
from dcode_app_factory.settings import RuntimeSettings


REQUEST_KIND_CHOICES = ["AUTO", "FULL_APP", "FEATURE", "BUGFIX", "REFACTOR", "TASK"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dcode_app_factory outer graph")
    parser.add_argument("--request-file", type=Path, default=None, help="Optional path to markdown work request input")
    parser.add_argument("--request-text", default=None, help="Inline work request text (mutually exclusive with request file)")
    parser.add_argument(
        "--request-kind",
        type=lambda value: value.upper(),
        default="AUTO",
        choices=REQUEST_KIND_CHOICES,
        help="Request intent: full app build or incremental work item routing",
    )
    parser.add_argument(
        "--target-codebase-root",
        type=Path,
        default=None,
        help="Optional path to target repository root for existing-codebase feature work",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help="Root directory agents will read/write files into (default: cwd)",
    )
    parser.add_argument(
        "--approval-action",
        default="APPROVE",
        choices=["APPROVE", "REJECT", "AMEND"],
        help="Approval gate action used for non-interactive execution",
    )
    parser.add_argument("--approval-feedback", default=None, help="Optional feedback payload for REJECT/AMEND")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def load_raw_request(*, request_file: Path | None, request_text: str | None, repo_root: Path | None = None) -> str:
    if request_text is not None and request_file is not None:
        raise ValueError("request_text cannot be combined with request_file input")

    settings = RuntimeSettings.from_env()
    if request_text is not None:
        trimmed = request_text.strip()
        if not trimmed:
            raise ValueError("request_text must be non-empty")
        return trimmed

    if request_file is not None:
        if not request_file.is_file():
            raise FileNotFoundError(f"Requested input file does not exist: {request_file}")
        return request_file.read_text(encoding="utf-8")

    effective_root = repo_root if repo_root is not None else Path(__file__).resolve().parents[3]
    default_path = settings.default_request_file(effective_root)
    if default_path.is_file():
        return default_path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Default request file does not exist: {default_path}")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Set workspace root before constructing any settings objects.
    if args.workspace_root is not None:
        workspace_root = args.workspace_root.resolve()
        workspace_root.mkdir(parents=True, exist_ok=True)
        factory_parent = Path(__file__).resolve().parents[2]
        if workspace_root == factory_parent or factory_parent in workspace_root.parents:
            logging.warning(
                "workspace_root (%s) resolves inside the factory package directory (%s). "
                "Agents will write files into the factory source tree. "
                "Pass a separate --workspace-root to avoid this.",
                workspace_root,
                factory_parent,
            )
        os.environ["FACTORY_WORKSPACE_ROOT"] = str(workspace_root)

    repo_root = Path(__file__).resolve().parents[3]

    try:
        raw_request = load_raw_request(
            request_file=args.request_file,
            request_text=args.request_text,
            repo_root=repo_root,
        )
        target_codebase_root = args.target_codebase_root
        if target_codebase_root is not None:
            if not target_codebase_root.exists():
                raise FileNotFoundError(f"Target codebase root does not exist: {target_codebase_root}")
            if not target_codebase_root.is_dir():
                raise ValueError(f"Target codebase root is not a directory: {target_codebase_root}")
    except (OSError, ValueError) as exc:
        logging.error("Unable to load request input: %s", exc)
        return 1

    orchestrator = FactoryOrchestrator(
        raw_spec=raw_request,
        request_kind=args.request_kind,
        target_codebase_root=str(target_codebase_root.resolve()) if target_codebase_root is not None else None,
    )
    try:
        result = orchestrator.run(
            approval_action=args.approval_action,
            approval_feedback=args.approval_feedback,
        )
    except Exception as exc:  # noqa: BLE001
        logging.exception("Factory execution failed: %s", exc)
        return 1

    project_success = bool(result.get("project_success", False))
    print(f"project_success={project_success}")

    release_result = result.get("release_result")
    if release_result is not None:
        overall = release_result.get("overall_result", "UNKNOWN")
        print(f"release_result={overall}")
        print("release_details:")
        print(json.dumps(release_result, indent=2, default=str))

    return 0 if project_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
