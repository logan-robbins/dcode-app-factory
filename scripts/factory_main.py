"""Run the AI software product factory end-to-end."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dcode_app_factory import FactoryOrchestrator
from dcode_app_factory.settings import RuntimeSettings


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUEST_TEXT = "# Product\n## Factory architecture\n## Project loop\n## Engineering debate\n"
REQUEST_KIND_CHOICES = ["AUTO", "FULL_APP", "FEATURE", "BUGFIX", "REFACTOR", "TASK"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dcode_app_factory outer graph")
    parser.add_argument("--request-file", type=Path, default=None, help="Optional path to markdown work request input")
    parser.add_argument("--request-text", default=None, help="Inline work request text (mutually exclusive with request/spec file)")
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
    parser.add_argument("--spec-file", type=Path, default=None, help=argparse.SUPPRESS)
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


def load_raw_request(*, request_file: Path | None, request_text: str | None, spec_file: Path | None) -> str:
    if request_text is not None and (request_file is not None or spec_file is not None):
        raise ValueError("request_text cannot be combined with request/spec file input")
    if request_file is not None and spec_file is not None:
        raise ValueError("request_file and spec_file are mutually exclusive")

    settings = RuntimeSettings.from_env()
    if request_text is not None:
        trimmed = request_text.strip()
        if not trimmed:
            raise ValueError("request_text must be non-empty")
        return trimmed

    source_path = request_file if request_file is not None else spec_file
    if source_path is not None:
        if not source_path.is_file():
            raise FileNotFoundError(f"Requested input file does not exist: {source_path}")
        return source_path.read_text(encoding="utf-8")

    default_path = settings.default_request_file(REPO_ROOT)
    if default_path.is_file():
        return default_path.read_text(encoding="utf-8")

    return DEFAULT_REQUEST_TEXT


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        raw_request = load_raw_request(
            request_file=args.request_file,
            request_text=args.request_text,
            spec_file=args.spec_file,
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
