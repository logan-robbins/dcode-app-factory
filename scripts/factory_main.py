"""Run the AI software product factory end-to-end."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dcode_app_factory import FactoryOrchestrator
from dcode_app_factory.settings import RuntimeSettings


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC_TEXT = "# Product\n## Factory architecture\n## Project loop\n## Engineering debate\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dcode_app_factory outer graph")
    parser.add_argument("--spec-file", type=Path, default=None, help="Optional path to markdown spec input")
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


def load_raw_spec(spec_file: Path | None) -> str:
    settings = RuntimeSettings.from_env()
    if spec_file is not None:
        if not spec_file.is_file():
            raise FileNotFoundError(f"Requested spec file does not exist: {spec_file}")
        return spec_file.read_text(encoding="utf-8")

    default_path = settings.default_spec_file(REPO_ROOT)
    if default_path.is_file():
        return default_path.read_text(encoding="utf-8")

    return DEFAULT_SPEC_TEXT


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        raw_spec = load_raw_spec(args.spec_file)
    except OSError as exc:
        logging.error("Unable to load spec text: %s", exc)
        return 1

    orchestrator = FactoryOrchestrator(raw_spec=raw_spec)
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
