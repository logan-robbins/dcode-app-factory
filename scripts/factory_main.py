"""Run the skeleton AI software product factory end-to-end.

The script reads raw spec text, runs ProductLoop to build a structured
spec, executes ProjectLoop (which dispatches EngineeringLoop per task),
and prints the registered micro-modules from the in-memory CodeIndex.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from dcode_app_factory import CodeIndex, ProductLoop, ProjectLoop
from dcode_app_factory.settings import RuntimeSettings


DEFAULT_SPEC_TEXT = "Placeholder specification for dcode_app_factory."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dcode_app_factory loops")
    parser.add_argument(
        "--spec-file",
        type=Path,
        default=None,
        help="Optional path to SPEC.md-style input text.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def load_raw_spec(spec_file: Path | None) -> str:
    settings = RuntimeSettings.from_env()
    if spec_file is not None:
        if not spec_file.is_file():
            raise FileNotFoundError(
                f"Requested spec file does not exist: {spec_file}"
            )
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

    product_loop = ProductLoop(raw_spec)
    structured_spec = product_loop.run()

    code_index = CodeIndex()
    project_loop = ProjectLoop(structured_spec, code_index)
    success = project_loop.run()

    print(f"project_success={success}")
    print("registered_modules:")
    for slug, contract in code_index.items():
        print(f"- {slug}: {contract.name} ({contract.module_id})")

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
