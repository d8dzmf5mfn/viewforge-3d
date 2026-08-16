from __future__ import annotations

import argparse
from pathlib import Path

from .jobs import run_job


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one immutable ViewForge local Blender job.")
    parser.add_argument("--request", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    return run_job(arguments.request)


if __name__ == "__main__":
    raise SystemExit(main())
