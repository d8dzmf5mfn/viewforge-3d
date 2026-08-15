#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _common import atomic_write_json, file_record, load_json, parse_assignments, utc_now

STAGES = ("intake", "fit", "skin", "qa", "package")
STATUSES = ("pass", "fail", "blocked")


def manifest_path(value: Path) -> Path:
    resolved = value.expanduser().resolve()
    return resolved / "experiment.json" if resolved.is_dir() else resolved


def main() -> int:
    parser = argparse.ArgumentParser(description="Record one native pipeline stage atomically.")
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--status", required=True, choices=STATUSES)
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--artifact", action="append", default=[])
    arguments = parser.parse_args()

    path = manifest_path(arguments.experiment)
    manifest = load_json(path)
    if manifest.get("route") != "continuous-template-deformation":
        parser.error("experiment route is not continuous-template-deformation")
    metrics: dict[str, Any] = {}
    if arguments.metrics_json is not None:
        metrics = load_json(arguments.metrics_json.expanduser().resolve())
    if arguments.status == "pass" and not metrics:
        parser.error("passing a stage requires --metrics-json")
    try:
        artifact_paths = parse_assignments(arguments.artifact, "artifact")
    except ValueError as error:
        parser.error(str(error))
    artifacts = [file_record(Path(raw), role=role) for role, raw in sorted(artifact_paths.items())]

    index = STAGES.index(arguments.stage)
    if arguments.status == "pass" and index > 0:
        previous = STAGES[index - 1]
        if manifest["stages"][previous]["status"] != "pass":
            parser.error(f"cannot pass {arguments.stage} before {previous} passes")
    event = {
        "recordedAt": utc_now(),
        "stage": arguments.stage,
        "status": arguments.status,
        "previousStatus": manifest["stages"][arguments.stage]["status"],
    }
    manifest["events"].append(event)
    manifest["stages"][arguments.stage] = {
        "status": arguments.status,
        "metrics": metrics,
        "artifacts": artifacts,
        "recordedAt": event["recordedAt"],
    }
    manifest["state"] = (
        f"{arguments.stage}-passed" if arguments.status == "pass" else arguments.status
    )
    atomic_write_json(path, manifest)
    print(
        json.dumps(
            manifest["stages"][arguments.stage], ensure_ascii=False, indent=2, sort_keys=True
        )
    )
    return 0 if arguments.status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
