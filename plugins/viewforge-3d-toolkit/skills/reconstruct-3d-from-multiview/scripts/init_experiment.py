#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from _common import (
    atomic_write_json,
    file_record,
    load_json,
    parse_assignments,
    sha256_file,
    utc_now,
)

SKILL_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE = SKILL_ROOT / "assets/subject-profile.template.json"
DEFAULT_QUALITY = SKILL_ROOT / "assets/quality-contract.template.json"
EXPERIMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def image_record(role: str, path: Path) -> dict[str, Any]:
    record = file_record(path, role=role)
    record.update({"width": None, "height": None, "format": path.suffix.lower().lstrip(".")})
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as source:
            normalized = ImageOps.exif_transpose(source)
            record["width"], record["height"] = normalized.size
            record["format"] = (source.format or record["format"]).lower()
    except (ImportError, OSError):
        record["dimensionProbeFailed"] = True
    return record


def copy_profile_template(
    source: Path, profile_id: str, target_class: str, views: list[str]
) -> dict[str, Any]:
    profile = load_json(source)
    profile["profileId"] = profile_id
    profile["targetClass"] = target_class
    profile.setdefault("views", {})["required"] = views
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an immutable-input multiview reconstruction experiment ledger."
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--target-class", required=True)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--license-record", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--profile-contract", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--quality-contract", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--required-views", default="front,left45,right45")
    parser.add_argument("--view", action="append", default=[])
    arguments = parser.parse_args()

    if EXPERIMENT_NAME.fullmatch(arguments.name) is None:
        parser.error("--name must contain only letters, digits, dot, underscore, or hyphen")
    output = arguments.output.expanduser().resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    required_views = [item.strip() for item in arguments.required_views.split(",") if item.strip()]
    if len(required_views) < 2 or len(set(required_views)) != len(required_views):
        parser.error("--required-views must contain at least two unique roles")
    try:
        view_paths = parse_assignments(arguments.view, "view")
    except ValueError as error:
        parser.error(str(error))
    missing_roles = sorted(set(required_views) - set(view_paths))
    extra_roles = sorted(set(view_paths) - set(required_views))
    if missing_roles:
        parser.error(f"missing required view roles: {missing_roles}")
    missing_view_files = [
        str(Path(raw).expanduser().resolve())
        for raw in view_paths.values()
        if not Path(raw).expanduser().resolve().is_file()
    ]
    if missing_view_files:
        parser.error(f"missing view files: {missing_view_files}")

    project = arguments.project.expanduser().resolve()
    template = arguments.template.expanduser().resolve()
    license_record = arguments.license_record.expanduser().resolve()
    profile_source = arguments.profile_contract.expanduser().resolve()
    quality_source = arguments.quality_contract.expanduser().resolve()
    config = arguments.config.expanduser().resolve() if arguments.config else None
    required_files = [template, license_record, profile_source, quality_source]
    if config is not None:
        required_files.append(config)
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        parser.error(f"missing files: {missing_files}")

    view_records = {
        role: image_record(role, Path(raw).expanduser().resolve())
        for role, raw in sorted(view_paths.items())
    }
    quality = load_json(quality_source)
    minimum_short_side = int(quality["input"]["minimumShortSide"])
    issues: list[dict[str, Any]] = []
    for role, record in view_records.items():
        width = record.get("width")
        height = record.get("height")
        if not isinstance(width, int) or not isinstance(height, int):
            issues.append({"code": "image-dimensions-unavailable", "role": role})
        elif min(width, height) < minimum_short_side:
            issues.append(
                {
                    "code": "image-too-small",
                    "role": role,
                    "shortSide": min(width, height),
                    "minimum": minimum_short_side,
                }
            )
    by_hash: dict[str, list[str]] = {}
    for role, record in view_records.items():
        by_hash.setdefault(str(record["sha256"]), []).append(role)
    duplicate_groups = [roles for roles in by_hash.values() if len(roles) > 1]
    if duplicate_groups:
        issues.append({"code": "duplicate-view-content", "groups": duplicate_groups})

    output.mkdir(parents=True)
    for relative in ("work", "outputs", "qa", "logs"):
        (output / relative).mkdir()
    profile = copy_profile_template(
        profile_source,
        arguments.profile_id,
        arguments.target_class,
        required_views,
    )
    atomic_write_json(output / "profile.json", profile)
    atomic_write_json(output / "quality-contract.json", quality)

    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "experimentId": arguments.name,
        "createdAt": utc_now(),
        "state": "admission-ready" if not issues else "blocked",
        "route": "continuous-template-deformation",
        "projectRoot": str(project),
        "profile": {
            "id": arguments.profile_id,
            "targetClass": arguments.target_class,
            "path": "profile.json",
            "sha256": sha256_file(output / "profile.json"),
        },
        "qualityContract": {
            "path": "quality-contract.json",
            "sha256": sha256_file(output / "quality-contract.json"),
        },
        "inputs": {
            "requiredViews": required_views,
            "extraViews": extra_roles,
            "views": view_records,
        },
        "template": {
            **file_record(template),
            "license": file_record(license_record),
        },
        "config": file_record(config) if config is not None else None,
        "technicalAdmission": {
            "status": "pass" if not issues else "fail",
            "issues": issues,
        },
        "stages": {
            stage: {"status": "pending", "metrics": {}, "artifacts": []}
            for stage in ("intake", "fit", "skin", "qa", "package")
        },
        "events": [],
        "resourcePolicy": {
            "localOnly": True,
            "downloadApprovalBytes": int(quality["resources"]["downloadApprovalBytes"]),
            "downloads": [],
        },
    }
    atomic_write_json(output / "experiment.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
