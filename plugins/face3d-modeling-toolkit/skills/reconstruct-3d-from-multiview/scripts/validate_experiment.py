#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _common import atomic_write_json, load_json, sha256_file, utc_now, valid_sha256

STAGES = ("intake", "fit", "skin", "qa", "package")


def manifest_path(value: Path) -> Path:
    resolved = value.expanduser().resolve()
    return resolved / "experiment.json" if resolved.is_dir() else resolved


def add(violations: list[dict[str, Any]], code: str, **details: Any) -> None:
    violations.append({"code": code, **details})


def check_record(
    violations: list[dict[str, Any]],
    record: dict[str, Any],
    label: str,
) -> None:
    path = Path(str(record.get("path", "")))
    if not path.is_file():
        add(violations, "file-missing", label=label, path=str(path))
        return
    expected = record.get("sha256")
    actual = sha256_file(path)
    if expected != actual:
        add(
            violations,
            "hash-mismatch",
            label=label,
            path=str(path),
            expected=expected,
            actual=actual,
        )


def admission_violations(
    manifest: dict[str, Any], root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    if manifest.get("schemaVersion") != 1:
        add(violations, "schema-version-invalid")
    if manifest.get("route") != "continuous-template-deformation":
        add(violations, "route-invalid", actual=manifest.get("route"))
    quality_record = manifest.get("qualityContract", {})
    profile_record = manifest.get("profile", {})
    quality_path = root / str(quality_record.get("path", ""))
    profile_path = root / str(profile_record.get("path", ""))
    quality: dict[str, Any] = {}
    for label, path, record in (
        ("quality-contract", quality_path, quality_record),
        ("profile", profile_path, profile_record),
    ):
        if not path.is_file():
            add(violations, "file-missing", label=label, path=str(path))
        else:
            actual = sha256_file(path)
            if actual != record.get("sha256"):
                add(
                    violations,
                    "hash-mismatch",
                    label=label,
                    expected=record.get("sha256"),
                    actual=actual,
                )
    if quality_path.is_file():
        quality = load_json(quality_path)

    inputs = manifest.get("inputs", {})
    required = inputs.get("requiredViews", [])
    views = inputs.get("views", {})
    if not isinstance(required, list) or len(required) < 2:
        add(violations, "required-views-invalid")
        required = []
    missing = sorted(set(required) - set(views)) if isinstance(views, dict) else required
    if missing:
        add(violations, "missing-view", roles=missing)
    hashes: dict[str, list[str]] = {}
    minimum_short_side = int(quality.get("input", {}).get("minimumShortSide", 1024))
    if isinstance(views, dict):
        for role, record in views.items():
            check_record(violations, record, f"view:{role}")
            hashes.setdefault(str(record.get("sha256")), []).append(role)
            width = record.get("width")
            height = record.get("height")
            if not isinstance(width, int) or not isinstance(height, int):
                add(violations, "image-dimensions-unavailable", role=role)
            elif min(width, height) < minimum_short_side:
                add(
                    violations,
                    "image-too-small",
                    role=role,
                    shortSide=min(width, height),
                    minimum=minimum_short_side,
                )
    duplicates = [roles for roles in hashes.values() if len(roles) > 1]
    if duplicates:
        add(violations, "duplicate-view-content", groups=duplicates)
    template = manifest.get("template", {})
    if isinstance(template, dict):
        check_record(violations, template, "template")
        license_record = template.get("license")
        if isinstance(license_record, dict):
            check_record(violations, license_record, "template-license")
        else:
            add(violations, "template-license-missing")
    else:
        add(violations, "template-missing")
    config = manifest.get("config")
    if isinstance(config, dict):
        check_record(violations, config, "config")
    return violations, quality


def metric_number(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def final_violations(manifest: dict[str, Any], quality: dict[str, Any]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    stages = manifest.get("stages", {})
    for stage in STAGES:
        if stages.get(stage, {}).get("status") != "pass":
            add(violations, "stage-not-passed", stage=stage)
    if violations:
        return violations

    intake = stages["intake"]["metrics"]
    for key in ("viewConsistencyConfirmed", "masksConfirmed", "sameSubjectConfirmed"):
        if intake.get(key) is not True:
            add(violations, "intake-review-missing", field=key)

    fit = stages["fit"]["metrics"]
    fitted_hash = fit.get("geometryHash")
    uv_hash = fit.get("uvHash")
    if not valid_sha256(fitted_hash):
        add(violations, "fit-geometry-hash-invalid")
    if not valid_sha256(uv_hash):
        add(violations, "fit-uv-hash-invalid")
    fit_contract = quality.get("fit", {})
    required_views = manifest["inputs"]["requiredViews"]
    per_view = fit.get("perView", {})
    for role in required_views:
        metrics = per_view.get(role, {})
        silhouette = metric_number(metrics, "silhouetteIou")
        minimum_iou = float(fit_contract.get("minimumSilhouetteIou", 0.9))
        if silhouette is None or silhouette < minimum_iou:
            add(
                violations,
                "fit-silhouette-failed",
                role=role,
                actual=silhouette,
                minimum=minimum_iou,
            )
        feature_nme = metric_number(metrics, "featureNme")
        if fit_contract.get("featureNmeRequired") is True and feature_nme is None:
            add(violations, "fit-feature-nme-missing", role=role)
        if feature_nme is not None:
            maximum_nme = float(fit_contract.get("maximumFeatureNme", 0.03))
            if feature_nme > maximum_nme:
                add(
                    violations,
                    "fit-feature-nme-failed",
                    role=role,
                    actual=feature_nme,
                    maximum=maximum_nme,
                )
    for key, threshold_key in (
        ("invertedFaces", "maximumInvertedFaces"),
        ("selfIntersectionPairs", "maximumSelfIntersectionPairs"),
    ):
        value = fit.get(key)
        maximum = int(fit_contract.get(threshold_key, 0))
        if not isinstance(value, int) or value > maximum:
            add(violations, "fit-geometry-safety-failed", metric=key, actual=value, maximum=maximum)

    skin = stages["skin"]["metrics"]
    skin_contract = quality.get("skin", {})
    if skin.get("geometryHash") != fitted_hash or skin.get("geometryChanged") is not False:
        add(violations, "skin-geometry-hash-mismatch")
    if skin.get("uvHash") != uv_hash:
        add(violations, "skin-uv-hash-mismatch")
    uv_difference = metric_number(skin, "uvMaximumDifference")
    maximum_uv_difference = float(skin_contract.get("maximumUvDelta", 0.0))
    if uv_difference is None or uv_difference > maximum_uv_difference:
        add(violations, "skin-uv-drift", actual=uv_difference, maximum=maximum_uv_difference)
    for key, contract_key, direction in (
        ("observedFraction", "minimumObservedFraction", "minimum"),
        ("seamDeltaE00Median", "maximumSeamDeltaE00Median", "maximum"),
        ("seamDeltaE00P95", "maximumSeamDeltaE00P95", "maximum"),
    ):
        value = metric_number(skin, key)
        threshold = float(skin_contract[contract_key])
        failed = value is None or (
            value < threshold if direction == "minimum" else value > threshold
        )
        if failed:
            add(violations, "skin-quality-failed", metric=key, actual=value, threshold=threshold)

    qa = stages["qa"]["metrics"]
    if qa.get("geometryHash") != fitted_hash:
        add(violations, "qa-geometry-hash-mismatch")
    topology = qa.get("topology", {})
    topology_contract = quality.get("topology", {})
    connected_components = topology.get("connectedComponents")
    required_components = topology_contract.get("requiredConnectedComponents")
    if not isinstance(connected_components, int) or connected_components != required_components:
        add(
            violations,
            "topology-gate-failed",
            metric="connectedComponents",
            actual=connected_components,
            required=required_components,
        )
    maximum_checks = (
        ("boundaryEdges", "maximumBoundaryEdges"),
        ("nonManifoldEdges", "maximumNonManifoldEdges"),
        ("degenerateFaces", "maximumDegenerateFaces"),
        ("invertedFaces", "maximumInvertedFaces"),
        ("selfIntersectionPairs", "maximumSelfIntersectionPairs"),
    )
    for key, contract_key in maximum_checks:
        actual = topology.get(key)
        maximum = topology_contract.get(contract_key)
        if not isinstance(actual, int) or not isinstance(maximum, int) or actual > maximum:
            add(
                violations,
                "topology-gate-failed",
                metric=key,
                actual=actual,
                maximum=maximum,
            )
    sdf = qa.get("sdf", {})
    sdf_contract = quality.get("sdf", {})
    if sdf.get("role") != sdf_contract.get("requiredRole", "qa-only"):
        add(violations, "sdf-role-invalid", actual=sdf.get("role"))
    if sdf.get("surfaceGenerated") is not False:
        add(violations, "sdf-generated-surface")
    delivery_contract = quality.get("delivery", {})
    review_views = set(qa.get("reviewViews", []))
    missing_review = sorted(set(delivery_contract.get("requiredReviewViews", [])) - review_views)
    if missing_review:
        add(violations, "review-view-missing", roles=missing_review)
    if qa.get("criticalRegionsPassed") is not True:
        add(violations, "critical-region-failed")
    if delivery_contract.get("requireUserSignoff") and qa.get("userSignoff") is not True:
        add(violations, "user-signoff-missing")

    package = stages["package"]["metrics"]
    if package.get("geometryHash") != fitted_hash:
        add(violations, "package-geometry-hash-mismatch")
    if package.get("finalSurfaceSource") != delivery_contract.get("requiredSurfaceSource"):
        add(violations, "package-surface-source-invalid", actual=package.get("finalSurfaceSource"))
    if delivery_contract.get("requireLocalOnly") and package.get("localOnly") is not True:
        add(violations, "package-not-local-only")
    if package.get("externalRequests") != 0:
        add(violations, "external-request-detected", actual=package.get("externalRequests"))

    for stage in STAGES:
        for artifact in stages[stage].get("artifacts", []):
            check_record(violations, artifact, f"artifact:{stage}:{artifact.get('role')}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate admission or final experiment contracts."
    )
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--stage", choices=("admission", "final"), default="admission")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    path = manifest_path(arguments.experiment)
    manifest = load_json(path)
    violations, quality = admission_violations(manifest, path.parent)
    if arguments.stage == "final" and not violations:
        violations.extend(final_violations(manifest, quality))
    result = {
        "schemaVersion": 1,
        "checkedAt": utc_now(),
        "experiment": str(path),
        "stage": arguments.stage,
        "passed": not violations,
        "violations": violations,
    }
    if arguments.output is not None:
        atomic_write_json(arguments.output.expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
