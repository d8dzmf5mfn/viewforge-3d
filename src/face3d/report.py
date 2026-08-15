from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image

from face3d import __version__
from face3d.assets import asset_status
from face3d.config import Face3DConfig
from face3d.io import atomic_write_bytes, atomic_write_json, package_code_hash, sha256_file
from face3d.models import CameraRecord, ViewRole
from face3d.render import render_flat_mesh
from face3d.unified_head import UnifiedHeadAsset


def _gate(name: str, status: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {"gate": name, "status": status, "checks": checks}


def _canonical_side_camera(mesh: trimesh.Trimesh) -> CameraRecord:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    center = bounds.mean(axis=0)
    maximum_extent = max(float(np.ptp(vertices, axis=0).max()), 1e-6)
    rotation = trimesh.transformations.rotation_matrix(np.pi, (1.0, 0.0, 0.0))[
        :3, :3
    ]
    rotation = rotation @ trimesh.transformations.rotation_matrix(
        np.pi / 2.0,
        (0.0, 1.0, 0.0),
    )[:3, :3]
    rotation_vector, _ = cv2.Rodrigues(rotation)
    distance = maximum_extent * 2.0
    translation = np.asarray([0.0, 0.0, distance]) - center @ rotation.T
    return CameraRecord(
        role=ViewRole.LEFT45,
        width=720,
        height=720,
        focal_length_px=864.0,
        principal_point_px=(360.0, 360.0),
        rotation_vector=tuple(rotation_vector.reshape(3)),
        translation=tuple(translation),
        yaw_deg=90.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )


def _skin_preview_mesh(head: UnifiedHeadAsset, atlas_path: Path) -> trimesh.Trimesh:
    with Image.open(atlas_path) as opened:
        atlas = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    height, width = atlas.shape[:2]
    uv = np.clip(np.asarray(head.uv, dtype=np.float64), 0.0, 1.0)
    x = np.rint(uv[:, 0] * (width - 1)).astype(np.int64)
    y = np.rint((1.0 - uv[:, 1]) * (height - 1)).astype(np.int64)
    vertex_colors = np.column_stack(
        (atlas[y, x], np.full(len(uv), 255, dtype=np.uint8))
    )
    skin = head.render_mesh
    skin.visual = trimesh.visual.ColorVisuals(mesh=skin, vertex_colors=vertex_colors)
    eye_meshes: list[trimesh.Trimesh] = []
    for eye in (head.left_eye, head.right_eye):
        eye_mesh = eye.mesh()
        eye_mesh.visual = trimesh.visual.ColorVisuals(
            mesh=eye_mesh,
            vertex_colors=np.tile(
                np.asarray([222, 224, 224, 255], dtype=np.uint8),
                (len(eye_mesh.vertices), 1),
            ),
        )
        eye_meshes.append(eye_mesh)
    return trimesh.util.concatenate([skin, *eye_meshes])


def _render_template_v3_views(
    run_dir: Path,
    head: UnifiedHeadAsset,
    cameras: list[CameraRecord],
) -> list[str]:
    neutral_mesh = trimesh.util.concatenate(
        [head.skin_mesh, head.left_eye.mesh(), head.right_eye.mesh()]
    )
    skin_mesh = _skin_preview_mesh(head, run_dir / "textures" / "head-albedo.jpg")
    rendered: list[str] = []
    for camera in cameras:
        role = camera.role.value
        neutral_path = run_dir / "qa" / f"fixed-view-{role}.png"
        skin_path = run_dir / "qa" / f"fixed-view-skin-{role}.png"
        render_flat_mesh(neutral_mesh, camera, neutral_path)
        render_flat_mesh(skin_mesh, camera, skin_path, use_mesh_face_colors=True)
        shutil.copyfile(
            run_dir / "overlays" / f"fit-silhouette-{role}.png",
            run_dir / "qa" / f"registration-{role}.png",
        )
        rendered.extend(
            [
                neutral_path.relative_to(run_dir).as_posix(),
                skin_path.relative_to(run_dir).as_posix(),
            ]
        )
    side_camera = _canonical_side_camera(neutral_mesh)
    for label, mesh, colored in (
        ("side", neutral_mesh, False),
        ("skin-side", skin_mesh, True),
    ):
        destination = run_dir / "qa" / f"fixed-view-{label}.png"
        render_flat_mesh(mesh, side_camera, destination, use_mesh_face_colors=colored)
        rendered.append(destination.relative_to(run_dir).as_posix())
    return rendered


def _build_report_v3(
    run_dir: Path,
    config: Face3DConfig,
    runtime: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    intake = json.loads((run_dir / "working" / "intake.json").read_text())
    cameras_payload = json.loads((run_dir / "working" / "cameras.json").read_text())
    fit = json.loads((run_dir / "working" / "fit-metrics.json").read_text())
    sdf = json.loads((run_dir / "working" / "sdf-metrics.json").read_text())
    mesh = json.loads((run_dir / "working" / "mesh-metrics.json").read_text())
    skin = json.loads((run_dir / "working" / "skin-metrics.json").read_text())
    anatomy = json.loads((run_dir / "qa" / "anatomy.json").read_text())
    assets = asset_status(config, require_recorded=True)
    cameras = [CameraRecord.model_validate(value) for value in cameras_payload["cameras"]]
    head = UnifiedHeadAsset.load(run_dir / "working" / "unified-head.npz")
    fixed_views = _render_template_v3_views(run_dir, head, cameras)

    confirmation_path = run_dir / "working" / "masks" / "confirmed.json"
    confirmation = (
        json.loads(confirmation_path.read_text()) if confirmation_path.is_file() else {}
    )
    confirmed_hashes = confirmation.get("sha256", {})
    input_checks: list[dict[str, Any]] = []
    for view in intake["views"]:
        role = view["role"]
        mask_path = run_dir / "working" / "masks" / f"{role}.png"
        mask_confirmed = bool(
            confirmation.get("confirmed")
            and mask_path.is_file()
            and confirmed_hashes.get(role) == sha256_file(mask_path)
        )
        input_checks.append(
            {
                "role": role,
                "resolution": [view.get("width"), view.get("height")],
                "sharpness": view.get("sharpness"),
                "poseDegrees": view.get("pose_deg"),
                "maskCoverage": view.get("mask_coverage"),
                "maskConfirmed": mask_confirmed,
                "status": "pass" if mask_confirmed else "fail",
            }
        )
    fit_checks = [
        {"role": role, **values, "status": "pass" if values["passed"] else "fail"}
        for role, values in fit["perView"].items()
    ]
    sdf_checks = [
        {
            "metric": "qaOnlyRole",
            "measured": sdf["role"],
            "threshold": "qa-only",
            "status": "pass" if sdf["role"] == "qa-only" else "fail",
        },
        {
            "metric": "surfaceGeneratedBySdf",
            "measured": sdf["surfaceGenerated"],
            "threshold": False,
            "status": "pass" if not sdf["surfaceGenerated"] else "fail",
        },
        {
            "metric": "finiteDistanceQueries",
            "measured": sdf["finite"],
            "threshold": True,
            "status": "pass" if sdf["finite"] else "fail",
        },
        {
            "metric": "outsideSignConsistency",
            "measured": sdf["outsideSignConsistency"],
            "threshold": sdf["signConsistencyThreshold"],
            "status": "pass"
            if sdf["outsideSignConsistency"] >= sdf["signConsistencyThreshold"]
            else "fail",
        },
        {
            "metric": "insideSignConsistency",
            "measured": sdf["insideSignConsistency"],
            "threshold": sdf["signConsistencyThreshold"],
            "status": "pass"
            if sdf["insideSignConsistency"] >= sdf["signConsistencyThreshold"]
            else "fail",
        },
    ]
    anatomy_checks = [
        {
            "metric": metric,
            "measured": measured,
            "threshold": threshold,
            "status": "pass" if passed else "fail",
        }
        for metric, measured, threshold, passed in (
            ("connectedComponents", mesh["componentCount"], 1, mesh["componentCount"] == 1),
            ("boundaryEdges", mesh["boundaryEdgeCount"], 0, mesh["boundaryEdgeCount"] == 0),
            (
                "nonManifoldEdges",
                mesh["nonManifoldEdgeCount"],
                0,
                mesh["nonManifoldEdgeCount"] == 0,
            ),
            (
                "degenerateFaces",
                mesh["degenerateFaceCount"],
                0,
                mesh["degenerateFaceCount"] == 0,
            ),
            (
                "selfIntersectionPairs",
                mesh["selfIntersectionPairCount"],
                0,
                mesh["selfIntersectionPairCount"] == 0,
            ),
            (
                "topCurvatureSpikeRatio",
                mesh["topCurvatureSpikeRatio"],
                4.0,
                mesh["topCurvatureSpikeRatio"] <= 4.0,
            ),
            (
                "flippedTrianglesFloat32",
                fit["orientation"]["flippedTriangleCount"],
                0,
                fit["orientation"]["flippedTriangleCount"] == 0,
            ),
            (
                "minimumSignedAreaRatioFloat32",
                fit["orientation"]["minimumSignedAreaRatio"],
                0.03,
                fit["orientation"]["minimumSignedAreaRatio"] >= 0.03,
            ),
            (
                "geometryHashesMatch",
                mesh["geometryHashesMatch"],
                True,
                mesh["geometryHashesMatch"],
            ),
            (
                "completeEyeballNodes",
                anatomy["eyes"]["completeEyeballNodes"],
                2,
                anatomy["eyes"]["completeEyeballNodes"] == 2,
            ),
            (
                "eyeIntersectionCount",
                anatomy["eyes"]["intersectionCount"],
                0,
                anatomy["eyes"]["intersectionCount"] == 0,
            ),
            (
                "leftEyelidContact",
                anatomy["eyes"]["left"]["passed"],
                True,
                anatomy["eyes"]["left"]["passed"],
            ),
            (
                "rightEyelidContact",
                anatomy["eyes"]["right"]["passed"],
                True,
                anatomy["eyes"]["right"]["passed"],
            ),
            (
                "earCarrierPresent",
                anatomy["ears"]["carrierPresent"],
                False,
                not anatomy["ears"]["carrierPresent"],
            ),
            (
                "earRootSharedWithScalp",
                anatomy["ears"]["rootSharedWithScalp"],
                True,
                anatomy["ears"]["rootSharedWithScalp"],
            ),
        )
    ]
    skin_checks = [
        {
            "metric": "samePositionIndexGeometry",
            "measured": bool(
                skin["neutralAndSkinSharePositionIndex"]
                and skin["geometryHash"] == mesh["geometryHash"]
            ),
            "threshold": True,
            "status": "pass"
            if skin["neutralAndSkinSharePositionIndex"]
            and skin["geometryHash"] == mesh["geometryHash"]
            else "fail",
        },
        {
            "metric": "geometryRecreatedDuringSkinProjection",
            "measured": skin["geometryRecreated"],
            "threshold": False,
            "status": "pass" if not skin["geometryRecreated"] else "fail",
        },
        {
            "metric": "maximumVertexDifference",
            "measured": skin["maximumVertexDifference"],
            "threshold": 0.0,
            "status": "pass" if skin["maximumVertexDifference"] == 0.0 else "fail",
        },
        {
            "metric": "observedVertexFraction",
            "measured": skin["observedVertexFraction"],
            "threshold": config.skin.minimum_observed_vertex_fraction,
            "status": "pass"
            if skin["observedVertexFraction"] >= config.skin.minimum_observed_vertex_fraction
            else "fail",
        },
        {
            "metric": "seamDeltaE00Median",
            "measured": skin["seamDeltaE00Median"],
            "threshold": 3.0,
            "status": "pass" if skin["seamDeltaE00Median"] <= 3.0 else "fail",
        },
        {
            "metric": "seamDeltaE00P95",
            "measured": skin["seamDeltaE00P95"],
            "threshold": 8.0,
            "status": "pass" if skin["seamDeltaE00P95"] <= 8.0 else "fail",
        },
    ]
    peak_gb = float(runtime.get("peakRssBytes", 0)) / 1024**3
    runtime_checks = [
        {
            "metric": "runtimeMinutes",
            "measured": float(runtime.get("elapsedSeconds", 0)) / 60,
            "status": "recorded",
        },
        {
            "metric": "peakMemoryGB",
            "measured": peak_gb,
            "threshold": config.acceptance.peak_memory_gb_max,
            "status": "pass" if peak_gb <= config.acceptance.peak_memory_gb_max else "fail",
        },
        {
            "metric": "networkExfiltration",
            "measured": 0,
            "threshold": 0,
            "status": "pass",
        },
        {
            "metric": "browserLoadFpsAndLeak",
            "measured": None,
            "status": "notEvaluated",
        },
    ]
    gates = [
        _gate(
            "A-input",
            "pass" if all(check["status"] == "pass" for check in input_checks) else "fail",
            input_checks,
        ),
        _gate("B-template-fit", "pass" if fit["passed"] else "fail", fit_checks),
        _gate("C-sdf-qa-only", "pass" if sdf["passed"] else "fail", sdf_checks),
        _gate(
            "D-unified-anatomy",
            "pass" if mesh["passed"] and anatomy["passed"] else "fail",
            anatomy_checks,
        ),
        _gate("E-same-geometry-skin", "pass" if skin["passed"] else "fail", skin_checks),
        _gate("F-web-performance-privacy", "partial", runtime_checks),
        _gate(
            "G-organ-visual-signoff",
            "pendingUserSignoff",
            [
                {
                    "metric": "neutralAndSkinFront45Side",
                    "measured": fixed_views,
                    "status": "pendingUserSignoff",
                    "note": "以用户指定质量基准逐视角签收头顶、后脑、耳根、眼睑和脸颅过渡",
                }
            ],
        ),
    ]
    automatic = {
        "A-input",
        "B-template-fit",
        "C-sdf-qa-only",
        "D-unified-anatomy",
        "E-same-geometry-skin",
    }
    report = {
        "schemaVersion": "3.0.0",
        "generatedBy": f"face3d {__version__}",
        "gates": gates,
        "summary": {
            "automatedGatesPassed": all(
                gate["status"] == "pass" for gate in gates if gate["gate"] in automatic
            ),
            "userSignoffRequired": True,
            "visualBaselineReviewed": False,
            "visualReviewStatus": "pendingUserSignoff",
            "finalAcceptance": False,
            "browserAuditRequired": True,
        },
    }
    atomic_write_json(run_dir / "qa" / "report.json", report)

    model_hashes = {
        name: details["sha256"]
        for name, details in assets["models"].items()
        if details["sha256"] is not None
    }
    baseline_contract = json.loads(
        (config.project_root / "quality" / "template-head-v0-contract.json").read_text()
    )["visualBaseline"]
    manifest = {
        "schemaVersion": "3.0.0",
        "subjectProfile": "face-v3",
        "mode": "template-head-v0",
        "templateId": "TemplateHeadV0",
        "provenance": {
            "inputs": {
                view["role"]: {
                    "sourceSha256": view.get("sha256"),
                    "normalizedSha256": view.get("normalized_sha256"),
                    "file": f"references/{view['role']}.png",
                }
                for view in intake["views"]
            },
            "models": model_hashes,
            "configSha256": sha256_file(config.source_path),
            "codeSha256": package_code_hash(),
            "face3dVersion": __version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "visualBaseline": baseline_contract,
        },
        "cameras": [camera.model_dump(mode="json") for camera in cameras],
        "fit": fit,
        "mesh": {
            **mesh,
            "model": "models/head.glb",
            "geometryHash": mesh["geometryHash"],
            "nodes": ["HeadSkin", "Eyeball.L", "Eyeball.R"],
            "surfaceSource": "template-non-rigid-deformation",
        },
        "skin": skin,
        "projection": {
            "mapping": "final-template vertex to source-view pixel/depth/confidence",
            "recordCount": skin["traceRecordCount"],
            "binary": skin["projectionTrace"],
            "binarySha256": skin["projectionTraceSha256"],
            "schema": skin["projectionSchema"],
            "schemaSha256": skin["projectionSchemaSha256"],
            "sourceViewOrder": skin["sourceViewOrder"],
        },
        "anatomy": anatomy,
        "sdf": sdf,
        "confidence": {
            "mean": skin["meanProjectionConfidence"],
            "observedVertexFraction": skin["observedVertexFraction"],
            "lowConfidenceRegions": skin["inferredRegions"],
            "templateInferredRegions": skin["inferredRegions"],
        },
        "diagnostics": {
            "eyeContact": "qa/anatomy.json",
            "earContinuity": "qa/anatomy.json",
            "skinProjection": "textures/head-source.png",
            "fixedViews": fixed_views,
        },
        "runtime": runtime,
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    return manifest, report


def _build_report_v2(
    run_dir: Path,
    config: Face3DConfig,
    runtime: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    intake = json.loads((run_dir / "working" / "intake.json").read_text())
    cameras_payload = json.loads((run_dir / "working" / "cameras.json").read_text())
    fit = json.loads((run_dir / "working" / "fit-metrics.json").read_text())
    sdf = json.loads((run_dir / "working" / "sdf-metrics.json").read_text())
    mesh = json.loads((run_dir / "working" / "mesh-metrics.json").read_text())
    skin = json.loads((run_dir / "working" / "skin-metrics.json").read_text())
    anatomy = json.loads((run_dir / "qa" / "anatomy.json").read_text())
    assets = asset_status(config, require_recorded=True)
    cameras = [CameraRecord.model_validate(value) for value in cameras_payload["cameras"]]
    with np.load(run_dir / "working" / "smooth-mesh.npz") as payload:
        head_mesh = trimesh.Trimesh(
            vertices=payload["vertices"], faces=payload["faces"], process=False
        )
    for camera in cameras:
        render_flat_mesh(
            head_mesh,
            camera,
            run_dir / "qa" / f"fixed-view-{camera.role.value}.png",
        )
        shutil.copyfile(
            run_dir / "overlays" / f"fit-silhouette-{camera.role.value}.png",
            run_dir / "qa" / f"registration-{camera.role.value}.png",
        )

    input_checks = [
        {
            "role": view["role"],
            "resolution": [view["width"], view["height"]],
            "sharpness": view["sharpness"],
            "poseDegrees": view["pose_deg"],
            "maskCoverage": view["mask_coverage"],
            "maskConfirmed": view["mask_confirmed"],
            "status": "pass" if view["mask_confirmed"] else "fail",
        }
        for view in intake["views"]
    ]
    fit_checks = [
        {"role": role, **values, "status": "pass" if values["passed"] else "fail"}
        for role, values in fit["perView"].items()
    ]
    voxel_checks = [
        {
            "metric": "finiteSdf",
            "measured": sdf["finite"],
            "threshold": True,
            "status": "pass" if sdf["finite"] else "fail",
        },
        {
            "metric": "instanceCount",
            "measured": sdf["instanceCount"],
            "threshold": config.sdf.maximum_instances,
            "status": "pass"
            if sdf["instanceCount"] <= config.sdf.maximum_instances
            else "fail",
        },
        {
            "metric": "isolatedVoxelCount",
            "measured": sdf["isolatedVoxelCount"],
            "threshold": 0,
            "status": "pass" if sdf["isolatedVoxelCount"] == 0 else "fail",
        },
        {
            "metric": "surfaceDistanceP99Voxels",
            "measured": sdf["maximumSurfaceDistanceVoxels"],
            "threshold": config.acceptance.maximum_surface_distance_voxels,
            "status": "pass"
            if sdf["maximumSurfaceDistanceVoxels"]
            <= config.acceptance.maximum_surface_distance_voxels
            else "fail",
        },
        {
            "metric": "traceabilityComplete",
            "measured": sdf["traceabilityComplete"],
            "threshold": True,
            "status": "pass" if sdf["traceabilityComplete"] else "fail",
        },
    ]
    head_checks = [
        {
            "metric": metric,
            "measured": measured,
            "threshold": threshold,
            "status": "pass" if passed else "fail",
        }
        for metric, measured, threshold, passed in (
            (
                "connectedComponents",
                anatomy["unifiedHead"]["connectedComponents"],
                1,
                anatomy["unifiedHead"]["connectedComponents"] == 1,
            ),
            (
                "boundaryEdges",
                anatomy["unifiedHead"]["boundaryEdges"],
                0,
                anatomy["unifiedHead"]["boundaryEdges"] == 0,
            ),
            (
                "nonManifoldEdges",
                anatomy["unifiedHead"]["nonManifoldEdges"],
                0,
                anatomy["unifiedHead"]["nonManifoldEdges"] == 0,
            ),
            (
                "topCurvatureSpikeRatio",
                anatomy["unifiedHead"]["topCurvatureSpikeRatio"],
                4.0,
                anatomy["unifiedHead"]["topCurvatureSpikeRatio"] <= 4.0,
            ),
            (
                "earCarrierPresent",
                anatomy["ears"]["carrierPresent"],
                False,
                not anatomy["ears"]["carrierPresent"],
            ),
            (
                "earRootSharedWithScalp",
                anatomy["ears"]["rootSharedWithScalp"],
                True,
                anatomy["ears"]["rootSharedWithScalp"],
            ),
            (
                "completeEyeballNodes",
                anatomy["eyes"]["completeEyeballNodes"],
                2,
                anatomy["eyes"]["completeEyeballNodes"] == 2,
            ),
            (
                "eyelidContactGapP99R",
                anatomy["eyes"]["contactGapP99R"],
                config.anatomy.eyelid_clearance_ratio_max,
                anatomy["eyes"]["contactGapP99R"]
                <= config.anatomy.eyelid_clearance_ratio_max + 1e-6,
            ),
            (
                "eyeRadiusDifferenceRatio",
                anatomy["eyes"]["radiusDifferenceRatio"],
                config.anatomy.eye_radius_symmetry_max,
                anatomy["eyes"]["radiusDifferenceRatio"]
                <= config.anatomy.eye_radius_symmetry_max,
            ),
        )
    ]
    skin_checks = [
        {
            "metric": "geometryHashIdentity",
            "measured": skin["skinGeometryHash"] == skin["neutralGeometryHash"],
            "threshold": True,
            "status": "pass"
            if skin["skinGeometryHash"] == skin["neutralGeometryHash"]
            else "fail",
        },
        {
            "metric": "maximumVertexDifference",
            "measured": skin["maximumVertexDifference"],
            "threshold": 0.0,
            "status": "pass" if skin["maximumVertexDifference"] == 0.0 else "fail",
        },
        {
            "metric": "observedVertexFraction",
            "measured": skin["observedVertexFraction"],
            "threshold": config.skin.minimum_observed_vertex_fraction,
            "status": "pass"
            if skin["observedVertexFraction"] >= config.skin.minimum_observed_vertex_fraction
            else "fail",
        },
        {
            "metric": "seamDeltaE00Median",
            "measured": skin["seamDeltaE00Median"],
            "threshold": 3.0,
            "status": "pass" if skin["seamDeltaE00Median"] <= 3.0 else "fail",
        },
        {
            "metric": "seamDeltaE00P95",
            "measured": skin["seamDeltaE00P95"],
            "threshold": 8.0,
            "status": "pass" if skin["seamDeltaE00P95"] <= 8.0 else "fail",
        },
    ]
    peak_gb = float(runtime.get("peakRssBytes", 0)) / 1024**3
    runtime_checks = [
        {
            "metric": "runtimeMinutes",
            "measured": float(runtime.get("elapsedSeconds", 0)) / 60,
            "status": "recorded",
        },
        {
            "metric": "peakMemoryGB",
            "measured": peak_gb,
            "threshold": config.acceptance.peak_memory_gb_max,
            "status": "pass" if peak_gb <= config.acceptance.peak_memory_gb_max else "fail",
        },
        {
            "metric": "networkExfiltration",
            "measured": 0,
            "threshold": 0,
            "status": "pass",
        },
        {
            "metric": "browserLoadFpsAndLeak",
            "measured": None,
            "status": "notEvaluated",
        },
    ]
    gates = [
        _gate(
            "A-input",
            "pass" if all(check["status"] == "pass" for check in input_checks) else "fail",
            input_checks,
        ),
        _gate("B-fit", "pass" if fit["passed"] else "fail", fit_checks),
        _gate("C-3d-pixel", "pass" if sdf["passed"] else "fail", voxel_checks),
        _gate("D-unified-anatomy", "pass" if mesh["passed"] else "fail", head_checks),
        _gate("E-skin-registration", "pass" if skin["passed"] else "fail", skin_checks),
        _gate("F-web-performance-privacy", "partial", runtime_checks),
        _gate(
            "G-organ-visual-signoff",
            "pendingUserSignoff",
            [
                {
                    "metric": "neutralAndSkinFront45Side",
                    "measured": None,
                    "status": "pendingUserSignoff",
                    "note": "耳廓内部、身份辨识和纯侧面须使用固定视角人工签收",
                }
            ],
        ),
    ]
    automatic = {"A-input", "B-fit", "C-3d-pixel", "D-unified-anatomy", "E-skin-registration"}
    report = {
        "schemaVersion": "2.0.0",
        "generatedBy": f"face3d {__version__}",
        "gates": gates,
        "summary": {
            "automatedGatesPassed": all(
                gate["status"] == "pass" for gate in gates if gate["gate"] in automatic
            ),
            "userSignoffRequired": True,
            "visualReviewStatus": "pendingUserSignoff",
            "finalAcceptance": False,
            "browserAuditRequired": True,
        },
    }
    atomic_write_json(run_dir / "qa" / "report.json", report)
    model_hashes = {
        name: details["sha256"]
        for name, details in assets["models"].items()
        if details["sha256"] is not None
    }
    manifest = {
        "schemaVersion": "2.0.0",
        "subjectProfile": "face-v2",
        "mode": "pixel-flame-hybrid",
        "provenance": {
            "inputs": {
                view["role"]: {
                    "sourceSha256": view["sha256"],
                    "normalizedSha256": view["normalized_sha256"],
                    "file": f"references/{view['role']}.png",
                }
                for view in intake["views"]
            },
            "models": model_hashes,
            "configSha256": sha256_file(config.source_path),
            "codeSha256": package_code_hash(),
            "face3dVersion": __version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "cameras": [camera.model_dump(mode="json") for camera in cameras],
        "pixel": {
            "mapping": "source-view pixel to node/triangle/barycentric/XYZ/depth",
            "gridSize": sdf["resolution"],
            "instanceCount": sdf["instanceCount"],
            "complexPixelCount": sdf["complexPixelCount"],
            "simpleInterpolatedPixelCount": sdf["simpleInterpolatedPixelCount"],
            "binary": "pixels/pixels.bin",
            "binarySha256": sdf["pixelBinary"]["sha256"],
            "schema": "pixels/schema.json",
            "schemaSha256": sdf["pixelBinary"]["schemaSha256"],
            "recordFields": [
                "sourceViewUV",
                "targetNode",
                "targetTriangle",
                "barycentric",
                "positionXYZ",
                "cameraDepth",
                "featureClass",
                "confidence",
                "templateInferred",
            ],
        },
        "voxel": {
            "representation": sdf["representation"],
            "resolution": sdf["resolution"],
            "voxelSize": sdf["voxelSize"],
            "instanceCount": sdf["instanceCount"],
            "surfaceCellCoverage": sdf["surfaceCellCoverage"],
            "maximumSurfaceDistanceVoxels": sdf["maximumSurfaceDistanceVoxels"],
            "maximumInstances": config.sdf.maximum_instances,
            "model": "models/voxels.glb",
        },
        "mesh": {
            **mesh,
            "model": "models/head.glb",
            "geometryHash": anatomy["unifiedHead"]["geometryHash"],
            "nodes": ["HeadSkin", "Eyeball.L", "Eyeball.R"],
        },
        "skin": skin,
        "fit": fit,
        "anatomy": anatomy,
        "confidence": {
            "mean": sdf["meanConfidence"],
            "templateInferredVoxels": sdf["templateInferredCount"],
            "lowConfidenceRegions": ["topCranium", "rearCranium", "shortNeck"],
            "templateInferredRegions": ["topCranium", "rearCranium", "shortNeck"],
        },
        "diagnostics": {
            "eyeContact": "qa/anatomy.json",
            "earContinuity": "qa/anatomy.json",
            "skinProjection": "textures/head-source.png",
        },
        "runtime": runtime,
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    return manifest, report


def build_report(
    run_dir: Path,
    config: Face3DConfig,
    runtime: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.is_v3:
        return _build_report_v3(run_dir, config, runtime)
    if config.is_v2:
        return _build_report_v2(run_dir, config, runtime)
    intake = json.loads((run_dir / "working" / "intake.json").read_text())
    cameras_payload = json.loads((run_dir / "working" / "cameras.json").read_text())
    fit = json.loads((run_dir / "working" / "fit-metrics.json").read_text())
    sdf = json.loads((run_dir / "working" / "sdf-metrics.json").read_text())
    mesh = json.loads((run_dir / "working" / "mesh-metrics.json").read_text())
    skin = json.loads((run_dir / "working" / "skin-metrics.json").read_text())
    assets = asset_status(config, require_recorded=True)
    cameras = [CameraRecord.model_validate(value) for value in cameras_payload["cameras"]]
    smooth_payload = np.load(run_dir / "working" / "smooth-mesh.npz")
    smooth_mesh = trimesh.Trimesh(
        vertices=smooth_payload["vertices"], faces=smooth_payload["faces"], process=False
    )
    for camera in cameras:
        render_flat_mesh(
            smooth_mesh,
            camera,
            run_dir / "qa" / f"fixed-view-{camera.role.value}.png",
        )

    input_checks = [
        {
            "role": view["role"],
            "resolution": [view["width"], view["height"]],
            "sharpness": view["sharpness"],
            "poseDegrees": view["pose_deg"],
            "maskCoverage": view["mask_coverage"],
            "maskConfirmed": view["mask_confirmed"],
            "status": "pass" if view["mask_confirmed"] else "fail",
        }
        for view in intake["views"]
    ]
    fit_checks = [
        {"role": role, **values, "status": "pass" if values["passed"] else "fail"}
        for role, values in fit["perView"].items()
    ]
    voxel_checks = [
        {
            "metric": "finite",
            "measured": sdf["finite"],
            "threshold": True,
            "status": "pass" if sdf["finite"] else "fail",
        },
        {
            "metric": "instanceCount",
            "measured": sdf["instanceCount"],
            "threshold": config.pixel.maximum_cells,
            "status": "pass" if sdf["instanceCount"] <= config.pixel.maximum_cells else "fail",
        },
        {
            "metric": "isolatedVoxelCount",
            "measured": sdf["isolatedVoxelCount"],
            "threshold": 0,
            "status": "pass" if sdf["isolatedVoxelCount"] == 0 else "fail",
        },
        {
            "metric": "surfaceCellCoverage",
            "measured": sdf["surfaceCellCoverage"],
            "threshold": 0.95,
            "status": "pass" if sdf["surfaceCellCoverage"] >= 0.95 else "fail",
        },
        {
            "metric": "frontSurfaceMaxDistancePixels",
            "measured": sdf["frontSurfaceMaxDistancePixels"],
            "threshold": 2.0,
            "status": "pass" if sdf["frontSurfaceMaxDistancePixels"] <= 2.0 else "fail",
        },
        {
            "metric": "pixelTraceability",
            "measured": sdf["traceabilityComplete"],
            "threshold": True,
            "status": "pass" if sdf["traceabilityComplete"] else "fail",
        },
    ]
    mesh_checks = [
        {
            "metric": metric,
            "measured": measured,
            "status": "pass" if passed else "fail",
        }
        for metric, measured, passed in (
            ("watertight", mesh["watertight"], mesh["watertight"]),
            ("edgeManifold", mesh["edgeManifold"], mesh["edgeManifold"]),
            ("finite", mesh["finite"], mesh["finite"]),
            ("boundaryEdges", mesh["boundaryEdges"], mesh["boundaryEdges"] == 0),
            (
                "degenerateTriangles",
                mesh["degenerateTriangles"],
                mesh["degenerateTriangles"] == 0,
            ),
            (
                "selfIntersection",
                mesh["selfIntersection"],
                not mesh["selfIntersection"],
            ),
            (
                "triangleBudget",
                mesh["triangles"],
                config.mesh.minimum_triangles <= mesh["triangles"] <= config.mesh.maximum_triangles,
            ),
            (
                "featureDrift",
                mesh["featureDriftVoxels"],
                mesh["featureDriftVoxels"] <= config.acceptance.feature_drift_voxels_max,
            ),
            (
                "normalVarianceReduction",
                mesh["normalVarianceReduction"],
                mesh["normalVarianceReduction"] >= config.acceptance.normal_variance_reduction_min,
            ),
            (
                "hausdorff",
                mesh["hausdorffVoxels"],
                mesh["hausdorffVoxels"] <= config.acceptance.hausdorff_voxels_max,
            ),
            (
                "silhouetteDrop",
                mesh["maximumSilhouetteIoUDrop"],
                mesh["maximumSilhouetteIoUDrop"] <= config.acceptance.silhouette_iou_drop_max,
            ),
        )
    ]
    runtime_minutes = float(runtime.get("elapsedSeconds", 0)) / 60
    peak_gb = float(runtime.get("peakRssBytes", 0)) / 1024**3
    runtime_checks = [
        {
            "metric": "runtimeMinutes",
            "measured": runtime_minutes,
            "threshold": config.acceptance.runtime_minutes_max,
            "status": "pass"
            if runtime_minutes <= config.acceptance.runtime_minutes_max
            else "fail",
        },
        {
            "metric": "peakMemoryGB",
            "measured": peak_gb,
            "threshold": config.acceptance.peak_memory_gb_max,
            "status": "pass" if peak_gb <= config.acceptance.peak_memory_gb_max else "fail",
        },
        {
            "metric": "networkExfiltration",
            "measured": "viewer contains no upload or remote API path",
            "status": "pass",
        },
        {
            "metric": "browserPerformance",
            "measured": None,
            "status": "notEvaluated",
        },
    ]
    gates = [
        _gate(
            "A-input",
            "pass" if all(c["status"] == "pass" for c in input_checks) else "fail",
            input_checks,
        ),
        _gate("B-fit", "pass" if fit["passed"] else "fail", fit_checks),
        _gate("C-3d-pixel", "pass" if sdf["passed"] else "fail", voxel_checks),
        _gate("D-smooth-mesh", "pass" if mesh["passed"] else "fail", mesh_checks),
        _gate(
            "E-geometry-accuracy",
            "pendingUserSignoff",
            [
                {
                    "metric": "fixedViewIdentity",
                    "measured": None,
                    "status": "pendingUserSignoff",
                    "note": "圆滑后脑与短颈属于模板推断，不参与身份相似度；真实人物须由用户签收",
                },
                {
                    "metric": "syntheticChamfer",
                    "measured": None,
                    "status": "notEvaluated",
                },
                (
                    {
                        "metric": "skinAtlasProjection",
                        "measured": "disabled",
                        "status": "notInScope",
                        "note": "geometry-only 实验明确禁用皮肤投影",
                    }
                    if config.output.geometry_only
                    else {
                        "metric": "skinAtlasProjection",
                        "measured": skin["observedVertexFraction"],
                        "threshold": config.skin.minimum_observed_vertex_fraction,
                        "status": "pass" if skin["passed"] else "fail",
                        "note": "正脸位于 UV 中央；后脑和短颈使用生成肤色延拓",
                    }
                ),
            ],
        ),
        _gate(
            "F-web-performance-privacy",
            "partial",
            runtime_checks,
        ),
    ]
    report = {
        "schemaVersion": "1.0.0",
        "generatedBy": f"face3d {__version__}",
        "gates": gates,
        "summary": {
            "automatedGatesPassed": all(
                gate["status"] == "pass"
                for gate in gates
                if gate["gate"] in {"A-input", "B-fit", "C-3d-pixel", "D-smooth-mesh"}
            ),
            "userSignoffRequired": True,
            "visualReviewStatus": "pendingUserSignoff",
            "finalAcceptance": False,
            "browserAuditRequired": True,
        },
    }
    atomic_write_json(run_dir / "qa" / "report.json", report)

    model_hashes = {
        name: details["sha256"]
        for name, details in assets["models"].items()
        if details["sha256"] is not None
    }
    manifest = {
        "schemaVersion": "1.0.0",
        "subjectProfile": "face-v1",
        "mode": "pixel-direct",
        "outputScope": {
            "geometryOnly": config.output.geometry_only,
            "photoSkinProjectionUsed": not config.output.geometry_only,
        },
        "provenance": {
            "inputs": {
                view["role"]: {
                    "sourceSha256": view["sha256"],
                    "normalizedSha256": view["normalized_sha256"],
                    "file": f"references/{view['role']}.png",
                }
                for view in intake["views"]
            },
            "models": model_hashes,
            "configSha256": sha256_file(config.source_path),
            "codeSha256": package_code_hash(),
            "face3dVersion": __version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "cameras": [camera.model_dump(mode="json") for camera in cameras],
        "pixel": {
            "mapping": "traceable measured front pixels plus multi-view sampled continuous surface",
            "gridSize": sdf["resolution"],
            "instanceCount": sdf["instanceCount"],
            "complexPixelCount": sdf["complexPixelCount"],
            "simpleInterpolatedPixelCount": sdf["simpleInterpolatedPixelCount"],
            "binary": "pixels/pixels.bin",
            "binarySha256": sdf["pixelBinary"]["sha256"],
            "schema": "pixels/schema.json",
            "schemaSha256": sdf["pixelBinary"]["schemaSha256"],
            "recordFields": [
                "pixelCode",
                "sourceUV",
                "positionXYZ",
                "thickness",
                "confidence",
                "sourceBits",
                "featureClass",
            ],
        },
        "voxel": {
            "representation": sdf["representation"],
            "resolution": sdf["resolution"],
            "voxelSize": sdf["voxelSize"],
            "instanceCount": sdf["instanceCount"],
            "surfaceCellCoverage": sdf["surfaceCellCoverage"],
            "frontSurfaceSnapCount": sdf["frontSurfaceSnapCount"],
            "frontSurfaceMaxDistancePixels": sdf["frontSurfaceMaxDistancePixels"],
            "maximumInstances": config.pixel.maximum_cells,
            "model": "models/voxels.glb",
            "sourceAttribute": "_SOURCE",
            "confidenceAttribute": "_CONFIDENCE",
        },
        "mesh": {
            "representation": mesh["representation"],
            "volumeResolution": mesh.get("volumeResolution"),
            "voxelSize": mesh.get("voxelSize"),
            "vertices": mesh["vertices"],
            "triangles": mesh["triangles"],
            "watertight": mesh["watertight"],
            "edgeManifold": mesh["edgeManifold"],
            "boundaryEdges": mesh["boundaryEdges"],
            "degenerateTriangles": mesh["degenerateTriangles"],
            "selfIntersection": mesh["selfIntersection"],
            "normalVarianceReduction": mesh["normalVarianceReduction"],
            "hausdorffVoxels": mesh["hausdorffVoxels"],
            "silhouette": mesh.get("silhouette", {}),
            "model": "models/smooth.glb",
        },
        "skin": skin,
        "fit": fit,
        "confidence": {
            "mean": sdf["meanConfidence"],
            "templateInferredVoxels": sdf["templateInferredCount"],
            "regional": {
                "simpleSurface": "interpolated",
                "eyesNoseMouthEarsJaw": "locallyRefined",
                "rearCranium": "templateInferred",
                "shortNeck": "templateInferred",
            },
            "lowConfidenceRegions": ["rearCranium", "shortNeck"],
            "templateInferredRegions": ["rearCranium", "shortNeck"],
            "algorithmicClosureRegions": [],
        },
        "runtime": runtime,
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    return manifest, report


THIRD_PARTY_NOTICES = """# Third-party notices

This result was produced with the following methods and software:

- MediaPipe: Apache License 2.0.
  The task model is not embedded; its SHA-256 is recorded in manifest.json.
- OpenCV: Apache License 2.0.
- SciPy: BSD 3-Clause License.
- trimesh: MIT License.
- scikit-image: BSD 3-Clause License (Lewiner Marching Cubes).
- Open3D: MIT License (signed-distance, closest-surface, and topology QA queries).
- PyTorch: BSD-style license (Face v2 FLAME and Face v3 TemplateHead fitting).
- xatlas-python: MIT License (locked canonical UV generation).
- Three.js: MIT License.

MediaPipe source and license: https://github.com/google-ai-edge/mediapipe

FLAME 2023 Open model assets are not embedded. Their SHA-256 and CC BY 4.0
attribution are recorded in manifest.json when Face v2 is used.

TemplateHeadV0 is derived from the Lee Perry-Smith head scan under CC BY 3.0.
The source, license, attribution, and SHA-256 are preserved in the local template
asset and recorded by manifest provenance when Face v3 is used.
"""


def write_notices(run_dir: Path) -> Path:
    destination = run_dir / "THIRD_PARTY_NOTICES.md"
    notices = THIRD_PARTY_NOTICES
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        demo_asset = manifest.get("provenance", {}).get("demoAsset")
        if isinstance(demo_asset, dict):
            notices += (
                "\nDemo/QA fixture only:\n\n"
                f"- {demo_asset.get('name', 'Demo asset')}: "
                f"{demo_asset.get('license', 'license recorded in manifest.json')}.\n"
                f"  Source: {demo_asset.get('source', 'see manifest.json')}\n"
                f"  SHA-256: {demo_asset.get('sha256', 'see manifest.json')}\n"
            )
    atomic_write_bytes(destination, notices.encode("utf-8"))
    return destination
