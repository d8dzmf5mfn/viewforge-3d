from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import trimesh

from face3d.config import Face3DConfig
from face3d.errors import fail
from face3d.io import atomic_write_json
from face3d.template_head_anatomy import (
    _self_intersection_pairs,
    _top_curvature_spike_ratio,
)
from face3d.template_head_v0 import _edge_and_component_metrics
from face3d.unified_head import EyeballAsset, UnifiedHeadAsset, geometry_hash


def _scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)),
    )
    return scene


def _signed_distance(
    scene: o3d.t.geometry.RaycastingScene,
    points: np.ndarray,
) -> np.ndarray:
    return scene.compute_signed_distance(
        o3d.core.Tensor(np.asarray(points, dtype=np.float32))
    ).numpy()


def _eye_metrics(
    head: UnifiedHeadAsset,
    scene: o3d.t.geometry.RaycastingScene,
    side: str,
    eye: EyeballAsset,
    diagonal: float,
) -> dict[str, Any]:
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=eye.radius)
    sphere_points = np.asarray(sphere.vertices, dtype=np.float64) + eye.center
    signed = _signed_distance(scene, sphere_points)
    intersection_count = int(np.count_nonzero(signed < -diagonal * 1e-7))

    eyelid = np.asarray(head.regions[f"{side}_eyelid"], dtype=np.int64)
    gap_ratio = (
        np.linalg.norm(head.skin_vertices[eyelid] - eye.center, axis=1) - eye.radius
    ) / eye.radius
    contact = gap_ratio[(gap_ratio >= -1e-5) & (gap_ratio <= 0.031)]
    return {
        "completeSphere": True,
        "center": eye.center.astype(float).tolist(),
        "radius": eye.radius,
        "surfaceSampleCount": len(sphere_points),
        "intersectionCount": intersection_count,
        "minimumSignedClearance": float(np.min(signed)),
        "contactVertexCount": len(contact),
        "contactGapMinimumR": float(np.min(contact, initial=np.inf)),
        "contactGapP99R": float(np.quantile(contact, 0.99)) if len(contact) else None,
        "passed": bool(
            intersection_count == 0
            and len(contact) >= 48
            and float(np.min(contact, initial=np.inf)) >= -1e-5
            and float(np.quantile(contact, 0.99)) <= 0.03 + 1e-5
        )
        if len(contact)
        else False,
    }


def run_template_qa(run_dir: Path, config: Face3DConfig) -> dict[str, Any]:
    if not config.is_v3:
        fail(
            "config-invalid",
            "TemplateHeadV0 QA 只接受 face-v3 配置",
            stage="template-qa",
        )
    run_dir = run_dir.expanduser().resolve()
    required = {
        "head": run_dir / "working" / "unified-head.npz",
        "fit": run_dir / "working" / "fit-metrics.json",
        "skin": run_dir / "working" / "skin-metrics.json",
        "model": run_dir / "models" / "head.glb",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        fail(
            "template-qa-upstream-missing",
            "TemplateHeadV0 QA 缺少拟合或皮肤产物",
            stage="template-qa",
            details={"missing": missing},
        )
    fit = json.loads(required["fit"].read_text())
    skin = json.loads(required["skin"].read_text())
    if not fit.get("passed") or not skin.get("passed"):
        fail(
            "template-qa-upstream-failed",
            "拟合或皮肤门禁失败，禁止生成通过状态的 QA",
            stage="template-qa",
            details={"fitPassed": fit.get("passed"), "skinPassed": skin.get("passed")},
        )

    head = UnifiedHeadAsset.load(required["head"])
    vertices = np.asarray(head.skin_vertices, dtype=np.float32).astype(np.float64)
    faces = np.asarray(head.skin_faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    topology = _edge_and_component_metrics(vertices, faces)
    self_intersection_pairs = _self_intersection_pairs(mesh)
    self_intersections = len(self_intersection_pairs)
    involved_faces = np.unique(self_intersection_pairs.reshape(-1))
    involved_vertices = (
        np.unique(faces[involved_faces]) if len(involved_faces) else np.empty(0, dtype=np.int64)
    )
    self_intersection_regions = {
        name: int(
            np.intersect1d(
                involved_vertices,
                np.asarray(indices, dtype=np.int64),
                assume_unique=False,
            ).size
        )
        for name, indices in head.regions.items()
    }
    self_intersection_regions = {
        name: count for name, count in self_intersection_regions.items() if count
    }
    top_curvature_spike_ratio = _top_curvature_spike_ratio(mesh)
    diagonal = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1e-12)
    scene = _scene(mesh)

    face_indices = np.linspace(
        0,
        len(faces) - 1,
        min(len(faces), 4096),
        dtype=np.int64,
    )
    triangles = vertices[faces[face_indices]]
    centers = triangles.mean(axis=1)
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    epsilon = diagonal * 1e-4
    outside = _signed_distance(scene, centers + normals * epsilon)
    inside = _signed_distance(scene, centers - normals * epsilon)
    sign_tolerance = epsilon * 0.20
    outside_consistency = float(np.mean(outside >= -sign_tolerance))
    inside_consistency = float(np.mean(inside <= sign_tolerance))
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    padding = np.ptp(bounds, axis=0) * 0.05
    low = bounds[0] - padding
    high = bounds[1] + padding
    corners = np.asarray(
        [
            (x, y, z)
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=np.float64,
    )
    corner_signed = _signed_distance(scene, corners)
    sdf_finite = bool(
        np.all(np.isfinite(outside))
        and np.all(np.isfinite(inside))
        and np.all(np.isfinite(corner_signed))
    )
    sdf_passed = bool(
        sdf_finite
        and np.all(corner_signed > 0.0)
        and outside_consistency >= 0.99
        and inside_consistency >= 0.99
    )

    left_eye = _eye_metrics(head, scene, "left", head.left_eye, diagonal)
    right_eye = _eye_metrics(head, scene, "right", head.right_eye, diagonal)
    template_eye_anatomy = head.anatomy.get("eyes", {})
    fitting_contours = {
        side: template_eye_anatomy.get(side, {}).get("fittingContour", {})
        for side in ("left", "right")
    }
    contour_symmetry = template_eye_anatomy.get("contourSymmetry", {})
    eyelid_binding_passed = bool(
        all(metrics.get("passed") is True for metrics in fitting_contours.values())
        and contour_symmetry.get("passed") is True
    )
    actual_geometry_hash = geometry_hash(head.render_vertices, head.render_faces)
    geometry_hashes_match = bool(
        actual_geometry_hash
        == head.geometry_sha256
        == fit.get("fittedGeometrySha256")
        == skin.get("geometryHash")
        == skin.get("neutralGeometryHash")
        == skin.get("skinGeometryHash")
    )
    expected_topology: dict[str, int | bool] = {
        "componentCount": 1,
        "boundaryEdgeCount": 0,
        "nonManifoldEdgeCount": 0,
        "degenerateFaceCount": 0,
        "duplicateFaceCount": 0,
        "duplicateVertexCount": 0,
        "watertight": True,
        "windingConsistent": True,
    }
    topology_passed = all(topology[name] == value for name, value in expected_topology.items())
    passed = bool(
        topology_passed
        and self_intersections == 0
        and top_curvature_spike_ratio <= 4.0
        and left_eye["passed"]
        and right_eye["passed"]
        and eyelid_binding_passed
        and sdf_passed
        and geometry_hashes_match
    )
    mesh_metrics = {
        "schemaVersion": 3,
        "representation": "fitted-canonical-template-head-v0",
        "vertices": len(vertices),
        "triangles": len(faces),
        **topology,
        "selfIntersectionPairCount": self_intersections,
        "selfIntersectionInvolvedFaceCount": int(len(involved_faces)),
        "selfIntersectionRegions": self_intersection_regions,
        "selfIntersectionPairSample": self_intersection_pairs[:16].astype(int).tolist(),
        "topCurvatureSpikeRatio": top_curvature_spike_ratio,
        "geometryHash": actual_geometry_hash,
        "geometryHashesMatch": geometry_hashes_match,
        "surfaceGeneratedBySdf": False,
        "passed": bool(
            topology_passed
            and self_intersections == 0
            and top_curvature_spike_ratio <= 4.0
            and geometry_hashes_match
        ),
    }
    sdf_metrics = {
        "schemaVersion": 3,
        "role": "qa-only",
        "surfaceGenerated": False,
        "gridAllocated": False,
        "queryCount": int(len(outside) + len(inside) + len(corner_signed)),
        "finite": sdf_finite,
        "cornerOutsidePositive": bool(np.all(corner_signed > 0.0)),
        "outsideSignConsistency": outside_consistency,
        "insideSignConsistency": inside_consistency,
        "signConsistencyThreshold": 0.99,
        "passed": sdf_passed,
    }
    anatomy = {
        "schemaVersion": 3,
        "templateId": "TemplateHeadV0",
        "geometry": mesh_metrics,
        "eyes": {
            "completeEyeballNodes": 2,
            "left": left_eye,
            "right": right_eye,
            "fittingContours": fitting_contours,
            "contourSymmetry": contour_symmetry,
            "fittingBindingPassed": eyelid_binding_passed,
            "intersectionCount": int(
                left_eye["intersectionCount"] + right_eye["intersectionCount"]
            ),
        },
        "ears": {
            "carrierPresent": False,
            "rootSharedWithScalp": True,
            "leftVertexCount": len(head.regions["left_ear"]),
            "rightVertexCount": len(head.regions["right_ear"]),
        },
        "skin": {
            "geometryHash": skin["geometryHash"],
            "neutralGeometryHash": skin["neutralGeometryHash"],
            "maximumVertexDifference": skin["maximumVertexDifference"],
        },
        "sdf": sdf_metrics,
        "passed": passed,
    }
    atomic_write_json(run_dir / "working" / "mesh-metrics.json", mesh_metrics)
    atomic_write_json(run_dir / "working" / "sdf-metrics.json", sdf_metrics)
    atomic_write_json(run_dir / "qa" / "anatomy.json", anatomy)
    if not passed:
        fail(
            "template-qa-gate-failed",
            "TemplateHeadV0 拓扑、眼球接触或 SDF 距离 QA 未通过",
            stage="template-qa",
            details={
                "mesh": mesh_metrics,
                "sdf": sdf_metrics,
                "leftEye": left_eye,
                "rightEye": right_eye,
                "fittingContours": fitting_contours,
                "contourSymmetry": contour_symmetry,
            },
        )
    return anatomy
