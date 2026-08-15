from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from face3d.config import Face3DConfig
from face3d.errors import fail
from face3d.glb import export_neutral_mesh
from face3d.io import atomic_write_json
from face3d.models import CameraRecord, ViewRole
from face3d.stages.fit import _render_silhouette, _silhouette_iou


def _normal_variation(mesh: trimesh.Trimesh) -> float:
    edges = mesh.edges_unique
    normals = np.asarray(mesh.vertex_normals)
    dots = np.sum(normals[edges[:, 0]] * normals[edges[:, 1]], axis=1)
    return float(np.mean(1 - np.clip(dots, -1, 1)))


def _weighted_laplacian_step(
    vertices: np.ndarray,
    edges: np.ndarray,
    amount: float,
    weights: np.ndarray,
) -> np.ndarray:
    neighbor_sum = np.zeros_like(vertices)
    counts = np.zeros(len(vertices), dtype=np.float64)
    np.add.at(neighbor_sum, edges[:, 0], vertices[edges[:, 1]])
    np.add.at(neighbor_sum, edges[:, 1], vertices[edges[:, 0]])
    np.add.at(counts, edges[:, 0], 1)
    np.add.at(counts, edges[:, 1], 1)
    mean = neighbor_sum / np.maximum(counts[:, None], 1)
    return vertices + amount * weights[:, None] * (mean - vertices)


def _feature_weights(
    vertices: np.ndarray,
    feature_points: np.ndarray,
    radius: float,
) -> np.ndarray:
    distances, _ = cKDTree(feature_points).query(vertices, k=1)
    normalized = np.clip(distances / max(radius, 1e-12), 0, 1)
    return 0.12 + 0.88 * normalized * normalized * (3 - 2 * normalized)


def _taubin(
    mesh: trimesh.Trimesh,
    feature_points: np.ndarray,
    radius: float,
    iterations: int,
    lambda_value: float,
    nu_value: float,
) -> tuple[trimesh.Trimesh, float]:
    vertices = np.asarray(mesh.vertices).copy()
    edges = np.asarray(mesh.edges_unique)
    weights = _feature_weights(vertices, feature_points, radius)
    feature_indices = cKDTree(vertices).query(feature_points, k=1)[1]
    original_features = vertices[feature_indices].copy()
    for _ in range(iterations):
        vertices = _weighted_laplacian_step(vertices, edges, lambda_value, weights)
        vertices = _weighted_laplacian_step(vertices, edges, nu_value, weights)
    drift = float(np.max(np.linalg.norm(vertices[feature_indices] - original_features, axis=1)))
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False), drift


def _main_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    components = mesh.split(only_watertight=False)
    if not components:
        fail("mesh-gate-failed", "Marching Cubes 未产生连通表面", stage="mesh")
    return max(components, key=lambda value: len(value.faces))


def _approximate_hausdorff(first: trimesh.Trimesh, second: trimesh.Trimesh) -> float:
    first_vertices = np.asarray(first.vertices)
    second_vertices = np.asarray(second.vertices)
    first_stride = max(1, len(first_vertices) // 30000)
    second_stride = max(1, len(second_vertices) // 30000)
    first_sample = first_vertices[::first_stride]
    second_sample = second_vertices[::second_stride]
    forward = cKDTree(second_sample).query(first_sample, k=1)[0].max()
    backward = cKDTree(first_sample).query(second_sample, k=1)[0].max()
    return float(max(forward, backward))


def _degenerate_count(mesh: trimesh.Trimesh) -> int:
    triangles = np.asarray(mesh.triangles)
    double_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    return int(np.count_nonzero(double_area <= 1e-12))


def run_mesh(run_dir: Path, config: Face3DConfig) -> dict[str, Any]:
    sdf_payload = np.load(run_dir / "working" / "sdf.npz")
    sdf = sdf_payload["sdf"]
    voxel_size = float(sdf_payload["voxel_size"])
    grid_min = sdf_payload["grid_min"].astype(np.float64)
    vertices_zyx, faces, _, _ = marching_cubes(
        sdf,
        level=0.0,
        spacing=(voxel_size, voxel_size, voxel_size),
        method="lewiner",
        gradient_direction="ascent",
        allow_degenerate=False,
    )
    vertices = vertices_zyx[:, [2, 1, 0]] + grid_min + voxel_size * 0.5
    raw = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=True)
    raw = _main_component(raw)
    if raw.volume < 0:
        raw.invert()
    raw.remove_unreferenced_vertices()
    raw.update_faces(raw.nondegenerate_faces())
    raw.merge_vertices()
    trimesh.repair.fix_normals(raw, multibody=True)
    fit = np.load(run_dir / "working" / "fit.npz")
    feature_points = fit["feature_landmarks"][0].astype(np.float64)
    before_variation = _normal_variation(raw)
    smoothed, feature_drift = _taubin(
        raw,
        feature_points,
        config.mesh.feature_protection_radius_voxels * voxel_size,
        config.mesh.taubin_iterations,
        config.mesh.taubin_lambda,
        config.mesh.taubin_nu,
    )
    after_variation_pre_simplify = _normal_variation(smoothed)
    if len(smoothed.faces) > config.mesh.target_triangles:
        smoothed = smoothed.simplify_quadric_decimation(
            face_count=config.mesh.target_triangles, aggression=5
        )
    smoothed = trimesh.Trimesh(
        vertices=np.asarray(smoothed.vertices),
        faces=np.asarray(smoothed.faces),
        process=True,
        validate=True,
    )
    smoothed.update_faces(smoothed.nondegenerate_faces())
    smoothed.update_faces(smoothed.unique_faces())
    smoothed.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(smoothed, multibody=True)
    if smoothed.volume < 0:
        smoothed.invert()
    after_variation = _normal_variation(smoothed)
    normal_reduction = float(1 - after_variation_pre_simplify / max(before_variation, 1e-12))
    hausdorff = _approximate_hausdorff(raw, smoothed)

    cameras_payload = json.loads((run_dir / "working" / "cameras.json").read_text())
    cameras = [CameraRecord.model_validate(item) for item in cameras_payload["cameras"]]
    intake = json.loads((run_dir / "working" / "intake.json").read_text())
    view_by_role = {ViewRole(item["role"]): item for item in intake["views"]}
    silhouette_metrics: dict[str, Any] = {}
    maximum_iou_drop = 0.0
    for camera in cameras:
        target = cv2.imread(view_by_role[camera.role]["mask_path"], cv2.IMREAD_GRAYSCALE)
        raw_iou = _silhouette_iou(
            _render_silhouette(np.asarray(raw.vertices), np.asarray(raw.faces), camera), target
        )
        smooth_iou = _silhouette_iou(
            _render_silhouette(np.asarray(smoothed.vertices), np.asarray(smoothed.faces), camera),
            target,
        )
        drop = max(0.0, raw_iou - smooth_iou)
        maximum_iou_drop = max(maximum_iou_drop, drop)
        silhouette_metrics[camera.role.value] = {
            "rawIoU": raw_iou,
            "smoothIoU": smooth_iou,
            "drop": drop,
        }

    degenerate = _degenerate_count(smoothed)
    boundary_edges = int(len(_boundary_edges(smoothed)))
    finite = bool(np.isfinite(smoothed.vertices).all())
    open3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(smoothed.vertices)),
        o3d.utility.Vector3iVector(np.asarray(smoothed.faces, dtype=np.int32)),
    )
    self_intersection = bool(open3d_mesh.is_self_intersecting())
    metrics = {
        "vertices": int(len(smoothed.vertices)),
        "triangles": int(len(smoothed.faces)),
        "watertight": bool(smoothed.is_watertight),
        "windingConsistent": bool(smoothed.is_winding_consistent),
        "edgeManifold": bool(smoothed.is_watertight),
        "boundaryEdges": boundary_edges,
        "degenerateTriangles": degenerate,
        "finite": finite,
        "selfIntersection": self_intersection,
        "featureDriftVoxels": feature_drift / voxel_size,
        "normalVariationBefore": before_variation,
        "normalVariationAfter": after_variation,
        "normalVarianceReduction": normal_reduction,
        "hausdorffVoxels": hausdorff / voxel_size,
        "silhouette": silhouette_metrics,
        "maximumSilhouetteIoUDrop": maximum_iou_drop,
    }
    failures: list[str] = []
    if not config.mesh.minimum_triangles <= len(smoothed.faces) <= config.mesh.maximum_triangles:
        failures.append("triangle-budget")
    if (
        not smoothed.is_watertight
        or boundary_edges
        or degenerate
        or not finite
        or self_intersection
    ):
        failures.append("topology")
    if feature_drift / voxel_size > config.acceptance.feature_drift_voxels_max:
        failures.append("feature-drift")
    if normal_reduction < config.acceptance.normal_variance_reduction_min:
        failures.append("normal-variance")
    if hausdorff / voxel_size > config.acceptance.hausdorff_voxels_max:
        failures.append("hausdorff")
    if maximum_iou_drop > config.acceptance.silhouette_iou_drop_max:
        failures.append("silhouette-drop")
    metrics["passed"] = not failures
    metrics["failures"] = failures
    export_neutral_mesh(raw, run_dir / "models" / "raw-isosurface.glb")
    export_neutral_mesh(smoothed, run_dir / "models" / "smooth.glb")
    np.savez_compressed(
        run_dir / "working" / "smooth-mesh.npz",
        vertices=np.asarray(smoothed.vertices, dtype=np.float32),
        faces=np.asarray(smoothed.faces, dtype=np.int32),
    )
    atomic_write_json(run_dir / "working" / "mesh-metrics.json", metrics)
    if failures:
        fail(
            "mesh-gate-failed",
            "平滑网格未通过 Gate D",
            stage="mesh",
            details={"failures": failures, "metrics": metrics},
        )
    return metrics


def _boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    edges = np.sort(np.asarray(mesh.edges), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]
