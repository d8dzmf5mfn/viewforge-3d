from __future__ import annotations

import copy
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from trimesh.interfaces.blender import _blender_executable
from trimesh.interfaces.generic import MeshScript
from trimesh.resources import get_string

from face3d.errors import fail
from face3d.glb import export_neutral_mesh
from face3d.io import atomic_write_json, sha256_file
from face3d.models import CameraRecord
from face3d.render import render_flat_mesh
from face3d.stages.intake import _detect, face_landmarker
from face3d.template_head_v0 import (
    RawTemplateHeadV0,
    _edge_and_component_metrics,
    _extract_raw_asset,
)
from face3d.unified_head import EyeballAsset, UnifiedHeadAsset, geometry_hash

ANATOMY_SCHEMA_VERSION = "0.4.0"
ANATOMY_DIRECTORY = "anatomy"
STABILIZED_FACE_COUNT = 21_400
ANATOMICAL_EYE_LANDMARKS = {
    # MediaPipe names these from the subject's perspective.
    "right": (33, 133, 159, 145),
    "left": (263, 362, 386, 374),
}
EYELID_CONTOUR_LANDMARKS = {
    "right": (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
    "left": (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466),
}
IRIS_LANDMARKS = tuple(range(468, 478))
REQUIRED_SEMANTIC_REGIONS = (
    "face",
    "cranium",
    "rear_cranium",
    "neck_shoulders",
    "left_ear",
    "right_ear",
    "left_eye",
    "right_eye",
    "left_eyelid",
    "right_eyelid",
    "left_eyelid_ring",
    "right_eyelid_ring",
    "nose",
    "mouth",
    "jaw",
)


@dataclass(frozen=True, slots=True)
class EyeSurgerySpec:
    anatomical_side: str
    center: np.ndarray
    eyeball_radius: float
    socket_radius: float
    deep_center: np.ndarray
    deep_radius: float
    slot_center: np.ndarray
    slot_axes: np.ndarray
    landmark_indices: tuple[int, int, int, int]
    landmark_width: float

    def as_json(self) -> dict[str, Any]:
        return {
            "anatomicalSide": self.anatomical_side,
            "center": self.center.astype(float).tolist(),
            "eyeballRadius": self.eyeball_radius,
            "socketRadius": self.socket_radius,
            "deepCenter": self.deep_center.astype(float).tolist(),
            "deepRadius": self.deep_radius,
            "slotCenter": self.slot_center.astype(float).tolist(),
            "slotAxes": self.slot_axes.astype(float).tolist(),
            "landmarkIndices": list(self.landmark_indices),
            "landmarkWidth": self.landmark_width,
        }


@dataclass(frozen=True, slots=True)
class LandmarkSurfaceMap:
    normalized: np.ndarray
    points: np.ndarray
    valid: np.ndarray
    triangle: np.ndarray
    barycentric: np.ndarray
    reprojection_error_px: np.ndarray


@dataclass(frozen=True, slots=True)
class BooleanCleanup:
    raw_component_count: int
    discarded_component_count: int
    discarded_face_count: int
    maximum_artifact_face_count: int
    discarded_area_ratio: float
    discarded_volume_ratio: float
    degenerate_face_count_removed: int
    boundary_edge_count_filled: int
    face_count_after: int

    def as_json(self) -> dict[str, int]:
        return {
            "rawComponentCount": self.raw_component_count,
            "discardedComponentCount": self.discarded_component_count,
            "discardedFaceCount": self.discarded_face_count,
            "maximumArtifactFaceCount": self.maximum_artifact_face_count,
            "discardedAreaRatio": self.discarded_area_ratio,
            "discardedVolumeRatio": self.discarded_volume_ratio,
            "degenerateFaceCountRemoved": self.degenerate_face_count_removed,
            "boundaryEdgeCountFilled": self.boundary_edge_count_filled,
            "faceCountAfter": self.face_count_after,
        }


@dataclass(frozen=True, slots=True)
class TopologyStabilization:
    method: str
    source_vertex_count: int
    source_face_count: int
    requested_face_count: int
    output_vertex_count: int
    output_face_count: int
    minimum_relative_area_before: float
    minimum_relative_area_after: float
    float32_minimum_relative_area: float
    self_intersection_pair_count_before: int
    self_intersection_pair_count_after: int
    float32_self_intersection_pair_count: int
    bounds_maximum_drift: float
    output_to_source_p99_diagonal: float
    output_to_source_maximum_diagonal: float
    source_to_output_p99_diagonal: float
    source_to_output_maximum_diagonal: float

    def as_json(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "sourceVertexCount": self.source_vertex_count,
            "sourceFaceCount": self.source_face_count,
            "requestedFaceCount": self.requested_face_count,
            "outputVertexCount": self.output_vertex_count,
            "outputFaceCount": self.output_face_count,
            "minimumRelativeAreaBefore": self.minimum_relative_area_before,
            "minimumRelativeAreaAfter": self.minimum_relative_area_after,
            "float32MinimumRelativeArea": self.float32_minimum_relative_area,
            "selfIntersectionPairCountBefore": self.self_intersection_pair_count_before,
            "selfIntersectionPairCountAfter": self.self_intersection_pair_count_after,
            "float32SelfIntersectionPairCount": self.float32_self_intersection_pair_count,
            "boundsMaximumDrift": self.bounds_maximum_drift,
            "outputToSourceP99Diagonal": self.output_to_source_p99_diagonal,
            "outputToSourceMaximumDiagonal": self.output_to_source_maximum_diagonal,
            "sourceToOutputP99Diagonal": self.source_to_output_p99_diagonal,
            "sourceToOutputMaximumDiagonal": self.source_to_output_maximum_diagonal,
            "sdfUsed": False,
        }


BooleanDifference = Callable[
    [trimesh.Trimesh, trimesh.Trimesh],
    tuple[trimesh.Trimesh, BooleanCleanup],
]


def _o3d_scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(
            np.asarray(mesh.vertices, dtype=np.float32),
            dtype=o3d.core.Dtype.Float32,
        ),
        o3d.core.Tensor(
            np.asarray(mesh.faces, dtype=np.uint32),
            dtype=o3d.core.Dtype.UInt32,
        ),
    )
    return scene


def _project(points: np.ndarray, camera: CameraRecord) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(camera.rotation_vector, dtype=np.float64))
    camera_points = np.asarray(points, dtype=np.float64) @ rotation.T + np.asarray(
        camera.translation,
        dtype=np.float64,
    )
    depth = camera_points[:, 2]
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = depth > 1e-8
    pixels[valid] = camera_points[valid, :2] / depth[valid, None]
    pixels[valid] *= camera.focal_length_px
    pixels[valid] += np.asarray(camera.principal_point_px, dtype=np.float64)
    return pixels


def map_landmarks_to_template(
    mesh: trimesh.Trimesh,
    camera: CameraRecord,
    normalized: np.ndarray,
) -> LandmarkSurfaceMap:
    normalized = np.asarray(normalized, dtype=np.float64)
    if normalized.ndim != 2 or normalized.shape[1] < 2 or len(normalized) < 468:
        fail(
            "template-landmarks-invalid",
            "TemplateHeadV0 语义映射需要至少 468 个 MediaPipe 地标",
            stage="template-head-anatomy",
            details={"shape": list(normalized.shape)},
        )
    if not np.isfinite(normalized).all():
        fail(
            "template-landmarks-invalid",
            "TemplateHeadV0 地标包含 NaN 或 Inf",
            stage="template-head-anatomy",
        )

    rotation, _ = cv2.Rodrigues(np.asarray(camera.rotation_vector, dtype=np.float64))
    translation = np.asarray(camera.translation, dtype=np.float64)
    origin = -rotation.T @ translation
    pixels = normalized[:, :2] * np.asarray([camera.width, camera.height], dtype=np.float64)
    camera_directions = np.column_stack(
        (
            (pixels[:, 0] - camera.principal_point_px[0]) / camera.focal_length_px,
            (pixels[:, 1] - camera.principal_point_px[1]) / camera.focal_length_px,
            np.ones(len(pixels), dtype=np.float64),
        )
    )
    directions = camera_directions @ rotation
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    rays = np.column_stack((np.broadcast_to(origin, directions.shape), directions)).astype(
        np.float32
    )
    result = _o3d_scene(mesh).cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    distance = result["t_hit"].numpy()
    valid = np.isfinite(distance)
    points = np.full((len(normalized), 3), np.nan, dtype=np.float64)
    points[valid] = origin + directions[valid] * distance[valid, None]
    triangle = result["primitive_ids"].numpy().astype(np.uint32)
    primitive_uv = result["primitive_uvs"].numpy().astype(np.float64)
    barycentric = np.column_stack((1.0 - primitive_uv[:, 0] - primitive_uv[:, 1], primitive_uv))
    barycentric[~valid] = np.nan
    reprojected = _project(points, camera)
    error = np.full(len(normalized), np.nan, dtype=np.float64)
    error[valid] = np.linalg.norm(reprojected[valid] - pixels[valid], axis=1)
    return LandmarkSurfaceMap(
        normalized=normalized,
        points=points,
        valid=valid,
        triangle=triangle,
        barycentric=barycentric,
        reprojection_error_px=error,
    )


def derive_eye_specs(surface_map: LandmarkSurfaceMap) -> dict[str, EyeSurgerySpec]:
    for side, indices in ANATOMICAL_EYE_LANDMARKS.items():
        if not np.all(surface_map.valid[np.asarray(indices, dtype=np.int64)]):
            fail(
                "template-eye-landmark-miss",
                f"{side} 眼部地标没有全部命中 TemplateHeadV0",
                stage="template-head-anatomy",
                details={"indices": list(indices)},
            )

    widths: dict[str, float] = {}
    corner_centers: dict[str, np.ndarray] = {}
    for side, indices in ANATOMICAL_EYE_LANDMARKS.items():
        corners = surface_map.points[np.asarray(indices[:2], dtype=np.int64)]
        widths[side] = float(np.linalg.norm(corners[0] - corners[1]))
        corner_centers[side] = corners.mean(axis=0)
    interocular = float(np.linalg.norm(corner_centers["left"] - corner_centers["right"]))
    raw_radius = float(np.mean(list(widths.values())) * 0.58)
    shared_radius = float(np.clip(raw_radius, interocular * 0.22, interocular * 0.30))
    if not np.isfinite(shared_radius) or shared_radius <= 0:
        fail(
            "template-eye-radius-invalid",
            "无法从 TemplateHeadV0 地标得到有效眼球半径",
            stage="template-head-anatomy",
        )

    specs: dict[str, EyeSurgerySpec] = {}
    for side, indices in ANATOMICAL_EYE_LANDMARKS.items():
        center = corner_centers[side].copy()
        center[2] -= shared_radius * 0.95
        width = widths[side]
        deep_center = center.copy()
        deep_center[2] -= shared_radius * 1.83
        slot_center = center.copy()
        slot_center[2] += shared_radius * 0.57
        specs[side] = EyeSurgerySpec(
            anatomical_side=side,
            center=center,
            eyeball_radius=shared_radius,
            socket_radius=shared_radius * 1.029,
            deep_center=deep_center,
            deep_radius=shared_radius * 1.33,
            slot_center=slot_center,
            slot_axes=np.asarray(
                [width * 0.54, width * 0.106, shared_radius * 2.67],
                dtype=np.float64,
            ),
            landmark_indices=indices,
            landmark_width=width,
        )
    if specs["left"].center[0] <= specs["right"].center[0]:
        fail(
            "template-eye-side-invalid",
            "TemplateHeadV0 左右眼解剖方向映射错误",
            stage="template-head-anatomy",
        )
    return specs


def remap_landmarks_to_anatomy(
    source: LandmarkSurfaceMap,
    mesh: trimesh.Trimesh,
    camera: CameraRecord,
    rings: dict[str, np.ndarray],
) -> tuple[LandmarkSurfaceMap, np.ndarray]:
    """Bind source landmarks to the post-surgery skin topology.

    The original ray hits are only used to construct the eye cutters. After the
    booleans, triangle ids are no longer stable. All skin landmarks are rebound
    by closest point and eyelid contours are explicitly snapped to their real
    interface cycles. Iris landmarks are excluded because they belong to the
    independent eyeball nodes.
    """

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    count = len(source.normalized)
    valid = np.asarray(source.valid, dtype=bool).copy()
    valid[np.asarray(IRIS_LANDMARKS, dtype=np.int64)] = False
    points = np.full((count, 3), np.nan, dtype=np.float64)
    triangle = np.full(count, np.iinfo(np.uint32).max, dtype=np.uint32)
    barycentric = np.full((count, 3), np.nan, dtype=np.float64)
    offset = np.full(count, np.nan, dtype=np.float64)

    selected = np.flatnonzero(valid)
    closest = _o3d_scene(mesh).compute_closest_points(
        o3d.core.Tensor(
            np.asarray(source.points[selected], dtype=np.float32),
            dtype=o3d.core.Dtype.Float32,
        )
    )
    primitive = closest["primitive_ids"].numpy().astype(np.uint32)
    primitive_uv = closest["primitive_uvs"].numpy().astype(np.float64)
    mapped = closest["points"].numpy().astype(np.float64)
    finite = np.all(np.isfinite(mapped), axis=1) & (primitive != np.iinfo(np.uint32).max)
    valid[selected[~finite]] = False
    selected = selected[finite]
    mapped = mapped[finite]
    primitive = primitive[finite]
    primitive_uv = primitive_uv[finite]
    points[selected] = mapped
    triangle[selected] = primitive
    barycentric[selected] = np.column_stack(
        (1.0 - primitive_uv[:, 0] - primitive_uv[:, 1], primitive_uv)
    )

    incident_faces: dict[int, np.ndarray] = {}
    for side, landmark_indices in EYELID_CONTOUR_LANDMARKS.items():
        ring = np.asarray(rings[side], dtype=np.int64)
        ring_points = vertices[ring]
        for landmark_index in landmark_indices:
            if not valid[landmark_index]:
                continue
            vertex_index = int(
                ring[
                    np.argmin(
                        np.linalg.norm(
                            ring_points - source.points[landmark_index],
                            axis=1,
                        )
                    )
                ]
            )
            candidates = incident_faces.get(vertex_index)
            if candidates is None:
                candidates = np.flatnonzero(np.any(faces == vertex_index, axis=1))
                incident_faces[vertex_index] = candidates
            if not len(candidates):
                fail(
                    "template-eyelid-landmark-unbound",
                    f"{side} 眼睑环顶点没有相邻三角面",
                    stage="template-head-anatomy",
                    details={"vertex": vertex_index, "landmark": landmark_index},
                )
            face_index = int(candidates[0])
            corner = int(np.flatnonzero(faces[face_index] == vertex_index)[0])
            weights = np.zeros(3, dtype=np.float64)
            weights[corner] = 1.0
            points[landmark_index] = vertices[vertex_index]
            triangle[landmark_index] = face_index
            barycentric[landmark_index] = weights

    valid_indices = np.flatnonzero(valid)
    reconstructed = np.einsum(
        "nvc,nv->nc",
        vertices[faces[triangle[valid_indices]]],
        barycentric[valid_indices],
    )
    reconstruction_error = np.linalg.norm(reconstructed - points[valid_indices], axis=1)
    reconstruction_tolerance = max(
        float(np.linalg.norm(np.ptp(vertices, axis=0))) * 1e-7,
        1e-7,
    )
    if np.max(reconstruction_error, initial=0.0) > reconstruction_tolerance:
        fail(
            "template-landmark-remap-invalid",
            "手术后地标的三角面重心坐标无法重建表面点",
            stage="template-head-anatomy",
            details={
                "maximumError": float(np.max(reconstruction_error)),
                "tolerance": reconstruction_tolerance,
            },
        )
    offset[valid_indices] = np.linalg.norm(
        points[valid_indices] - source.points[valid_indices],
        axis=1,
    )
    pixels = source.normalized[:, :2] * np.asarray(
        [camera.width, camera.height],
        dtype=np.float64,
    )
    reprojected = _project(points, camera)
    reprojection_error = np.full(count, np.nan, dtype=np.float64)
    reprojection_error[valid_indices] = np.linalg.norm(
        reprojected[valid_indices] - pixels[valid_indices],
        axis=1,
    )
    return (
        LandmarkSurfaceMap(
            normalized=np.asarray(source.normalized, dtype=np.float64),
            points=points,
            valid=valid,
            triangle=triangle,
            barycentric=barycentric,
            reprojection_error_px=reprojection_error,
        ),
        offset,
    )


def _eye_cutter(spec: EyeSurgerySpec) -> trimesh.Trimesh:
    deep = trimesh.creation.icosphere(subdivisions=3, radius=spec.deep_radius)
    deep.apply_translation(spec.deep_center)
    socket = trimesh.creation.icosphere(
        subdivisions=3,
        radius=spec.socket_radius,
    )
    socket.apply_translation(spec.center)
    slot = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    slot.vertices *= spec.slot_axes
    slot.apply_translation(spec.slot_center)
    try:
        cutter = trimesh.boolean.union(
            [deep, socket, slot],
            engine="blender",
            check_volume=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        fail(
            "template-eye-cutter-union-failed",
            f"{spec.anatomical_side} 眼部切割体合并失败",
            stage="template-head-anatomy",
            details={"reason": str(exc)},
        )
    if not isinstance(cutter, trimesh.Trimesh) or not cutter.is_volume:
        fail(
            "template-eye-cutter-invalid",
            f"{spec.anatomical_side} 眼部切割体不是单一正体积闭合网格",
            stage="template-head-anatomy",
        )
    if len(cutter.split(only_watertight=False)) != 1:
        fail(
            "template-eye-cutter-invalid",
            f"{spec.anatomical_side} 眼部切割体不连续",
            stage="template-head-anatomy",
        )
    return cutter


def _edge_counts(mesh: trimesh.Trimesh) -> tuple[int, int]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.sort(
        np.concatenate(
            (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]),
            axis=0,
        ),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts == 1)), int(np.count_nonzero(counts > 2))


def _relative_triangle_area_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> dict[str, float | int]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    triangles = vertices[faces]
    doubled_area = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    median = max(float(np.median(doubled_area)), np.finfo(np.float64).tiny)
    relative = doubled_area / median
    return {
        "minimumDoubledArea": float(np.min(doubled_area)),
        "medianDoubledArea": median,
        "minimumRelativeArea": float(np.min(relative)),
        "belowOneE4Count": int(np.count_nonzero(relative <= 1e-4)),
        "belowOneE6Count": int(np.count_nonzero(relative <= 1e-6)),
    }


def _o3d_legacy_mesh(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )


def _self_intersection_pairs(mesh: trimesh.Trimesh) -> np.ndarray:
    pairs = np.asarray(
        _o3d_legacy_mesh(mesh).get_self_intersecting_triangles(),
        dtype=np.int64,
    )
    return pairs.reshape((-1, 2)) if pairs.size else np.empty((0, 2), dtype=np.int64)


def _self_intersection_pair_count(mesh: trimesh.Trimesh) -> int:
    return len(_self_intersection_pairs(mesh))


def _normalized_surface_distance(
    surface: trimesh.Trimesh,
    query_points: np.ndarray,
    diagonal: float,
) -> tuple[float, float]:
    distance = _surface_distance(surface, query_points) / diagonal
    return float(np.quantile(distance, 0.99)), float(np.max(distance, initial=0.0))


def _stabilize_boolean_topology(
    source: trimesh.Trimesh,
    *,
    target_face_count: int = 21_400,
    minimum_relative_area: float = 5e-4,
    maximum_surface_p99_diagonal: float = 5e-5,
    maximum_surface_distance_diagonal: float = 2e-4,
    maximum_bounds_drift: float = 1e-6,
) -> tuple[trimesh.Trimesh, TopologyStabilization]:
    """Collapse exact-boolean sliver faces without replacing the template surface."""

    source_vertices = np.asarray(source.vertices, dtype=np.float64)
    source_faces = np.asarray(source.faces, dtype=np.int64)
    if target_face_count <= 0 or target_face_count >= len(source_faces):
        fail(
            "template-topology-stabilization-target-invalid",
            "眼睑拓扑稳定化目标面数必须小于布尔结果",
            stage="template-head-anatomy",
            details={
                "sourceFaceCount": len(source_faces),
                "targetFaceCount": target_face_count,
            },
        )
    source_topology = _edge_and_component_metrics(source_vertices, source_faces)
    if (
        source_topology["componentCount"] != 1
        or source_topology["boundaryEdgeCount"] != 0
        or source_topology["nonManifoldEdgeCount"] != 0
        or source_topology["degenerateFaceCount"] != 0
        or not source.is_watertight
        or not source.is_winding_consistent
        or float(source.volume) <= 0.0
    ):
        fail(
            "template-topology-stabilization-input-invalid",
            "眼睑拓扑稳定化输入不是单连通闭合正体积网格",
            stage="template-head-anatomy",
            details=source_topology,
        )

    simplified = _o3d_legacy_mesh(source).simplify_quadric_decimation(target_face_count)
    candidate = trimesh.Trimesh(
        vertices=np.asarray(simplified.vertices, dtype=np.float64),
        faces=np.asarray(simplified.triangles, dtype=np.int64),
        process=False,
        validate=False,
    )
    vertices = np.asarray(candidate.vertices, dtype=np.float64)
    faces = np.asarray(candidate.faces, dtype=np.int64)
    topology = _edge_and_component_metrics(vertices, faces)
    area_before = _relative_triangle_area_metrics(source_vertices, source_faces)
    area_after = _relative_triangle_area_metrics(vertices, faces)
    source_intersections = _self_intersection_pair_count(source)
    intersections = _self_intersection_pair_count(candidate)
    bounds_drift = float(np.max(np.abs(candidate.bounds - source.bounds)))
    diagonal = max(float(np.linalg.norm(np.ptp(source_vertices, axis=0))), 1e-12)
    output_to_source_p99, output_to_source_maximum = _normalized_surface_distance(
        source,
        vertices,
        diagonal,
    )
    source_to_output_p99, source_to_output_maximum = _normalized_surface_distance(
        candidate,
        source_vertices,
        diagonal,
    )

    delivery_vertices = vertices.astype(np.float32).astype(np.float64)
    delivery = trimesh.Trimesh(
        vertices=delivery_vertices,
        faces=faces,
        process=False,
        validate=False,
    )
    delivery_topology = _edge_and_component_metrics(delivery_vertices, faces)
    delivery_area = _relative_triangle_area_metrics(delivery_vertices, faces)
    delivery_intersections = _self_intersection_pair_count(delivery)

    failures: dict[str, Any] = {}
    required_topology: dict[str, int | bool] = {
        "componentCount": 1,
        "boundaryEdgeCount": 0,
        "nonManifoldEdgeCount": 0,
        "degenerateFaceCount": 0,
        "duplicateFaceCount": 0,
        "duplicateVertexCount": 0,
        "watertight": True,
        "windingConsistent": True,
    }
    mismatched = {
        name: {"expected": expected, "actual": topology[name]}
        for name, expected in required_topology.items()
        if topology[name] != expected
    }
    delivery_mismatched = {
        name: {"expected": expected, "actual": delivery_topology[name]}
        for name, expected in required_topology.items()
        if delivery_topology[name] != expected
    }
    if len(faces) != target_face_count:
        failures["faceCount"] = {"expected": target_face_count, "actual": len(faces)}
    if mismatched:
        failures["topology"] = mismatched
    if delivery_mismatched:
        failures["float32Topology"] = delivery_mismatched
    if float(candidate.volume) <= 0.0 or float(delivery.volume) <= 0.0:
        failures["positiveVolume"] = {
            "float64": float(candidate.volume),
            "float32": float(delivery.volume),
        }
    if intersections or delivery_intersections:
        failures["selfIntersections"] = {
            "float64PairCount": intersections,
            "float32PairCount": delivery_intersections,
        }
    if float(area_after["minimumRelativeArea"]) < minimum_relative_area:
        failures["minimumRelativeArea"] = {
            "threshold": minimum_relative_area,
            "actual": area_after["minimumRelativeArea"],
        }
    if float(delivery_area["minimumRelativeArea"]) < minimum_relative_area:
        failures["float32MinimumRelativeArea"] = {
            "threshold": minimum_relative_area,
            "actual": delivery_area["minimumRelativeArea"],
        }
    if bounds_drift > maximum_bounds_drift:
        failures["boundsDrift"] = {
            "threshold": maximum_bounds_drift,
            "actual": bounds_drift,
        }
    if max(output_to_source_p99, source_to_output_p99) > maximum_surface_p99_diagonal:
        failures["surfaceP99Diagonal"] = {
            "threshold": maximum_surface_p99_diagonal,
            "outputToSource": output_to_source_p99,
            "sourceToOutput": source_to_output_p99,
        }
    if max(output_to_source_maximum, source_to_output_maximum) > maximum_surface_distance_diagonal:
        failures["surfaceMaximumDiagonal"] = {
            "threshold": maximum_surface_distance_diagonal,
            "outputToSource": output_to_source_maximum,
            "sourceToOutput": source_to_output_maximum,
        }
    if failures:
        fail(
            "template-topology-stabilization-failed",
            "眼睑布尔微小三角面无法在无自交和低漂移约束下稳定化",
            stage="template-head-anatomy",
            details={
                "failures": failures,
                "areaBefore": area_before,
                "areaAfter": area_after,
                "float32Area": delivery_area,
                "sourceSelfIntersectionPairCount": source_intersections,
            },
        )

    return candidate, TopologyStabilization(
        method="open3d-qem-sliver-collapse",
        source_vertex_count=len(source_vertices),
        source_face_count=len(source_faces),
        requested_face_count=target_face_count,
        output_vertex_count=len(vertices),
        output_face_count=len(faces),
        minimum_relative_area_before=float(area_before["minimumRelativeArea"]),
        minimum_relative_area_after=float(area_after["minimumRelativeArea"]),
        float32_minimum_relative_area=float(delivery_area["minimumRelativeArea"]),
        self_intersection_pair_count_before=source_intersections,
        self_intersection_pair_count_after=intersections,
        float32_self_intersection_pair_count=delivery_intersections,
        bounds_maximum_drift=bounds_drift,
        output_to_source_p99_diagonal=output_to_source_p99,
        output_to_source_maximum_diagonal=output_to_source_maximum,
        source_to_output_p99_diagonal=source_to_output_p99,
        source_to_output_maximum_diagonal=source_to_output_maximum,
    )


def _cleanup_boolean_result(
    result: trimesh.Trimesh,
    *,
    maximum_artifact_faces: int = 128,
    maximum_faces_per_artifact: int = 16,
    maximum_discarded_area_ratio: float = 1e-5,
    maximum_discarded_volume_ratio: float = 1e-6,
) -> tuple[trimesh.Trimesh, BooleanCleanup]:
    components = result.split(only_watertight=False)
    if not components:
        fail(
            "template-eye-boolean-empty",
            "眼部布尔没有返回主头模",
            stage="template-head-anatomy",
        )
    ranked = sorted(
        components,
        key=lambda component: abs(float(component.volume)),
        reverse=True,
    )
    discarded_faces = sum(len(component.faces) for component in ranked[1:])
    maximum_component_faces = max(
        (len(component.faces) for component in ranked[1:]),
        default=0,
    )
    main_area = max(float(ranked[0].area), 1e-12)
    main_volume = max(abs(float(ranked[0].volume)), 1e-12)
    discarded_area_ratio = float(sum(float(component.area) for component in ranked[1:]) / main_area)
    discarded_volume_ratio = float(
        sum(abs(float(component.volume)) for component in ranked[1:]) / main_volume
    )
    if (
        discarded_faces > maximum_artifact_faces
        or maximum_component_faces > maximum_faces_per_artifact
        or discarded_area_ratio > maximum_discarded_area_ratio
        or discarded_volume_ratio > maximum_discarded_volume_ratio
    ):
        fail(
            "template-eye-boolean-artifacts",
            "眼部布尔产生了超出上限的非主组件",
            stage="template-head-anatomy",
            details={
                "discardedComponentCount": len(ranked) - 1,
                "discardedFaceCount": discarded_faces,
                "maximumArtifactFaces": maximum_artifact_faces,
                "maximumComponentFaceCount": maximum_component_faces,
                "maximumFacesPerArtifact": maximum_faces_per_artifact,
                "discardedAreaRatio": discarded_area_ratio,
                "maximumDiscardedAreaRatio": maximum_discarded_area_ratio,
                "discardedVolumeRatio": discarded_volume_ratio,
                "maximumDiscardedVolumeRatio": maximum_discarded_volume_ratio,
            },
        )
    main = ranked[0]
    before_faces = len(main.faces)
    cleaned = trimesh.Trimesh(
        vertices=np.asarray(main.vertices, dtype=np.float64),
        faces=np.asarray(main.faces, dtype=np.int64),
        process=True,
        validate=True,
    )
    degenerate_removed = before_faces - len(cleaned.faces)
    boundary_before, nonmanifold_before = _edge_counts(cleaned)
    if nonmanifold_before:
        fail(
            "template-eye-boolean-nonmanifold",
            "眼部布尔清理后仍有非流形边",
            stage="template-head-anatomy",
            details={"nonManifoldEdgeCount": nonmanifold_before},
        )
    if boundary_before and (
        boundary_before > maximum_artifact_faces or not trimesh.repair.fill_holes(cleaned)
    ):
        fail(
            "template-eye-boolean-hole",
            "眼部布尔的数值微孔无法安全封闭",
            stage="template-head-anatomy",
            details={"boundaryEdgeCount": boundary_before},
        )
    boundary_after, nonmanifold_after = _edge_counts(cleaned)
    if (
        not cleaned.is_volume
        or not cleaned.is_watertight
        or not cleaned.is_winding_consistent
        or boundary_after
        or nonmanifold_after
    ):
        fail(
            "template-eye-boolean-topology-invalid",
            "眼部布尔后主头模不是闭合单连通正体积网格",
            stage="template-head-anatomy",
            details={
                "watertight": bool(cleaned.is_watertight),
                "windingConsistent": bool(cleaned.is_winding_consistent),
                "boundaryEdgeCount": boundary_after,
                "nonManifoldEdgeCount": nonmanifold_after,
            },
        )
    return cleaned, BooleanCleanup(
        raw_component_count=len(ranked),
        discarded_component_count=len(ranked) - 1,
        discarded_face_count=discarded_faces,
        maximum_artifact_face_count=maximum_component_faces,
        discarded_area_ratio=discarded_area_ratio,
        discarded_volume_ratio=discarded_volume_ratio,
        degenerate_face_count_removed=degenerate_removed,
        boundary_edge_count_filled=boundary_before,
        face_count_after=len(cleaned.faces),
    )


def blender_hole_tolerant_difference(
    source: trimesh.Trimesh,
    cutter: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, BooleanCleanup]:
    if not source.is_volume or not cutter.is_volume:
        fail(
            "template-eye-boolean-input-invalid",
            "眼部布尔输入必须是正体积闭合网格",
            stage="template-head-anatomy",
        )
    template = get_string("templates/blender_boolean.py.tmpl")
    script = (
        template.replace("$OPERATION", "DIFFERENCE")
        .replace("$SOLVER_OPTIONS", "EXACT")
        .replace("$USE_SELF", "True")
        .replace(
            "mod.use_self = True",
            "mod.use_self = True\n        mod.use_hole_tolerant = True",
        )
    )
    try:
        with MeshScript(meshes=[source, cutter], script=script) as blend:
            loaded = blend.run(_blender_executable + " --background --python $SCRIPT")
        result = trimesh.util.concatenate(trimesh.util.make_sequence(loaded))
    except (OSError, RuntimeError, ValueError) as exc:
        fail(
            "template-eye-boolean-failed",
            "Blender 精确眼部布尔失败",
            stage="template-head-anatomy",
            details={"reason": str(exc)},
        )
    return _cleanup_boolean_result(result)


def _surface_distance(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    query = o3d.core.Tensor(
        np.asarray(points, dtype=np.float32),
        dtype=o3d.core.Dtype.Float32,
    )
    closest = _o3d_scene(mesh).compute_closest_points(query)["points"].numpy()
    return np.linalg.norm(closest - np.asarray(points, dtype=np.float32), axis=1)


def _ordered_simple_cycle(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    adjacency: dict[int, list[int]] = defaultdict(list)
    for first, second in edges:
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))
    if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("cycle edges must give every vertex degree two")
    start = min(adjacency)
    ordered = [start]
    previous = -1
    current = start
    while True:
        choices = sorted(adjacency[current])
        following = choices[0] if choices[0] != previous else choices[1]
        if following == start:
            break
        if following in ordered:
            raise ValueError("cycle self-intersects or contains a secondary loop")
        ordered.append(following)
        previous, current = current, following
    if len(ordered) != len(adjacency) or len(edges) != len(adjacency):
        raise ValueError("cycle does not cover its full edge component")
    return np.asarray(ordered, dtype=np.int64)


def _interface_cycles(
    original: trimesh.Trimesh,
    result: trimesh.Trimesh,
    cutter: trimesh.Trimesh,
    spec: EyeSurgerySpec,
) -> list[np.ndarray]:
    centroids = np.asarray(result.triangles_center, dtype=np.float32)
    source_distance = _surface_distance(original, centroids)
    cutter_distance = _surface_distance(cutter, centroids)
    diagonal = max(
        float(np.linalg.norm(np.ptp(np.asarray(original.vertices), axis=0))),
        1e-12,
    )
    cut_faces = (cutter_distance <= diagonal * 1e-5) & (source_distance > diagonal * 1.25e-7)
    edges: list[np.ndarray] = []
    for pair, edge in zip(
        result.face_adjacency,
        result.face_adjacency_edges,
        strict=True,
    ):
        if bool(cut_faces[pair[0]]) != bool(cut_faces[pair[1]]):
            edges.append(np.asarray(edge, dtype=np.int64))
    if not edges:
        return []
    edge_array = np.asarray(edges, dtype=np.int64)
    vertices = np.unique(edge_array)
    local = {int(value): index for index, value in enumerate(vertices)}
    row = np.asarray([local[int(value)] for value in edge_array[:, 0]])
    column = np.asarray([local[int(value)] for value in edge_array[:, 1]])
    graph = coo_matrix(
        (
            np.ones(len(edge_array) * 2, dtype=np.uint8),
            (np.concatenate((row, column)), np.concatenate((column, row))),
        ),
        shape=(len(vertices), len(vertices)),
    )
    count, labels = connected_components(graph, directed=False)
    result_vertices = np.asarray(result.vertices, dtype=np.float64)
    cycles: list[np.ndarray] = []
    for component in range(count):
        component_vertices = vertices[labels == component]
        component_set = set(int(value) for value in component_vertices)
        component_edges = np.asarray(
            [
                edge
                for edge in edge_array
                if int(edge[0]) in component_set and int(edge[1]) in component_set
            ],
            dtype=np.int64,
        )
        try:
            cycle = _ordered_simple_cycle(component_edges)
        except ValueError:
            continue
        points = result_vertices[cycle]
        if abs(float(points[:, 0].mean()) - float(spec.center[0])) <= spec.landmark_width and float(
            points[:, 2].mean()
        ) >= float(spec.center[2] + spec.eyeball_radius * 0.45):
            cycles.append(cycle)
    cycles.sort(
        key=lambda cycle: float(result_vertices[cycle, 2].mean()),
        reverse=True,
    )
    return cycles


def _remap_topological_cycle(
    source: trimesh.Trimesh,
    source_cycle: np.ndarray,
    result: trimesh.Trimesh,
    *,
    minimum_survival_fraction: float = 0.15,
    maximum_surface_distance_diagonal: float = 1e-3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Transfer an ordered boolean interface cycle across QEM edge collapses."""

    source_cycle = np.asarray(source_cycle, dtype=np.int64)
    source_points = np.asarray(source.vertices, dtype=np.float64)[source_cycle]
    result_vertices = np.asarray(result.vertices, dtype=np.float64)
    result_faces = np.asarray(result.faces, dtype=np.int64)
    diagonal = max(
        float(np.linalg.norm(np.ptp(np.asarray(source.vertices, dtype=np.float64), axis=0))),
        1e-12,
    )

    def point_to_closed_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
        segment_start = polyline
        segment_delta = np.roll(polyline, -1, axis=0) - segment_start
        denominator = np.maximum(
            np.einsum("ij,ij->i", segment_delta, segment_delta),
            np.finfo(np.float64).tiny,
        )
        output = np.empty(len(points), dtype=np.float64)
        for start in range(0, len(points), 2048):
            stop = min(start + 2048, len(points))
            offset = points[start:stop, None, :] - segment_start[None, :, :]
            parameter = np.clip(
                np.einsum("nsi,si->ns", offset, segment_delta) / denominator[None, :],
                0.0,
                1.0,
            )
            closest = segment_start[None, :, :] + parameter[:, :, None] * segment_delta[None, :, :]
            output[start:stop] = np.min(
                np.linalg.norm(points[start:stop, None, :] - closest, axis=2),
                axis=1,
            )
        return output

    distance, nearest = cKDTree(result_vertices).query(source_points, k=1)
    # Open3D QEM keeps untouched vertices bit-identical. Use only those
    # survivors as ordered anchors so collapsed vertices cannot map back and
    # forth across the loop.
    exact = distance <= diagonal * 1e-9
    anchors: list[int] = []
    for vertex in nearest[exact]:
        vertex = int(vertex)
        if not anchors or vertex != anchors[-1]:
            anchors.append(vertex)
    if len(anchors) > 1 and anchors[0] == anchors[-1]:
        anchors.pop()
    minimum_anchors = max(16, int(np.ceil(len(source_cycle) * minimum_survival_fraction)))
    if len(anchors) < minimum_anchors or len(set(anchors)) != len(anchors):
        fail(
            "template-eyelid-ring-remap-insufficient",
            "拓扑稳定化后没有保留足够且有序的眼睑环锚点",
            stage="template-head-anatomy",
            details={
                "sourceVertexCount": len(source_cycle),
                "survivingAnchorCount": len(anchors),
                "uniqueAnchorCount": len(set(anchors)),
                "minimumAnchorCount": minimum_anchors,
            },
        )

    edges = np.unique(
        np.sort(
            np.concatenate(
                (
                    result_faces[:, (0, 1)],
                    result_faces[:, (1, 2)],
                    result_faces[:, (2, 0)],
                ),
                axis=0,
            ),
            axis=1,
        ),
        axis=0,
    )
    edge_length = np.linalg.norm(
        result_vertices[edges[:, 0]] - result_vertices[edges[:, 1]],
        axis=1,
    )
    proximity = point_to_closed_polyline(result_vertices, source_points)
    proximity_scale = diagonal * maximum_surface_distance_diagonal
    edge_proximity = (proximity[edges[:, 0]] + proximity[edges[:, 1]]) * 0.5
    weights = edge_length * (1.0 + 25.0 * np.square(edge_proximity / max(proximity_scale, 1e-12)))
    graph = coo_matrix(
        (
            np.concatenate((weights, weights)),
            (
                np.concatenate((edges[:, 0], edges[:, 1])),
                np.concatenate((edges[:, 1], edges[:, 0])),
            ),
        ),
        shape=(len(result_vertices), len(result_vertices)),
    ).tocsr()
    _, predecessors = dijkstra(
        graph,
        directed=False,
        indices=np.asarray(anchors, dtype=np.int64),
        return_predecessors=True,
    )
    cycle_edges: set[tuple[int, int]] = set()
    maximum_path_vertices = 0
    for row, source_vertex in enumerate(anchors):
        target_vertex = anchors[(row + 1) % len(anchors)]
        current = target_vertex
        path = [current]
        while current != source_vertex:
            current = int(predecessors[row, current])
            if current < 0:
                fail(
                    "template-eyelid-ring-remap-disconnected",
                    "拓扑稳定化后眼睑环锚点不连通",
                    stage="template-head-anatomy",
                )
            path.append(current)
        path.reverse()
        maximum_path_vertices = max(maximum_path_vertices, len(path))
        cycle_edges.update(
            tuple(sorted((first, second)))
            for first, second in zip(path[:-1], path[1:], strict=True)
        )
    try:
        cycle = _ordered_simple_cycle(np.asarray(sorted(cycle_edges), dtype=np.int64))
    except ValueError as exc:
        fail(
            "template-eyelid-ring-remap-invalid",
            "拓扑稳定化后的眼睑环路径不是单一闭合环",
            stage="template-head-anatomy",
            details={"reason": str(exc), "edgeCount": len(cycle_edges)},
        )

    result_points = result_vertices[cycle]
    source_to_result = point_to_closed_polyline(source_points, result_points) / diagonal
    result_to_source = point_to_closed_polyline(result_points, source_points) / diagonal
    maximum_distance = float(
        max(
            np.max(source_to_result, initial=0.0),
            np.max(result_to_source, initial=0.0),
        )
    )
    if maximum_distance > maximum_surface_distance_diagonal:
        fail(
            "template-eyelid-ring-remap-drift",
            "拓扑稳定化后的眼睑环偏离原始布尔交界",
            stage="template-head-anatomy",
            details={
                "maximumDistanceDiagonal": maximum_distance,
                "threshold": maximum_surface_distance_diagonal,
                "sourceToOutputMaximumDiagonal": float(np.max(source_to_result, initial=0.0)),
                "outputToSourceMaximumDiagonal": float(np.max(result_to_source, initial=0.0)),
            },
        )
    return cycle, {
        "method": "ordered-survivor-anchors-and-shortest-edge-paths",
        "sourceVertexCount": len(source_cycle),
        "survivingAnchorCount": len(anchors),
        "outputVertexCount": len(cycle),
        "maximumPathVertexCount": maximum_path_vertices,
        "sourceToOutputP99Diagonal": float(np.quantile(source_to_result, 0.99)),
        "sourceToOutputMaximumDiagonal": float(np.max(source_to_result, initial=0.0)),
        "outputToSourceP99Diagonal": float(np.quantile(result_to_source, 0.99)),
        "outputToSourceMaximumDiagonal": float(np.max(result_to_source, initial=0.0)),
    }


def fitting_eyelid_contour(
    vertices: np.ndarray,
    surface_map: LandmarkSurfaceMap,
    side: str,
    *,
    samples_per_segment: int = 16,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bind the full MediaPipe eyelid contour to stable skin vertices.

    The Boolean interface is a geometry QA record, but it is not a reliable
    fitting control loop after QEM: a valid eye opening can retain many more
    interface vertices on one side than the other. This contour follows the
    complete observed eyelid polyline and is the loop consumed by landmark and
    eyelid losses. It never creates or moves geometry.
    """

    if side not in EYELID_CONTOUR_LANDMARKS:
        raise ValueError(f"unknown anatomical eye side: {side}")
    vertices = np.asarray(vertices, dtype=np.float64)
    indices = np.asarray(EYELID_CONTOUR_LANDMARKS[side], dtype=np.int64)
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least two")
    if not np.all(surface_map.valid[indices]):
        fail(
            "template-eyelid-contour-landmark-missing",
            f"{side} 眼睑闭合轮廓存在无效地标",
            stage="template-head-anatomy",
        )
    source_points = np.asarray(surface_map.points[indices], dtype=np.float64)
    samples = np.concatenate(
        [
            first
            + (second - first) * np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[:, None]
            for first, second in zip(
                source_points,
                np.roll(source_points, -1, axis=0),
                strict=True,
            )
        ],
        axis=0,
    )
    distance, nearest = cKDTree(vertices).query(samples, k=1)
    ordered: list[int] = []
    seen: set[int] = set()
    for value in nearest:
        vertex = int(value)
        if vertex not in seen:
            seen.add(vertex)
            ordered.append(vertex)
    ring = np.asarray(ordered, dtype=np.int64)
    if len(ring) < len(indices):
        fail(
            "template-eyelid-fitting-contour-incomplete",
            f"{side} 眼睑拟合轮廓没有覆盖完整闭合眼裂",
            stage="template-head-anatomy",
            details={"vertexCount": len(ring), "minimum": len(indices)},
        )

    ring_points = vertices[ring]
    source_span = np.ptp(source_points, axis=0)
    ring_span = np.ptp(ring_points, axis=0)
    landmark_width = max(float(source_span[0]), 1e-12)
    horizontal_coverage = float(ring_span[0] / landmark_width)
    vertical_coverage = float(ring_span[1] / max(float(source_span[1]), 1e-12))
    maximum_distance_ratio = float(np.max(distance, initial=0.0) / landmark_width)
    passed = bool(
        horizontal_coverage >= 0.90 and vertical_coverage >= 0.80 and maximum_distance_ratio <= 0.15
    )
    metrics: dict[str, Any] = {
        "method": "dense-mediapipe-closed-contour-nearest-skin",
        "closed": True,
        "sourceLandmarkCount": len(indices),
        "sampleCount": len(samples),
        "vertexCount": len(ring),
        "sourceSpan": source_span.astype(float).tolist(),
        "boundSpan": ring_span.astype(float).tolist(),
        "horizontalCoverageRatio": horizontal_coverage,
        "verticalCoverageRatio": vertical_coverage,
        "surfaceDistanceP99FaceWidth": float(np.quantile(distance, 0.99) / landmark_width),
        "surfaceDistanceMaximumFaceWidth": maximum_distance_ratio,
        "passed": passed,
    }
    if not passed:
        fail(
            "template-eyelid-fitting-contour-gate-failed",
            f"{side} 眼睑拟合轮廓覆盖不足",
            stage="template-head-anatomy",
            details=metrics,
        )
    return ring, metrics


def eyelid_contour_symmetry(
    metrics: dict[str, dict[str, Any]],
) -> dict[str, float | bool]:
    left = np.asarray(metrics["left"]["boundSpan"], dtype=np.float64)
    right = np.asarray(metrics["right"]["boundSpan"], dtype=np.float64)
    horizontal = float(min(left[0], right[0]) / max(left[0], right[0], 1e-12))
    vertical = float(min(left[1], right[1]) / max(left[1], right[1], 1e-12))
    passed = horizontal >= 0.90 and vertical >= 0.80
    result: dict[str, float | bool] = {
        "horizontalSpanRatio": horizontal,
        "verticalSpanRatio": vertical,
        "minimumHorizontalSpanRatio": 0.90,
        "minimumVerticalSpanRatio": 0.80,
        "passed": passed,
    }
    if not passed:
        fail(
            "template-eyelid-contour-asymmetry",
            "左右眼睑完整轮廓跨度不对称",
            stage="template-head-anatomy",
            details=result,
        )
    return result


def _contact_vertices(
    vertices: np.ndarray,
    spec: EyeSurgerySpec,
) -> tuple[np.ndarray, np.ndarray]:
    distance = np.linalg.norm(vertices - spec.center, axis=1)
    tolerance = max(spec.socket_radius * 5e-4, 1e-6)
    selected = np.flatnonzero(
        (np.abs(distance - spec.socket_radius) <= tolerance) & (vertices[:, 2] >= spec.center[2])
    )
    if len(selected) < 48:
        fail(
            "template-eyelid-contact-incomplete",
            f"{spec.anatomical_side} 眼球窝没有足够的眼睑接触顶点",
            stage="template-head-anatomy",
            details={"vertexCount": len(selected)},
        )
    gap = distance[selected] - spec.eyeball_radius
    return selected, gap


def _safe_hit_center(
    surface_map: LandmarkSurfaceMap,
    indices: tuple[int, ...],
    fallback: np.ndarray,
) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    valid = surface_map.valid[selected]
    return (
        surface_map.points[selected[valid]].mean(axis=0)
        if np.any(valid)
        else np.asarray(fallback, dtype=np.float64)
    )


def semantic_regions(
    vertices: np.ndarray,
    surface_map: LandmarkSurfaceMap,
    specs: dict[str, EyeSurgerySpec],
    rings: dict[str, np.ndarray],
    contacts: dict[str, np.ndarray],
    contact_rings: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    right_eye = specs["right"]
    left_eye = specs["left"]
    eye_y = float((right_eye.center[1] + left_eye.center[1]) * 0.5)
    chin = _safe_hit_center(surface_map, (152,), np.asarray([0.0, -0.4, 2.0]))
    face_top = _safe_hit_center(surface_map, (10,), np.asarray([0.0, 2.55, 2.0]))
    negative_cheek = _safe_hit_center(surface_map, (234,), np.asarray([-1.5, 1.35, 0.5]))
    positive_cheek = _safe_hit_center(surface_map, (454,), np.asarray([1.3, 1.35, 0.7]))
    mouth_center = _safe_hit_center(
        surface_map, (61, 291, 13, 14, 0, 17), np.asarray([0.0, 0.4, 2.1])
    )
    nose_center = _safe_hit_center(
        surface_map, (1, 2, 98, 327, 168, 6), np.asarray([0.0, 1.2, 2.2])
    )

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    within_face_width = (x >= negative_cheek[0] - 0.18) & (x <= positive_cheek[0] + 0.18)
    face = np.flatnonzero(
        within_face_width & (y >= chin[1] - 0.10) & (y <= face_top[1] + 0.20) & (z >= 0.25)
    )
    left_ear = np.flatnonzero(
        (x >= positive_cheek[0] - 0.16)
        & (y >= eye_y - 1.05)
        & (y <= eye_y + 0.92)
        & (z >= -0.30)
        & (z <= 1.55)
    )
    right_ear = np.flatnonzero(
        (x <= negative_cheek[0] + 0.16)
        & (y >= eye_y - 1.05)
        & (y <= eye_y + 0.92)
        & (z >= -0.30)
        & (z <= 1.55)
    )
    neck_shoulders = np.flatnonzero(y <= chin[1] - 0.05)
    cranium = np.flatnonzero(
        (y >= eye_y + 0.12) & (y <= vertices[:, 1].max()) & (np.abs(x) <= 1.85)
    )
    rear_cranium = np.flatnonzero((z <= 0.45) & (y >= chin[1]) & (np.abs(x) <= 1.85))
    jaw = np.flatnonzero(
        within_face_width & (y >= chin[1] - 0.15) & (y <= mouth_center[1] + 0.28) & (z >= 0.65)
    )
    nose = np.flatnonzero(
        (np.abs(x - nose_center[0]) <= 0.48) & (np.abs(y - nose_center[1]) <= 0.62) & (z >= 1.75)
    )
    mouth = np.flatnonzero(
        (np.abs(x - mouth_center[0]) <= 0.62) & (np.abs(y - mouth_center[1]) <= 0.30) & (z >= 1.65)
    )

    regions: dict[str, np.ndarray] = {
        "face": face,
        "cranium": cranium,
        "rear_cranium": rear_cranium,
        "neck_shoulders": neck_shoulders,
        "left_ear": left_ear,
        "right_ear": right_ear,
        "nose": nose,
        "mouth": mouth,
        "jaw": jaw,
    }
    for side, spec in specs.items():
        contact_ring = rings[side] if contact_rings is None else contact_rings[side]
        local_distance = np.linalg.norm(vertices - spec.center, axis=1)
        regions[f"{side}_eye"] = np.unique(
            np.concatenate(
                (
                    np.flatnonzero(local_distance <= spec.eyeball_radius * 1.85),
                    rings[side],
                    contact_ring,
                    contacts[side],
                )
            )
        )
        regions[f"{side}_eyelid_ring"] = np.asarray(rings[side], dtype=np.int64)
        regions[f"{side}_eyelid"] = np.asarray(contacts[side], dtype=np.int64)
    regions = {
        name: np.unique(np.asarray(indices, dtype=np.int64)) for name, indices in regions.items()
    }
    missing = [name for name in REQUIRED_SEMANTIC_REGIONS if not len(regions[name])]
    if missing:
        fail(
            "template-semantic-region-empty",
            "TemplateHeadV0 必需语义区域为空",
            stage="template-head-anatomy",
            details={"missing": missing},
        )
    if np.intersect1d(regions["left_ear"], regions["right_ear"]).size:
        fail(
            "template-semantic-ear-overlap",
            "TemplateHeadV0 左右耳语义区域重叠",
            stage="template-head-anatomy",
        )
    return regions


def _top_curvature_spike_ratio(mesh: trimesh.Trimesh) -> float:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    threshold = float(vertices[:, 1].max() - np.ptp(vertices[:, 1]) * 0.18)
    top = np.flatnonzero(vertices[:, 1] >= threshold)
    values: list[float] = []
    for index in top:
        neighbors = np.asarray(mesh.vertex_neighbors[index], dtype=np.int64)
        if not len(neighbors):
            continue
        edge_scale = float(np.median(np.linalg.norm(vertices[neighbors] - vertices[index], axis=1)))
        if edge_scale <= 1e-12:
            continue
        laplacian = float(np.linalg.norm(vertices[index] - vertices[neighbors].mean(axis=0)))
        values.append(laplacian / edge_scale)
    if not values:
        return 0.0
    measured = np.asarray(values, dtype=np.float64)
    return float(np.quantile(measured, 0.99) / max(np.median(measured), 1e-12))


def _eye_intersection_metrics(
    head: trimesh.Trimesh,
    spec: EyeSurgerySpec,
) -> dict[str, Any]:
    sphere = trimesh.creation.icosphere(
        subdivisions=4,
        radius=spec.eyeball_radius,
    )
    points = np.asarray(sphere.vertices, dtype=np.float64) + spec.center
    signed = (
        _o3d_scene(head)
        .compute_signed_distance(
            o3d.core.Tensor(points.astype(np.float32), dtype=o3d.core.Dtype.Float32)
        )
        .numpy()
    )
    diagonal = max(
        float(np.linalg.norm(np.ptp(np.asarray(head.vertices), axis=0))),
        1e-12,
    )
    intersection_count = int(np.count_nonzero(signed < -diagonal * 1e-7))
    return {
        "intersectionCount": intersection_count,
        "minimumClearance": float(np.min(signed)),
        "minimumClearanceRatio": float(np.min(signed) / spec.eyeball_radius),
        "sampleCount": len(points),
    }


def _colored_combined_mesh(asset: UnifiedHeadAsset) -> trimesh.Trimesh:
    head = asset.skin_mesh.copy()
    head.visual = trimesh.visual.ColorVisuals(
        mesh=head,
        face_colors=np.tile(
            np.asarray([170, 181, 188, 255], dtype=np.uint8),
            (len(head.faces), 1),
        ),
    )
    eyes: list[trimesh.Trimesh] = []
    for eye in (asset.left_eye, asset.right_eye):
        sphere = eye.mesh()
        sphere.visual = trimesh.visual.ColorVisuals(
            mesh=sphere,
            face_colors=np.tile(
                np.asarray([235, 232, 220, 255], dtype=np.uint8),
                (len(sphere.faces), 1),
            ),
        )
        eyes.append(sphere)
    return trimesh.util.concatenate([head, *eyes])


def _semantic_mesh(asset: UnifiedHeadAsset) -> trimesh.Trimesh:
    mesh = asset.skin_mesh.copy()
    colors = np.tile(
        np.asarray([112, 118, 126, 255], dtype=np.uint8),
        (len(mesh.vertices), 1),
    )
    palette = {
        "face": (76, 142, 255, 255),
        "cranium": (146, 105, 245, 255),
        "left_ear": (255, 147, 79, 255),
        "right_ear": (255, 147, 79, 255),
        "left_eye": (111, 221, 166, 255),
        "right_eye": (111, 221, 166, 255),
        "nose": (255, 206, 84, 255),
        "mouth": (245, 108, 146, 255),
        "jaw": (104, 205, 224, 255),
    }
    for name, color in palette.items():
        colors[asset.regions[name]] = np.asarray(color, dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
    return mesh


def _detect_landmarks(front: Path, model: Path) -> np.ndarray:
    try:
        rgb = np.asarray(Image.open(front).convert("RGB"))
    except OSError as exc:
        fail(
            "template-front-reference-invalid",
            "无法读取 TemplateHeadV0 正面 QA 图",
            stage="template-head-anatomy",
            details={"reason": str(exc)},
        )
    with face_landmarker(model) as detector:
        landmarks, _, _ = _detect(detector, rgb)
    return np.asarray(landmarks, dtype=np.float64)


def _artifact_hashes(root: Path, artifacts: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, relative in artifacts.items():
        if isinstance(relative, str):
            hashes[name] = sha256_file(root / relative)
        elif isinstance(relative, dict):
            for child_name, child_relative in relative.items():
                hashes[f"{name}.{child_name}"] = sha256_file(root / child_relative)
    return hashes


def prepare_template_head_v0_anatomy(
    template_root: Path,
    face_landmarker_model: Path | None,
    *,
    landmark_override: np.ndarray | None = None,
    boolean_difference: BooleanDifference | None = None,
) -> dict[str, Any]:
    template_root = template_root.expanduser().resolve()
    root_manifest_path = template_root / "manifest.json"
    if not root_manifest_path.is_file():
        fail(
            "template-manifest-missing",
            "TemplateHeadV0 根 manifest 不存在",
            stage="template-head-anatomy",
            details={"path": str(root_manifest_path)},
        )
    root_manifest = json.loads(root_manifest_path.read_text())
    if root_manifest.get("templateId") != "TemplateHeadV0":
        fail(
            "template-id-mismatch",
            "输入目录不是 TemplateHeadV0",
            stage="template-head-anatomy",
        )
    output = template_root / ANATOMY_DIRECTORY
    if output.exists():
        fail(
            "template-anatomy-output-exists",
            "TemplateHeadV0 anatomy 已存在；为避免覆盖，已停止",
            stage="template-head-anatomy",
            details={"path": str(output)},
        )

    artifacts = root_manifest.get("artifacts", {})
    raw_relative = artifacts.get("rawTemplate")
    front_relative = artifacts.get("fixedViews", {}).get("front")
    if not isinstance(raw_relative, str) or not isinstance(front_relative, str):
        fail(
            "template-artifact-missing",
            "TemplateHeadV0 缺少原始模板或正面 QA 图",
            stage="template-head-anatomy",
        )
    raw_path = template_root / raw_relative
    front_path = template_root / front_relative
    raw = RawTemplateHeadV0.load(raw_path)
    original = raw.compute_mesh
    front_camera = CameraRecord.model_validate(root_manifest["cameras"]["front"])
    if landmark_override is None:
        if face_landmarker_model is None:
            fail(
                "face-landmarker-missing",
                "生成 TemplateHeadV0 anatomy 需要本地 Face Landmarker",
                stage="template-head-anatomy",
            )
        face_landmarker_model = face_landmarker_model.expanduser().resolve()
        if not face_landmarker_model.is_file():
            fail(
                "face-landmarker-missing",
                "本地 Face Landmarker 模型不存在",
                stage="template-head-anatomy",
                details={"path": str(face_landmarker_model)},
            )
        normalized = _detect_landmarks(front_path, face_landmarker_model)
    else:
        normalized = np.asarray(landmark_override, dtype=np.float64)
    surface_map = map_landmarks_to_template(original, front_camera, normalized)
    specs = derive_eye_specs(surface_map)

    difference = boolean_difference or blender_hole_tolerant_difference
    current = original.copy()
    cutters: dict[str, trimesh.Trimesh] = {}
    cleanup: dict[str, BooleanCleanup] = {}
    # Modify anatomical right first to keep operation order deterministic.
    for side in ("right", "left"):
        cutter = _eye_cutter(specs[side])
        cutters[side] = cutter
        current, cleanup[side] = difference(current, cutter)
    pre_stabilization_rings: dict[str, np.ndarray] = {}
    for side in ("right", "left"):
        cycles = _interface_cycles(original, current, cutters[side], specs[side])
        if not cycles:
            fail(
                "template-eyelid-ring-missing",
                f"{side} 眼裂没有形成可验证的闭合边界环",
                stage="template-head-anatomy",
            )
        pre_stabilization_rings[side] = cycles[0]
    final_mesh, stabilization = _stabilize_boolean_topology(
        current,
        target_face_count=STABILIZED_FACE_COUNT,
    )
    topology = _edge_and_component_metrics(
        np.asarray(final_mesh.vertices, dtype=np.float64),
        np.asarray(final_mesh.faces, dtype=np.int64),
    )
    if (
        topology["componentCount"] != 1
        or topology["boundaryEdgeCount"] != 0
        or topology["nonManifoldEdgeCount"] != 0
        or topology["degenerateFaceCount"] != 0
        or not final_mesh.is_watertight
        or not final_mesh.is_winding_consistent
    ):
        fail(
            "template-anatomy-topology-invalid",
            "TemplateHeadV0 anatomy 不是单连通闭合流形",
            stage="template-head-anatomy",
            details=topology,
        )
    if not np.allclose(final_mesh.bounds, original.bounds, rtol=0.0, atol=1e-6):
        fail(
            "template-anatomy-bounds-drift",
            "眼部手术意外改变了头颈外部边界",
            stage="template-head-anatomy",
            details={
                "originalBounds": np.asarray(original.bounds).tolist(),
                "finalBounds": np.asarray(final_mesh.bounds).tolist(),
            },
        )

    interface_rings: dict[str, np.ndarray] = {}
    contacts: dict[str, np.ndarray] = {}
    ring_remap_metrics: dict[str, dict[str, Any]] = {}
    eye_metrics: dict[str, Any] = {}
    vertices = np.asarray(final_mesh.vertices, dtype=np.float64)
    for side in ("right", "left"):
        interface_rings[side], ring_remap_metrics[side] = _remap_topological_cycle(
            current,
            pre_stabilization_rings[side],
            final_mesh,
        )
        contacts[side], gap = _contact_vertices(vertices, specs[side])
        intersection = _eye_intersection_metrics(final_mesh, specs[side])
        p99_gap_ratio = float(np.quantile(gap / specs[side].eyeball_radius, 0.99))
        if p99_gap_ratio > 0.03 + 1e-5 or intersection["intersectionCount"]:
            fail(
                "template-eye-contact-gate-failed",
                f"{side} 眼球接触或穿插门禁失败",
                stage="template-head-anatomy",
                details={
                    "contactGapP99R": p99_gap_ratio,
                    **intersection,
                },
            )
        ring_points = vertices[interface_rings[side]]
        eye_metrics[side] = {
            **specs[side].as_json(),
            "outerRingVertexCount": len(interface_rings[side]),
            "outerRingClosed": True,
            "outerRingBounds": np.stack((ring_points.min(axis=0), ring_points.max(axis=0)))
            .astype(float)
            .tolist(),
            "interfaceRingVertexCount": len(interface_rings[side]),
            "interfaceRingClosed": True,
            "interfaceRingBounds": np.stack((ring_points.min(axis=0), ring_points.max(axis=0)))
            .astype(float)
            .tolist(),
            "contactVertexCount": len(contacts[side]),
            "contactGapP99R": p99_gap_ratio,
            **intersection,
            "booleanCleanup": cleanup[side].as_json(),
            "ringRemap": ring_remap_metrics[side],
        }

    source_surface_map = surface_map
    fitting_rings: dict[str, np.ndarray] = {}
    fitting_contour_metrics: dict[str, dict[str, Any]] = {}
    for side in ("right", "left"):
        fitting_rings[side], fitting_contour_metrics[side] = fitting_eyelid_contour(
            vertices,
            source_surface_map,
            side,
        )
        eye_metrics[side]["fittingContour"] = fitting_contour_metrics[side]
    contour_symmetry = eyelid_contour_symmetry(fitting_contour_metrics)
    surface_map, landmark_surface_offset = remap_landmarks_to_anatomy(
        source_surface_map,
        final_mesh,
        front_camera,
        fitting_rings,
    )
    prepared, _, uv_metrics, uv_method = _extract_raw_asset(final_mesh, None)
    regions = semantic_regions(
        prepared.compute_vertices,
        surface_map,
        specs,
        fitting_rings,
        contacts,
        interface_rings,
    )
    left_eye = EyeballAsset(
        center=specs["left"].center,
        radius=specs["left"].eyeball_radius,
        gaze=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    )
    right_eye = EyeballAsset(
        center=specs["right"].center,
        radius=specs["right"].eyeball_radius,
        gaze=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    )
    delivery_hash = geometry_hash(prepared.render_vertices, prepared.render_faces)
    top_spike_ratio = _top_curvature_spike_ratio(final_mesh)
    if top_spike_ratio > 4.0:
        fail(
            "template-cranium-spike-gate-failed",
            "TemplateHeadV0 头顶曲率尖峰超过质量上限",
            stage="template-head-anatomy",
            details={"topCurvatureSpikeRatio": top_spike_ratio, "maximum": 4.0},
        )
    anatomy = {
        "schemaVersion": ANATOMY_SCHEMA_VERSION,
        "templateId": "TemplateHeadV0",
        "state": "anatomy-ready",
        "lineage": {
            "rawGeometrySha256": raw.geometry_sha256,
            "surfaceSource": "template-local-topology-surgery",
            "topologyStabilization": "qem-sliver-collapse-only",
            "sdfUsed": False,
        },
        "geometry": {
            "computeGeometrySha256": prepared.geometry_sha256,
            "deliveryGeometrySha256": delivery_hash,
            "computeVertexCount": len(prepared.compute_vertices),
            "computeFaceCount": len(prepared.compute_faces),
            "renderVertexCount": len(prepared.render_to_compute),
            "renderFaceCount": len(prepared.render_faces),
            "topCurvatureSpikeRatio": top_spike_ratio,
            "topologyStabilization": stabilization.as_json(),
            **topology,
        },
        "eyes": {
            "completeEyeballNodes": 2,
            "nodes": ["Eyeball.L", "Eyeball.R"],
            "radiusDifferenceRatio": 0.0,
            "right": eye_metrics["right"],
            "left": eye_metrics["left"],
            "contourSymmetry": contour_symmetry,
            "intersectionCount": int(
                eye_metrics["right"]["intersectionCount"] + eye_metrics["left"]["intersectionCount"]
            ),
        },
        "ears": {
            "source": "continuous-licensed-template-topology",
            "carrierPresent": False,
            "rootSharedWithScalp": True,
            "leftVertexCount": len(regions["left_ear"]),
            "rightVertexCount": len(regions["right_ear"]),
        },
        "semanticRegions": {
            name: {"vertexCount": len(indices)} for name, indices in sorted(regions.items())
        },
        "uv": {**uv_metrics, "method": uv_method},
        "landmarks": {
            "count": len(surface_map.normalized),
            "hitCount": int(np.count_nonzero(surface_map.valid)),
            "skinLandmarkCount": int(np.count_nonzero(surface_map.valid[:468])),
            "requiredEyeHitCount": 8,
            "irisAssignment": "independent-eyeballs",
            "bindingGeometrySha256": prepared.geometry_sha256,
            "surfaceOffsetMedian": float(np.nanmedian(landmark_surface_offset)),
            "surfaceOffsetP99": float(np.nanquantile(landmark_surface_offset, 0.99)),
            "surfaceOffsetMaximum": float(np.nanmax(landmark_surface_offset)),
            "reprojectionMedianPx": float(np.nanmedian(surface_map.reprojection_error_px)),
            "reprojectionP99Px": float(np.nanquantile(surface_map.reprojection_error_px, 0.99)),
        },
        "route": {
            "finalSurface": "canonical-template",
            "topology": "template-head-v0-continuous-head-neck",
            "sdfRole": "qa-only",
            "cubeOrVoxelSurfaceGeneration": False,
        },
    }
    unified = UnifiedHeadAsset(
        skin_vertices=prepared.compute_vertices,
        skin_faces=prepared.compute_faces,
        render_to_skin=prepared.render_to_compute,
        render_faces=prepared.render_faces,
        uv=prepared.uv,
        regions=regions,
        left_eye=left_eye,
        right_eye=right_eye,
        geometry_sha256=delivery_hash,
        anatomy=anatomy,
    )

    template_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".template-head-anatomy.",
        dir=template_root,
    ) as temporary:
        staging = Path(temporary) / ANATOMY_DIRECTORY
        staging.mkdir()
        unified.save(staging / "template-head-v0.unified.npz")
        export_neutral_mesh(
            unified.skin_mesh,
            staging / "template-head-v0-anatomy.glb",
        )
        atlas = Image.new("RGB", (512, 512), (176, 165, 154))
        unified.export_head_glb(staging / "head.glb", atlas)
        np.savez_compressed(
            staging / "landmark-surface-map.npz",
            normalized=surface_map.normalized.astype(np.float32),
            points=surface_map.points.astype(np.float32),
            source_points=source_surface_map.points.astype(np.float32),
            valid=surface_map.valid.astype(np.uint8),
            triangle=surface_map.triangle.astype(np.uint32),
            barycentric=surface_map.barycentric.astype(np.float32),
            surface_offset=landmark_surface_offset.astype(np.float32),
            reprojection_error_px=surface_map.reprojection_error_px.astype(np.float32),
        )
        cameras = {
            name: CameraRecord.model_validate(payload)
            for name, payload in root_manifest["cameras"].items()
        }
        combined = _colored_combined_mesh(unified)
        for name, camera in cameras.items():
            render_flat_mesh(
                combined,
                camera,
                staging / "qa" / f"fixed-view-{name}.png",
                width=720,
                height=720,
                use_mesh_face_colors=True,
            )
        render_flat_mesh(
            _semantic_mesh(unified),
            cameras["front"],
            staging / "qa" / "semantic-front.png",
            width=720,
            height=720,
            use_mesh_face_colors=True,
        )
        anatomy_artifacts = {
            "unifiedTemplate": "template-head-v0.unified.npz",
            "neutralTemplate": "template-head-v0-anatomy.glb",
            "headGlb": "head.glb",
            "landmarkSurfaceMap": "landmark-surface-map.npz",
            "fixedViews": {name: f"qa/fixed-view-{name}.png" for name in cameras},
            "semanticFront": "qa/semantic-front.png",
        }
        anatomy_manifest = {
            **anatomy,
            "assets": {
                "faceLandmarker": (
                    {
                        "path": str(face_landmarker_model),
                        "sha256": sha256_file(face_landmarker_model),
                    }
                    if face_landmarker_model is not None
                    else {"path": None, "sha256": None, "testOverride": True}
                )
            },
            "artifacts": anatomy_artifacts,
        }
        anatomy_manifest["artifactSha256"] = _artifact_hashes(
            staging,
            anatomy_artifacts,
        )
        anatomy_artifacts["anatomyManifest"] = "anatomy.json"
        atomic_write_json(staging / "anatomy.json", anatomy_manifest)
        os.replace(staging, output)

    root_relative = output.relative_to(template_root)
    root_artifacts = {
        "unifiedTemplate": str(root_relative / "template-head-v0.unified.npz"),
        "anatomyNeutralTemplate": str(root_relative / "template-head-v0-anatomy.glb"),
        "headTemplate": str(root_relative / "head.glb"),
        "anatomyManifest": str(root_relative / "anatomy.json"),
    }
    root_manifest["state"] = "anatomy-ready"
    root_manifest.setdefault("readiness", {}).update(
        {
            "semanticRegionsReady": True,
            "openEyelidRingsReady": True,
            "completeEyeballsReady": True,
        }
    )
    root_manifest.setdefault("artifacts", {}).update(root_artifacts)
    root_manifest.setdefault("artifactSha256", {}).update(
        {name: sha256_file(template_root / relative) for name, relative in root_artifacts.items()}
    )
    root_manifest["anatomy"] = {
        "schemaVersion": ANATOMY_SCHEMA_VERSION,
        "computeGeometrySha256": prepared.geometry_sha256,
        "deliveryGeometrySha256": delivery_hash,
        "semanticRegionCount": len(regions),
        "completeEyeballNodes": 2,
        "topologyStabilization": stabilization.as_json(),
        "sdfUsed": False,
    }
    atomic_write_json(root_manifest_path, root_manifest)
    return {
        "ok": True,
        "templateId": "TemplateHeadV0",
        "state": "anatomy-ready",
        "output": str(output),
        "computeGeometrySha256": prepared.geometry_sha256,
        "deliveryGeometrySha256": delivery_hash,
        "computeVertexCount": len(prepared.compute_vertices),
        "computeFaceCount": len(prepared.compute_faces),
        "semanticRegionCount": len(regions),
        "completeEyeballNodes": 2,
        "eyeIntersectionCount": anatomy["eyes"]["intersectionCount"],
        "topologyStabilization": stabilization.as_json(),
        "sdfUsed": False,
    }


def rebind_template_head_v0_eyelids(template_root: Path) -> dict[str, Any]:
    """Replace uneven Boolean-interface fitting rings without moving geometry."""

    template_root = template_root.expanduser().resolve()
    root_manifest_path = template_root / "manifest.json"
    if not root_manifest_path.is_file():
        fail(
            "template-manifest-missing",
            "TemplateHeadV0 根 manifest 不存在",
            stage="template-head-anatomy",
            details={"path": str(root_manifest_path)},
        )
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    artifacts = root_manifest.get("artifacts", {})
    required_artifacts = {
        "unifiedTemplate": artifacts.get("unifiedTemplate"),
        "anatomyManifest": artifacts.get("anatomyManifest"),
    }
    if not all(isinstance(value, str) for value in required_artifacts.values()):
        fail(
            "template-anatomy-artifact-missing",
            "TemplateHeadV0 缺少眼睑重绑定所需 anatomy 资产",
            stage="template-head-anatomy",
            details={"artifacts": required_artifacts},
        )

    unified_path = template_root / str(required_artifacts["unifiedTemplate"])
    anatomy_manifest_path = template_root / str(required_artifacts["anatomyManifest"])
    if not unified_path.is_file() or not anatomy_manifest_path.is_file():
        fail(
            "template-anatomy-artifact-missing",
            "TemplateHeadV0 眼睑重绑定输入不存在",
            stage="template-head-anatomy",
        )
    anatomy_manifest = json.loads(anatomy_manifest_path.read_text(encoding="utf-8"))
    anatomy_artifacts = anatomy_manifest.get("artifacts", {})
    landmark_relative = anatomy_artifacts.get("landmarkSurfaceMap")
    semantic_relative = anatomy_artifacts.get("semanticFront")
    if not isinstance(landmark_relative, str) or not isinstance(semantic_relative, str):
        fail(
            "template-anatomy-artifact-missing",
            "TemplateHeadV0 缺少地标绑定或语义图",
            stage="template-head-anatomy",
        )
    landmark_path = anatomy_manifest_path.parent / landmark_relative
    semantic_path = anatomy_manifest_path.parent / semantic_relative
    if not landmark_path.is_file():
        fail(
            "template-landmark-map-missing",
            "TemplateHeadV0 缺少可迁移的地标绑定",
            stage="template-head-anatomy",
        )

    asset = UnifiedHeadAsset.load(unified_path)
    before_vertices = np.asarray(asset.skin_vertices, dtype=np.float64).copy()
    before_faces = np.asarray(asset.skin_faces, dtype=np.int64).copy()
    before_render_to_skin = np.asarray(asset.render_to_skin, dtype=np.int64).copy()
    before_render_faces = np.asarray(asset.render_faces, dtype=np.int64).copy()
    before_uv = np.asarray(asset.uv, dtype=np.float32).copy()
    before_compute_hash = geometry_hash(before_vertices, before_faces)
    before_delivery_hash = geometry_hash(asset.render_vertices, asset.render_faces)

    with np.load(landmark_path, allow_pickle=False) as payload:
        required_fields = {"normalized", "source_points", "valid"}
        missing_fields = sorted(required_fields - set(payload.files))
        if missing_fields:
            fail(
                "template-landmark-map-incomplete",
                "TemplateHeadV0 地标绑定缺少原始表面点",
                stage="template-head-anatomy",
                details={"missing": missing_fields},
            )
        normalized = np.asarray(payload["normalized"], dtype=np.float64)
        source_points = np.asarray(payload["source_points"], dtype=np.float64)
        source_valid = np.asarray(payload["valid"], dtype=bool)

    count = len(normalized)
    source_map = LandmarkSurfaceMap(
        normalized=normalized,
        points=source_points,
        valid=source_valid,
        triangle=np.zeros(count, dtype=np.uint32),
        barycentric=np.zeros((count, 3), dtype=np.float64),
        reprojection_error_px=np.full(count, np.nan, dtype=np.float64),
    )
    fitting_rings: dict[str, np.ndarray] = {}
    fitting_metrics: dict[str, dict[str, Any]] = {}
    for side in ("right", "left"):
        fitting_rings[side], fitting_metrics[side] = fitting_eyelid_contour(
            asset.skin_vertices,
            source_map,
            side,
        )
    contour_symmetry = eyelid_contour_symmetry(fitting_metrics)

    front_camera = CameraRecord.model_validate(root_manifest["cameras"]["front"])
    rebound, surface_offset = remap_landmarks_to_anatomy(
        source_map,
        asset.skin_mesh,
        front_camera,
        fitting_rings,
    )
    regions = {
        name: np.asarray(indices, dtype=np.int64).copy() for name, indices in asset.regions.items()
    }
    for side in ("right", "left"):
        regions[f"{side}_eyelid_ring"] = fitting_rings[side]
        regions[f"{side}_eye"] = np.unique(
            np.concatenate((regions[f"{side}_eye"], fitting_rings[side]))
        )
        eye = asset.left_eye if side == "left" else asset.right_eye
        eyelid = np.asarray(regions[f"{side}_eyelid"], dtype=np.int64)
        gap_ratio = (
            np.linalg.norm(asset.skin_vertices[eyelid] - eye.center, axis=1) - eye.radius
        ) / eye.radius
        contact = eyelid[(gap_ratio >= -1e-5) & (gap_ratio <= 0.031)]
        if len(contact) < 48:
            fail(
                "template-eyelid-contact-incomplete",
                f"{side} 眼睑重绑定后接触带不足",
                stage="template-head-anatomy",
                details={"contactVertexCount": len(contact)},
            )
        regions[f"{side}_eyelid"] = contact

    updated_anatomy = copy.deepcopy(asset.anatomy)
    updated_anatomy["schemaVersion"] = ANATOMY_SCHEMA_VERSION
    updated_anatomy.setdefault("lineage", {})["eyelidFittingBinding"] = (
        "dense-mediapipe-closed-contour-nearest-skin"
    )
    updated_anatomy.setdefault("eyes", {})["contourSymmetry"] = contour_symmetry
    for side in ("right", "left"):
        eye_anatomy = updated_anatomy["eyes"][side]
        eye_anatomy["interfaceRingVertexCount"] = eye_anatomy["outerRingVertexCount"]
        eye_anatomy["interfaceRingClosed"] = eye_anatomy["outerRingClosed"]
        eye_anatomy["interfaceRingBounds"] = eye_anatomy["outerRingBounds"]
        eye_anatomy["fittingContour"] = fitting_metrics[side]
    updated_anatomy["semanticRegions"] = {
        name: {"vertexCount": len(indices)} for name, indices in sorted(regions.items())
    }
    updated_anatomy["landmarks"].update(
        {
            "count": len(rebound.normalized),
            "hitCount": int(np.count_nonzero(rebound.valid)),
            "skinLandmarkCount": int(np.count_nonzero(rebound.valid[:468])),
            "bindingGeometrySha256": before_compute_hash,
            "surfaceOffsetMedian": float(np.nanmedian(surface_offset)),
            "surfaceOffsetP99": float(np.nanquantile(surface_offset, 0.99)),
            "surfaceOffsetMaximum": float(np.nanmax(surface_offset)),
            "reprojectionMedianPx": float(np.nanmedian(rebound.reprojection_error_px)),
            "reprojectionP99Px": float(np.nanquantile(rebound.reprojection_error_px, 0.99)),
        }
    )
    updated = UnifiedHeadAsset(
        skin_vertices=before_vertices,
        skin_faces=before_faces,
        render_to_skin=before_render_to_skin,
        render_faces=before_render_faces,
        uv=before_uv,
        regions=regions,
        left_eye=asset.left_eye,
        right_eye=asset.right_eye,
        geometry_sha256=asset.geometry_sha256,
        anatomy=updated_anatomy,
    )
    after_compute_hash = geometry_hash(updated.skin_vertices, updated.skin_faces)
    after_delivery_hash = geometry_hash(updated.render_vertices, updated.render_faces)
    geometry_changed = bool(
        before_compute_hash != after_compute_hash
        or before_delivery_hash != after_delivery_hash
        or not np.array_equal(before_vertices, updated.skin_vertices)
        or not np.array_equal(before_faces, updated.skin_faces)
        or not np.array_equal(before_render_to_skin, updated.render_to_skin)
        or not np.array_equal(before_render_faces, updated.render_faces)
    )
    uv_changed = not np.array_equal(before_uv, updated.uv)
    if geometry_changed or uv_changed:
        fail(
            "template-eyelid-rebind-mutated-asset",
            "眼睑重绑定意外修改了 TemplateHeadV0 几何或 UV",
            stage="template-head-anatomy",
            details={"geometryChanged": geometry_changed, "uvChanged": uv_changed},
        )

    with tempfile.TemporaryDirectory(
        prefix=".template-head-eyelid-rebind.",
        dir=template_root,
    ) as temporary:
        staging = Path(temporary)
        staged_unified = staging / "template-head-v0.unified.npz"
        staged_landmarks = staging / "landmark-surface-map.npz"
        staged_semantic = staging / "semantic-front.png"
        staged_manifest = staging / "anatomy.json"
        updated.save(staged_unified)
        np.savez_compressed(
            staged_landmarks,
            normalized=rebound.normalized.astype(np.float32),
            points=rebound.points.astype(np.float32),
            source_points=source_points.astype(np.float32),
            valid=rebound.valid.astype(np.uint8),
            triangle=rebound.triangle.astype(np.uint32),
            barycentric=rebound.barycentric.astype(np.float32),
            surface_offset=surface_offset.astype(np.float32),
            reprojection_error_px=rebound.reprojection_error_px.astype(np.float32),
        )
        render_flat_mesh(
            _semantic_mesh(updated),
            front_camera,
            staged_semantic,
            width=720,
            height=720,
            use_mesh_face_colors=True,
        )

        os.replace(staged_unified, unified_path)
        os.replace(staged_landmarks, landmark_path)
        semantic_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_semantic, semantic_path)

        updated_manifest = copy.deepcopy(anatomy_manifest)
        for key, value in updated_anatomy.items():
            updated_manifest[key] = value
        hashable_artifacts = {
            name: relative
            for name, relative in updated_manifest["artifacts"].items()
            if name != "anatomyManifest"
        }
        updated_manifest["artifactSha256"] = _artifact_hashes(
            anatomy_manifest_path.parent,
            hashable_artifacts,
        )
        atomic_write_json(staged_manifest, updated_manifest)
        os.replace(staged_manifest, anatomy_manifest_path)

    root_manifest["anatomy"] = {
        **root_manifest.get("anatomy", {}),
        "schemaVersion": ANATOMY_SCHEMA_VERSION,
        "computeGeometrySha256": after_compute_hash,
        "deliveryGeometrySha256": after_delivery_hash,
        "semanticRegionCount": len(regions),
        "completeEyeballNodes": 2,
        "contourSymmetry": contour_symmetry,
        "sdfUsed": False,
    }
    for name, relative in required_artifacts.items():
        root_manifest.setdefault("artifactSha256", {})[name] = sha256_file(
            template_root / str(relative)
        )
    atomic_write_json(root_manifest_path, root_manifest)
    return {
        "ok": True,
        "templateId": "TemplateHeadV0",
        "schemaVersion": ANATOMY_SCHEMA_VERSION,
        "geometryChanged": geometry_changed,
        "uvChanged": uv_changed,
        "computeGeometrySha256": after_compute_hash,
        "deliveryGeometrySha256": after_delivery_hash,
        "fittingContours": fitting_metrics,
        "contourSymmetry": contour_symmetry,
    }
