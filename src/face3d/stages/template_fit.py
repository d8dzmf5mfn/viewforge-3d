from __future__ import annotations

import copy
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional
import trimesh
from PIL import Image
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

from face3d.config import Face3DConfig
from face3d.errors import fail
from face3d.io import atomic_write_bytes, atomic_write_json, sha256_file
from face3d.models import REQUIRED_VIEWS, CameraRecord, ViewRole
from face3d.profiles.face_v1 import MEDIAPIPE_TO_IBUG68
from face3d.stages.fit import (
    _axis_angle_to_matrix,
    _initial_camera,
    _mask_boundary_distance,
    _project,
    _render_silhouette,
    _sample_distance_field,
    _save_silhouette_overlay,
    _silhouette_iou,
)
from face3d.template_head_anatomy import _self_intersection_pairs
from face3d.template_head_v0 import _edge_and_component_metrics
from face3d.unified_head import EyeballAsset, UnifiedHeadAsset, geometry_hash

_STABILITY_AREA_RATIO = 1e-4
_MINIMUM_SIGNED_AREA_RATIO = 0.03


@dataclass(frozen=True, slots=True)
class TemplateLandmarkBinding:
    normalized: np.ndarray
    points: np.ndarray
    valid: np.ndarray
    triangle: np.ndarray
    barycentric: np.ndarray
    surface_offset: np.ndarray

    @classmethod
    def load(
        cls,
        source: Path,
        asset: UnifiedHeadAsset,
        manifest: dict[str, Any],
    ) -> TemplateLandmarkBinding:
        with np.load(source, allow_pickle=False) as payload:
            required = {
                "normalized",
                "points",
                "valid",
                "triangle",
                "barycentric",
                "surface_offset",
            }
            missing = sorted(required - set(payload.files))
            if missing:
                fail(
                    "template-landmark-binding-invalid",
                    "TemplateHeadV0 地标绑定缺少字段",
                    stage="template-fit",
                    details={"missing": missing},
                )
            binding = cls(
                normalized=np.asarray(payload["normalized"], dtype=np.float64),
                points=np.asarray(payload["points"], dtype=np.float64),
                valid=np.asarray(payload["valid"], dtype=bool),
                triangle=np.asarray(payload["triangle"], dtype=np.int64),
                barycentric=np.asarray(payload["barycentric"], dtype=np.float64),
                surface_offset=np.asarray(payload["surface_offset"], dtype=np.float64),
            )
        compute_hash = geometry_hash(asset.skin_vertices, asset.skin_faces)
        expected_hash = manifest.get("landmarks", {}).get("bindingGeometrySha256")
        if expected_hash != compute_hash:
            fail(
                "template-landmark-binding-hash-mismatch",
                "地标绑定与当前 TemplateHeadV0 计算拓扑不一致",
                stage="template-fit",
                details={"expected": expected_hash, "actual": compute_hash},
            )
        if binding.normalized.shape[0] < 478 or binding.points.shape != (
            len(binding.normalized),
            3,
        ):
            fail(
                "template-landmark-binding-invalid",
                "TemplateHeadV0 地标绑定形状错误",
                stage="template-fit",
            )
        selected = np.flatnonzero(binding.valid)
        if not len(selected) or np.any(binding.triangle[selected] >= len(asset.skin_faces)):
            fail(
                "template-landmark-binding-invalid",
                "TemplateHeadV0 地标绑定包含越界三角面",
                stage="template-fit",
            )
        reconstructed = np.einsum(
            "nvc,nv->nc",
            asset.skin_vertices[asset.skin_faces[binding.triangle[selected]]],
            binding.barycentric[selected],
        )
        tolerance = max(
            float(np.linalg.norm(np.ptp(asset.skin_vertices, axis=0))) * 1e-7,
            1e-7,
        )
        error = np.linalg.norm(reconstructed - binding.points[selected], axis=1)
        if np.max(error, initial=0.0) > tolerance:
            fail(
                "template-landmark-binding-invalid",
                "TemplateHeadV0 地标重心坐标不能重建表面点",
                stage="template-fit",
                details={
                    "maximumError": float(np.max(error)),
                    "tolerance": tolerance,
                },
            )
        return binding


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    return np.unique(
        np.sort(
            np.concatenate(
                (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]),
                axis=0,
            ),
            axis=1,
        ),
        axis=0,
    )


def deformation_stability_weights(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, dict[str, int | float]]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    triangles = vertices[faces]
    double_area = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    median_area = max(float(np.median(double_area)), 1e-18)
    threshold = median_area * _STABILITY_AREA_RATIO
    fragile_faces = np.flatnonzero(double_area <= threshold)
    fragile_vertices = np.unique(faces[fragile_faces])
    weights = np.ones(len(vertices), dtype=np.float64)
    weights[fragile_vertices] = 0.0
    edges = _unique_edges(faces)
    frontier = np.zeros(len(vertices), dtype=bool)
    frontier[fragile_vertices] = True
    visited = frontier.copy()
    for transition_weight in (0.25, 0.60, 0.85):
        touching = frontier[edges[:, 0]] | frontier[edges[:, 1]]
        neighbors = np.unique(edges[touching])
        new_vertices = neighbors[~visited[neighbors]]
        weights[new_vertices] = np.minimum(weights[new_vertices], transition_weight)
        frontier[:] = False
        frontier[new_vertices] = True
        visited[new_vertices] = True
    return weights, {
        "fragileFaceCount": int(len(fragile_faces)),
        "lockedVertexCount": int(len(fragile_vertices)),
        "transitionVertexCount": int(np.count_nonzero((weights > 0.0) & (weights < 1.0))),
        "doubleAreaThreshold": threshold,
    }


def _dynamic_silhouette_indices(projected: torch.Tensor, bins: int = 40) -> torch.Tensor:
    detached = projected.detach()
    y_min = torch.quantile(detached[:, 1], 0.01)
    y_max = torch.quantile(detached[:, 1], 0.99)
    edges = torch.linspace(y_min, y_max, bins + 1, dtype=projected.dtype)
    selected: list[torch.Tensor] = []
    for index in range(bins):
        candidates = torch.nonzero(
            (detached[:, 1] >= edges[index]) & (detached[:, 1] < edges[index + 1]),
            as_tuple=False,
        ).flatten()
        if not candidates.numel():
            continue
        x = detached[candidates, 0]
        selected.extend((candidates[torch.argmin(x)], candidates[torch.argmax(x)]))
    if not selected:
        return torch.empty(0, dtype=torch.long)
    return torch.unique(torch.stack(selected))


def _mask_horizontal_profile(
    mask: np.ndarray,
    bottom_y: float,
    bins: int = 48,
) -> np.ndarray:
    foreground = np.asarray(mask) > 127
    rows = np.flatnonzero(np.any(foreground, axis=1))
    if not len(rows):
        return np.empty((0, 5), dtype=np.float64)
    top = int(rows[0])
    bottom = int(np.clip(round(bottom_y), top + 2, foreground.shape[0] - 1))
    edges = np.linspace(top, bottom + 1, bins + 1)
    profile: list[tuple[float, float, float, float, float]] = []
    for index in range(bins):
        row_start = int(math.floor(edges[index]))
        row_stop = int(math.ceil(edges[index + 1]))
        left: list[int] = []
        right: list[int] = []
        sampled_rows: list[int] = []
        for row in range(row_start, min(row_stop, foreground.shape[0])):
            columns = np.flatnonzero(foreground[row])
            if len(columns):
                left.append(int(columns[0]))
                right.append(int(columns[-1]))
                sampled_rows.append(row)
        if sampled_rows:
            profile.append(
                (
                    float(row_start),
                    float(row_stop),
                    float(np.mean(sampled_rows)),
                    float(np.median(left)),
                    float(np.median(right)),
                )
            )
    return np.asarray(profile, dtype=np.float64)


def _silhouette_profile_loss(
    projected: torch.Tensor,
    profile: torch.Tensor,
    diagonal: float,
) -> torch.Tensor:
    if not projected.numel() or not profile.numel():
        return torch.zeros((), dtype=projected.dtype)
    detached = projected.detach()
    residuals: list[torch.Tensor] = []
    for row_start, row_stop, target_y, target_left, target_right in profile:
        candidates = torch.nonzero(
            (detached[:, 1] >= row_start) & (detached[:, 1] < row_stop),
            as_tuple=False,
        ).flatten()
        if not candidates.numel():
            continue
        x = detached[candidates, 0]
        left = projected[candidates[torch.argmin(x)]]
        right = projected[candidates[torch.argmax(x)]]
        residuals.extend(
            (
                (left[0] - target_left) / diagonal,
                (right[0] - target_right) / diagonal,
                (0.5 * (left[1] + right[1]) - target_y) / diagonal,
            )
        )
    if not residuals:
        return torch.zeros((), dtype=projected.dtype)
    stacked = torch.stack(residuals)
    # Residuals are normalized by the image diagonal and normally sit well
    # below SmoothL1's unit transition. SmoothL1 therefore made the contour
    # gradient almost vanish; L1 keeps a useful signal for the cranium and ears.
    return torch.mean(torch.abs(stacked))


def spectral_deformation_basis(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic low-frequency normalized graph basis."""

    vertices = np.asarray(vertices, dtype=np.float64)
    edges = _unique_edges(np.asarray(faces, dtype=np.int64))
    vertex_count = len(vertices)
    count = min(int(count), max(vertex_count - 2, 1))
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    columns = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = coo_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(vertex_count, vertex_count),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inverse_sqrt = np.zeros_like(degree)
    positive = degree > 0
    inverse_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    normalized = eye(vertex_count, format="csr") - adjacency.multiply(
        inverse_sqrt[:, None]
    ).multiply(inverse_sqrt[None, :])
    initial = np.linspace(0.5, 1.5, vertex_count, dtype=np.float64)
    eigenvalues, eigenvectors = eigsh(
        normalized,
        k=count + 1,
        which="SM",
        v0=initial,
        tol=1e-9,
        maxiter=max(vertex_count * 4, 2000),
    )
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order][1 : count + 1]
    basis = eigenvectors[:, order][:, 1 : count + 1]
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1.0
    basis /= np.maximum(np.max(np.abs(basis), axis=0, keepdims=True), 1e-12)
    return basis.astype(np.float64), eigenvalues.astype(np.float64)


def _symmetry_pairs(vertices: np.ndarray, center_x: float) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    face_width = max(float(np.ptp(vertices[:, 0])), 1e-12)
    first = np.flatnonzero(vertices[:, 0] > center_x + face_width * 0.005)
    second = np.flatnonzero(vertices[:, 0] < center_x - face_width * 0.005)
    if not len(first) or not len(second):
        return np.empty((0, 2), dtype=np.int64)
    mirrored = vertices[first].copy()
    mirrored[:, 0] = 2.0 * center_x - mirrored[:, 0]
    distance, nearest = cKDTree(vertices[second]).query(mirrored, k=1)
    keep = distance <= face_width * 0.035
    pairs = np.column_stack((first[keep], second[nearest[keep]])).astype(np.int64)
    return np.unique(pairs, axis=0)


def triangle_orientation_metrics(
    reference_vertices: np.ndarray,
    deformed_vertices: np.ndarray,
    faces: np.ndarray,
) -> dict[str, float | int]:
    faces = np.asarray(faces, dtype=np.int64)
    reference = np.asarray(reference_vertices, dtype=np.float64)[faces]
    deformed = np.asarray(deformed_vertices, dtype=np.float64)[faces]
    reference_cross = np.cross(reference[:, 1] - reference[:, 0], reference[:, 2] - reference[:, 0])
    deformed_cross = np.cross(deformed[:, 1] - deformed[:, 0], deformed[:, 2] - deformed[:, 0])
    reference_area = np.maximum(np.linalg.norm(reference_cross, axis=1), 1e-12)
    signed_ratio = np.einsum("ij,ij->i", deformed_cross, reference_cross) / reference_area**2
    return {
        "flippedTriangleCount": int(np.count_nonzero(signed_ratio <= 0.0)),
        "minimumSignedAreaRatio": float(np.min(signed_ratio)),
        "p01SignedAreaRatio": float(np.quantile(signed_ratio, 0.01)),
    }


def collision_safe_deformation(
    reference_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    maximum_passes: int = 12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Locally attenuate displacement until collision and orientation gates pass."""

    reference = np.asarray(reference_vertices, dtype=np.float64)
    candidate = np.asarray(candidate_vertices, dtype=np.float32).astype(np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    displacement = candidate - reference
    factors = np.ones(len(reference), dtype=np.float64)
    edges = _unique_edges(faces)
    neighbors: list[list[int]] = [[] for _ in range(len(reference))]
    for first, second in edges:
        neighbors[int(first)].append(int(second))
        neighbors[int(second)].append(int(first))

    def pairs_for(vertices: np.ndarray) -> np.ndarray:
        return _self_intersection_pairs(
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        )

    reference_triangles = reference[faces]
    reference_cross = np.cross(
        reference_triangles[:, 1] - reference_triangles[:, 0],
        reference_triangles[:, 2] - reference_triangles[:, 0],
    )
    reference_area_squared = np.maximum(
        np.einsum("ij,ij->i", reference_cross, reference_cross),
        1e-24,
    )

    def unsafe_faces_for(vertices: np.ndarray) -> np.ndarray:
        triangles = vertices[faces]
        deformed_cross = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        signed_ratio = (
            np.einsum("ij,ij->i", deformed_cross, reference_cross) / reference_area_squared
        )
        return np.flatnonzero(signed_ratio < _MINIMUM_SIGNED_AREA_RATIO)

    current = candidate.copy()
    pairs = pairs_for(current)
    unsafe_faces = unsafe_faces_for(current)
    initial_pair_count = len(pairs)
    initial_unsafe_face_count = len(unsafe_faces)
    touched = np.zeros(len(reference), dtype=bool)
    passes = 0
    while (len(pairs) or len(unsafe_faces)) and passes < maximum_passes:
        pair_faces = pairs.reshape(-1) if len(pairs) else np.empty(0, dtype=np.int64)
        involved_faces = np.unique(np.concatenate((pair_faces, unsafe_faces)))
        core = np.unique(faces[involved_faces].reshape(-1))
        rings: list[np.ndarray] = [core]
        seen = set(int(value) for value in core)
        frontier = set(seen)
        for _ in range(2):
            expanded = {
                neighbor
                for vertex in frontier
                for neighbor in neighbors[vertex]
                if neighbor not in seen
            }
            rings.append(np.asarray(sorted(expanded), dtype=np.int64))
            seen.update(expanded)
            frontier = expanded
        for ring, attenuation in zip(rings, (0.25, 0.55, 0.82), strict=True):
            if len(ring):
                factors[ring] *= attenuation
                touched[ring] = True
        current = (
            (reference + displacement * factors[:, None]).astype(np.float32).astype(np.float64)
        )
        pairs = pairs_for(current)
        unsafe_faces = unsafe_faces_for(current)
        passes += 1

    nonzero = np.linalg.norm(displacement, axis=1) > 1e-12
    retained = factors[nonzero] if np.any(nonzero) else np.ones(1, dtype=np.float64)
    metrics: dict[str, Any] = {
        "method": "local-k-ring-displacement-attenuation",
        "initialPairCount": initial_pair_count,
        "finalPairCount": len(pairs),
        "initialUnsafeTriangleCount": initial_unsafe_face_count,
        "finalUnsafeTriangleCount": len(unsafe_faces),
        "minimumSignedAreaRatio": _MINIMUM_SIGNED_AREA_RATIO,
        "passes": passes,
        "maximumPasses": maximum_passes,
        "touchedVertexCount": int(np.count_nonzero(touched)),
        "meanRetainedDisplacementFraction": float(np.mean(retained)),
        "minimumRetainedDisplacementFraction": float(np.min(retained)),
        "passed": len(pairs) == 0 and len(unsafe_faces) == 0,
    }
    return current, metrics


def eyelid_contact_corrected_eye(
    eye: EyeballAsset,
    vertices: np.ndarray,
    eyelid_region: np.ndarray,
    maximum_gap_ratio: float,
    clearance_region: np.ndarray | None = None,
) -> tuple[EyeballAsset, dict[str, Any]]:
    """Increase eye radius minimally so the fitted eyelid keeps a valid contact band."""

    region = np.asarray(eyelid_region, dtype=np.int64)
    distances = np.linalg.norm(
        np.asarray(vertices, dtype=np.float64)[region] - eye.center,
        axis=1,
    )
    radius = float(eye.radius)
    initial_radius = radius
    target_gap = max(float(maximum_gap_ratio) - 5e-4, 0.0)
    maximum_radius_candidates = [
        initial_radius * 1.10,
        float(np.min(distances)) / (1.0 - 1e-6),
    ]
    if clearance_region is not None:
        clearance = np.linalg.norm(
            np.asarray(vertices, dtype=np.float64)[np.asarray(clearance_region, dtype=np.int64)]
            - eye.center,
            axis=1,
        )
        maximum_radius_candidates.append(float(np.min(clearance)) / 1.002)
    maximum_radius = min(maximum_radius_candidates)
    radius = min(radius, maximum_radius)
    iterations = 0
    for _ in range(8):
        gaps = (distances - radius) / radius
        contact = gaps[(gaps >= -1e-5) & (gaps <= maximum_gap_ratio + 0.001)]
        if len(contact) < 48:
            break
        p99 = float(np.quantile(contact, 0.99))
        if p99 <= maximum_gap_ratio + 1e-5:
            break
        contact_distances = contact * radius + radius
        required = float(np.quantile(contact_distances, 0.99)) / (1.0 + target_gap)
        updated = min(max(radius, required), maximum_radius)
        if updated <= radius + np.finfo(np.float64).eps:
            break
        radius = updated
        iterations += 1

    gaps = (distances - radius) / radius
    contact = gaps[(gaps >= -1e-5) & (gaps <= maximum_gap_ratio + 0.001)]
    minimum = float(np.min(contact, initial=np.inf))
    p99 = float(np.quantile(contact, 0.99)) if len(contact) else None
    passed = bool(
        len(contact) >= 48
        and minimum >= -1e-5
        and p99 is not None
        and p99 <= maximum_gap_ratio + 1e-5
    )
    corrected = EyeballAsset(
        center=np.asarray(eye.center, dtype=np.float64),
        radius=radius,
        gaze=np.asarray(eye.gaze, dtype=np.float64),
    )
    return corrected, {
        "method": "minimal-radius-contact-band-correction",
        "initialRadius": initial_radius,
        "finalRadius": radius,
        "radiusScale": radius / max(initial_radius, 1e-12),
        "maximumRadius": maximum_radius,
        "iterations": iterations,
        "contactVertexCount": len(contact),
        "contactGapMinimumR": minimum,
        "contactGapP99R": p99,
        "maximumGapRatio": maximum_gap_ratio,
        "passed": passed,
    }


def eyelid_support_corrected_vertices(
    vertices: np.ndarray,
    eye: EyeballAsset,
    eyelid_region: np.ndarray,
    maximum_gap_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Move only the active eyelid contact band onto a safe spherical shell."""

    corrected = np.asarray(vertices, dtype=np.float64).copy()
    region = np.asarray(eyelid_region, dtype=np.int64)
    relative = corrected[region] - eye.center
    distance = np.linalg.norm(relative, axis=1)
    gap = (distance - eye.radius) / eye.radius
    active = (gap >= -0.01) & (gap <= maximum_gap_ratio + 0.001)
    target_minimum = 0.001
    target_maximum = max(maximum_gap_ratio - 5e-4, target_minimum)
    desired_gap = np.clip(gap[active], target_minimum, target_maximum)
    desired_distance = eye.radius * (1.0 + desired_gap)
    safe_direction = relative[active] / np.maximum(distance[active, None], 1e-12)
    moved = safe_direction * desired_distance[:, None] + eye.center
    movement = np.linalg.norm(moved - corrected[region[active]], axis=1)
    corrected[region[active]] = moved
    return corrected.astype(np.float32).astype(np.float64), {
        "method": "active-contact-band-radial-support",
        "activeVertexCount": int(np.count_nonzero(active)),
        "movedVertexCount": int(np.count_nonzero(movement > 1e-12)),
        "maximumMovementEyeRadius": float(np.max(movement, initial=0.0) / max(eye.radius, 1e-12)),
        "meanMovementEyeRadius": float(
            np.mean(movement) / max(eye.radius, 1e-12) if len(movement) else 0.0
        ),
        "targetGapMinimumR": target_minimum,
        "targetGapMaximumR": target_maximum,
        "applied": bool(np.any(movement > 1e-12)),
    }


def _deformed_eye(
    eye: EyeballAsset,
    region: np.ndarray,
    template_center: np.ndarray,
    scale: np.ndarray,
    scaled_vertices: np.ndarray,
    deformed_vertices: np.ndarray,
) -> EyeballAsset:
    region = np.asarray(region, dtype=np.int64)
    center = (eye.center - template_center) * scale + template_center
    if len(region):
        center += np.mean(deformed_vertices[region] - scaled_vertices[region], axis=0)
    radius_scale = float(np.sqrt(max(scale[0] * scale[1], 1e-12)))
    return EyeballAsset(
        center=center,
        radius=float(eye.radius * radius_scale),
        gaze=np.asarray(eye.gaze, dtype=np.float64),
    )


def _load_template(config: Face3DConfig) -> tuple[UnifiedHeadAsset, TemplateLandmarkBinding]:
    if not config.is_v3:
        fail(
            "config-invalid",
            "TemplateHeadV0 拟合只接受 face-v3 配置",
            stage="template-fit",
        )
    template_path = config.resolve_optional_asset(config.assets.template_head)
    landmark_path = config.resolve_optional_asset(config.assets.template_landmarks)
    manifest_path = config.resolve_optional_asset(config.assets.template_manifest)
    if template_path is None or landmark_path is None or manifest_path is None:
        fail(
            "asset-missing",
            "face-v3 缺少 TemplateHeadV0 资产路径",
            stage="template-fit",
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        asset = UnifiedHeadAsset.load(template_path)
    except (OSError, ValueError, KeyError) as exc:
        fail(
            "template-asset-invalid",
            "无法读取 TemplateHeadV0 统一资产",
            stage="template-fit",
            details={"reason": str(exc)},
        )
    if manifest.get("templateId") != "TemplateHeadV0" or manifest.get("state") != "anatomy-ready":
        fail(
            "template-asset-invalid",
            "TemplateHeadV0 尚未完成 anatomy-ready 门禁",
            stage="template-fit",
        )
    binding = TemplateLandmarkBinding.load(landmark_path, asset, manifest)
    return asset, binding


def _observed_vertex_mask(asset: UnifiedHeadAsset) -> np.ndarray:
    names = (
        "face",
        "jaw",
        "nose",
        "mouth",
        "left_eye",
        "right_eye",
        "left_eyelid",
        "right_eyelid",
        "left_ear",
        "right_ear",
    )
    mask = np.zeros(len(asset.skin_vertices), dtype=bool)
    for name in names:
        mask[np.asarray(asset.regions[name], dtype=np.int64)] = True
    return mask


def _head_vertex_indices(asset: UnifiedHeadAsset) -> np.ndarray:
    names = (
        "cranium",
        "rear_cranium",
        "face",
        "jaw",
        "left_ear",
        "right_ear",
    )
    return np.unique(
        np.concatenate([np.asarray(asset.regions[name], dtype=np.int64) for name in names])
    )


def _view_data(run_dir: Path) -> tuple[dict[ViewRole, dict[str, Any]], dict[str, Any]]:
    intake_path = run_dir / "working" / "intake.json"
    if not intake_path.is_file():
        fail(
            "intake-missing",
            "TemplateHeadV0 拟合前缺少 intake.json",
            stage="template-fit",
        )
    intake = json.loads(intake_path.read_text(encoding="utf-8"))
    views = {ViewRole(item["role"]): item for item in intake.get("views", [])}
    missing = [role.value for role in REQUIRED_VIEWS if role not in views]
    if missing:
        fail(
            "intake-invalid",
            "TemplateHeadV0 拟合缺少固定三视图",
            stage="template-fit",
            details={"missing": missing},
        )
    return views, intake


def _require_confirmed_masks(
    run_dir: Path,
    views: dict[ViewRole, dict[str, Any]],
    config: Face3DConfig,
) -> None:
    if not config.input.require_mask_confirmation:
        return
    confirmation_path = run_dir / "working" / "masks" / "confirmed.json"
    try:
        confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        confirmation = {}
    hashes = confirmation.get("sha256", {}) if isinstance(confirmation, dict) else {}
    mismatches: list[str] = []
    for role in REQUIRED_VIEWS:
        mask_path = Path(views[role]["mask_path"])
        expected = hashes.get(role.value) if isinstance(hashes, dict) else None
        if not mask_path.is_file() or expected != sha256_file(mask_path):
            mismatches.append(role.value)
    if confirmation.get("confirmed") is not True or mismatches:
        fail(
            "mask-review-required",
            "TemplateHeadV0 拟合要求先确认当前三张 mask",
            stage="template-fit",
            details={
                "confirmation": str(confirmation_path),
                "mismatchedViews": mismatches,
                "nextCommand": f"face3d confirm-masks --run {run_dir}",
            },
        )


def _save_npz(destination: Path, **arrays: np.ndarray) -> None:
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    atomic_write_bytes(destination, output.getvalue())


def run_template_fit(run_dir: Path, config: Face3DConfig) -> dict[str, Any]:
    torch.use_deterministic_algorithms(config.deterministic)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    run_dir = run_dir.expanduser().resolve()
    asset, binding = _load_template(config)
    views, _ = _view_data(run_dir)
    _require_confirmed_masks(run_dir, views, config)
    template_vertices = np.asarray(asset.skin_vertices, dtype=np.float64)
    faces_numpy = np.asarray(asset.skin_faces, dtype=np.int64)
    template_mesh = trimesh.Trimesh(
        vertices=template_vertices,
        faces=faces_numpy,
        process=False,
    )
    head_indices_numpy = _head_vertex_indices(asset)
    head_vertices = template_vertices[head_indices_numpy]
    face_width = max(float(np.ptp(head_vertices[:, 0])), 1e-12)
    template_center = np.mean(
        np.stack((head_vertices.min(axis=0), head_vertices.max(axis=0))),
        axis=0,
    )
    basis_numpy, eigenvalues = spectral_deformation_basis(
        template_vertices,
        faces_numpy,
        config.fit.low_frequency_basis_size,
    )
    stability_weights_numpy, stability_metrics = deformation_stability_weights(
        template_vertices,
        faces_numpy,
    )
    valid_landmarks = np.flatnonzero(binding.valid[:468])
    if len(valid_landmarks) < 400:
        fail(
            "template-landmark-binding-insufficient",
            "TemplateHeadV0 可用皮肤地标不足 400",
            stage="template-fit",
            details={"count": len(valid_landmarks)},
        )
    stable = np.asarray(MEDIAPIPE_TO_IBUG68[17:], dtype=np.int64)
    stable = stable[np.isin(stable, valid_landmarks)]
    target_numpy: list[np.ndarray] = []
    target_depth_numpy: list[np.ndarray] = []
    distance_numpy: list[np.ndarray] = []
    silhouette_profiles_numpy: list[np.ndarray] = []
    silhouette_bottoms: list[int] = []
    masks: list[np.ndarray] = []
    sizes: list[tuple[int, int]] = []
    initial_focal = float(
        np.mean([max(views[role]["width"], views[role]["height"]) for role in REQUIRED_VIEWS]) * 1.2
    )
    initial_rotations: list[np.ndarray] = []
    initial_translations: list[np.ndarray] = []
    for role in REQUIRED_VIEWS:
        view = views[role]
        with np.load(Path(view["landmarks_path"]), allow_pickle=False) as payload:
            all_landmarks = np.asarray(payload["all"], dtype=np.float64)
        if len(all_landmarks) < 478 or not np.isfinite(all_landmarks).all():
            fail(
                "landmarks-invalid",
                f"{role.value} 的 MediaPipe 地标不足 478 或包含非有限值",
                stage="template-fit",
            )
        target_pixels = all_landmarks[:, :2] * np.asarray(
            [view["width"], view["height"]],
            dtype=np.float64,
        )
        target_numpy.append(target_pixels[valid_landmarks])
        target_depth_numpy.append(all_landmarks[valid_landmarks, 2])
        rotation, translation = _initial_camera(
            binding.points[stable],
            target_pixels[stable],
            view["width"],
            view["height"],
            initial_focal,
        )
        initial_rotations.append(rotation)
        initial_translations.append(translation)
        mask = cv2.imread(str(view["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            fail(
                "mask-review-required",
                f"无法读取 mask: {role.value}",
                stage="template-fit",
            )
        masks.append(mask)
        distance_numpy.append(_mask_boundary_distance(mask))
        silhouette_bottom = int(
            np.clip(
                round(target_pixels[152, 1]),
                1,
                view["height"],
            )
        )
        silhouette_bottoms.append(silhouette_bottom)
        silhouette_profiles_numpy.append(_mask_horizontal_profile(mask, silhouette_bottom))
        sizes.append((int(view["width"]), int(view["height"])))

    dtype = torch.float64
    template = torch.as_tensor(template_vertices, dtype=dtype)
    faces = torch.as_tensor(faces_numpy, dtype=torch.long)
    basis = torch.as_tensor(basis_numpy, dtype=dtype)
    normals = torch.as_tensor(
        np.asarray(template_mesh.vertex_normals).copy(),
        dtype=dtype,
    )
    edges_numpy = _unique_edges(faces_numpy)
    edges = torch.as_tensor(edges_numpy, dtype=torch.long)
    degree_numpy = np.bincount(edges_numpy.reshape(-1), minlength=len(template_vertices))
    degree = torch.as_tensor(degree_numpy, dtype=dtype)
    observed = torch.as_tensor(_observed_vertex_mask(asset).astype(np.float64), dtype=dtype)
    stability_weights = torch.as_tensor(stability_weights_numpy, dtype=dtype)
    binding_faces = torch.as_tensor(
        faces_numpy[binding.triangle[valid_landmarks]],
        dtype=torch.long,
    )
    binding_barycentric = torch.as_tensor(
        binding.barycentric[valid_landmarks],
        dtype=dtype,
    )
    valid_position = {int(value): index for index, value in enumerate(valid_landmarks)}
    feature_positions = torch.as_tensor(
        [valid_position[index] for index in MEDIAPIPE_TO_IBUG68 if index in valid_position],
        dtype=torch.long,
    )
    targets = [torch.as_tensor(value, dtype=dtype) for value in target_numpy]
    depth_targets = [torch.as_tensor(value, dtype=dtype) for value in target_depth_numpy]
    distances = [torch.as_tensor(value, dtype=dtype) for value in distance_numpy]
    silhouette_profiles = [
        torch.as_tensor(value, dtype=dtype) for value in silhouette_profiles_numpy
    ]
    head_indices = torch.as_tensor(head_indices_numpy, dtype=torch.long)
    symmetry_center_x = float((asset.left_eye.center[0] + asset.right_eye.center[0]) * 0.5)
    symmetry_pairs = torch.as_tensor(
        _symmetry_pairs(template_vertices, symmetry_center_x),
        dtype=torch.long,
    )
    ear_indices = [
        torch.as_tensor(asset.regions["left_ear"], dtype=torch.long),
        torch.as_tensor(asset.regions["right_ear"], dtype=torch.long),
    ]
    all_ear_indices = torch.unique(torch.cat(ear_indices))
    eyelid_regions = [
        torch.as_tensor(asset.regions["left_eyelid"], dtype=torch.long),
        torch.as_tensor(asset.regions["right_eyelid"], dtype=torch.long),
    ]
    eye_centers = [
        torch.as_tensor(asset.left_eye.center, dtype=dtype),
        torch.as_tensor(asset.right_eye.center, dtype=dtype),
    ]
    eye_radii = [
        torch.as_tensor(asset.left_eye.radius, dtype=dtype),
        torch.as_tensor(asset.right_eye.radius, dtype=dtype),
    ]
    eyelid_contacts: list[torch.Tensor] = []
    eyelid_contact_targets: list[torch.Tensor] = []
    eye_socket_contacts: list[torch.Tensor] = []
    eye_socket_targets: list[torch.Tensor] = []
    for eyeball, region, socket_region in zip(
        (asset.left_eye, asset.right_eye),
        (asset.regions["left_eyelid"], asset.regions["right_eyelid"]),
        (asset.regions["left_eye"], asset.regions["right_eye"]),
        strict=True,
    ):
        region_numpy = np.asarray(region, dtype=np.int64)
        gap = (
            np.linalg.norm(template_vertices[region_numpy] - eyeball.center, axis=1)
            - eyeball.radius
        ) / eyeball.radius
        contact_mask = (gap >= -1e-5) & (gap <= config.anatomy.eyelid_clearance_ratio_max + 1e-5)
        if np.count_nonzero(contact_mask) < 48:
            fail(
                "template-eye-contact-invalid",
                "TemplateHeadV0 眼睑接触锚点不足",
                stage="template-fit",
                details={"contactVertexCount": int(np.count_nonzero(contact_mask))},
            )
        eyelid_contacts.append(torch.as_tensor(region_numpy[contact_mask], dtype=torch.long))
        eyelid_contact_targets.append(
            torch.as_tensor(
                np.clip(
                    gap[contact_mask],
                    0.0,
                    config.anatomy.eyelid_clearance_ratio_max,
                ),
                dtype=dtype,
            )
        )
        socket_numpy = np.asarray(socket_region, dtype=np.int64)
        socket_gap = (
            np.linalg.norm(template_vertices[socket_numpy] - eyeball.center, axis=1)
            - eyeball.radius
        ) / eyeball.radius
        socket_mask = (socket_gap >= 0.0) & (socket_gap <= 0.15)
        if np.count_nonzero(socket_mask) < 128:
            fail(
                "template-eye-socket-invalid",
                "TemplateHeadV0 眼窝球面约束锚点不足",
                stage="template-fit",
                details={"socketVertexCount": int(np.count_nonzero(socket_mask))},
            )
        eye_socket_contacts.append(torch.as_tensor(socket_numpy[socket_mask], dtype=torch.long))
        eye_socket_targets.append(torch.as_tensor(socket_gap[socket_mask], dtype=dtype))
    ear_weights: list[torch.Tensor] = []
    ear_roots: list[torch.Tensor] = []
    for side, region in zip((1.0, -1.0), ear_indices, strict=True):
        lateral = side * (template[region, 0] - symmetry_center_x)
        low = torch.quantile(lateral, 0.08)
        high = torch.quantile(lateral, 0.92)
        normalized = ((lateral - low) / (high - low).clamp_min(1e-8)).clamp(0.0, 1.0)
        weight = normalized.square() * (3.0 - 2.0 * normalized)
        root = region[lateral <= torch.quantile(lateral, 0.20)]
        ear_weights.append(weight)
        ear_roots.append(root)

    rotations = torch.tensor(np.asarray(initial_rotations), dtype=dtype, requires_grad=True)
    translations = torch.tensor(np.asarray(initial_translations), dtype=dtype, requires_grad=True)
    log_focal = torch.tensor(math.log(initial_focal), dtype=dtype, requires_grad=True)
    scale_raw = torch.zeros(3, dtype=dtype, requires_grad=True)
    low_coefficients = torch.zeros(
        (basis.shape[1], 3),
        dtype=dtype,
        requires_grad=True,
    )
    detail_raw = torch.zeros(len(template_vertices), dtype=dtype, requires_grad=True)
    ear_raw = torch.zeros((2, 5), dtype=dtype, requires_grad=True)
    center = torch.as_tensor(template_center, dtype=dtype)
    maximum_displacement = face_width * config.fit.maximum_vertex_displacement_face_width
    maximum_detail = face_width * config.fit.maximum_normal_offset_face_width

    def current_vertices(detail_enabled: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = torch.exp(torch.tanh(scale_raw) * 0.25)
        scaled = (template - center) * scale + center
        low_raw = basis @ low_coefficients
        low_norm = torch.linalg.vector_norm(low_raw, dim=1, keepdim=True).clamp_min(1e-12)
        low = (
            low_raw
            / low_norm
            * torch.tanh(low_norm)
            * maximum_displacement
            * stability_weights[:, None]
        )
        detail = (
            torch.tanh(detail_raw) * maximum_detail * observed * stability_weights
            if detail_enabled
            else torch.zeros_like(detail_raw)
        )
        vertices = scaled + low + normals * detail[:, None]
        ear_delta = torch.zeros_like(vertices)
        ear_weight_sum = torch.zeros(len(vertices), dtype=dtype)
        for ear_index, (side, region, weight, root) in enumerate(
            zip((1.0, -1.0), ear_indices, ear_weights, ear_roots, strict=True)
        ):
            values = vertices[region]
            pivot = torch.mean(vertices[root], dim=0)
            relative = values - pivot
            height_scale = torch.exp(torch.tanh(ear_raw[ear_index, 2]) * 0.22)
            depth_scale = torch.exp(torch.tanh(ear_raw[ear_index, 3]) * 0.22)
            angle = torch.tanh(ear_raw[ear_index, 4]) * math.radians(25.0) * side
            cosine = torch.cos(angle)
            sine = torch.sin(angle)
            scaled_relative = torch.stack(
                (
                    relative[:, 0],
                    relative[:, 1] * height_scale,
                    relative[:, 2] * depth_scale,
                ),
                dim=1,
            )
            rotated = torch.stack(
                (
                    cosine * scaled_relative[:, 0] + sine * scaled_relative[:, 2],
                    scaled_relative[:, 1],
                    -sine * scaled_relative[:, 0] + cosine * scaled_relative[:, 2],
                ),
                dim=1,
            )
            translation = torch.stack(
                (
                    side * torch.tanh(ear_raw[ear_index, 0]) * face_width * 0.10,
                    torch.tanh(ear_raw[ear_index, 1]) * face_width * 0.08,
                    torch.zeros((), dtype=dtype),
                )
            )
            transformed = pivot + rotated + translation
            weighted_delta = (transformed - values) * weight[:, None]
            ear_delta.index_add_(0, region, weighted_delta)
            ear_weight_sum.index_add_(0, region, weight)
        vertices = vertices + ear_delta / ear_weight_sum.clamp_min(1.0)[:, None]
        return vertices, scaled, scale

    def loss_terms(detail_enabled: bool) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        vertices, scaled, scale = current_vertices(detail_enabled)
        landmark_points = torch.einsum(
            "nvc,nv->nc",
            vertices[binding_faces],
            binding_barycentric,
        )
        zero = torch.zeros((), dtype=dtype)
        terms = {
            "landmark": zero.clone(),
            "denseLandmark": zero.clone(),
            "contour": zero.clone(),
            "relativeDepth": zero.clone(),
            "earContour": zero.clone(),
            "silhouetteProfile": zero.clone(),
        }
        focal = torch.exp(log_focal)
        for index, _role in enumerate(REQUIRED_VIEWS):
            principal = torch.tensor(
                [sizes[index][0] / 2, sizes[index][1] / 2],
                dtype=dtype,
            )
            projected_landmarks = _project(
                landmark_points,
                rotations[index],
                translations[index],
                focal,
                principal,
            )
            diagonal = math.hypot(*sizes[index])
            residual = (projected_landmarks - targets[index]) / diagonal
            terms["denseLandmark"] = terms["denseLandmark"] + torch.mean(
                torch.linalg.vector_norm(residual, dim=1)
            )
            terms["landmark"] = terms["landmark"] + torch.mean(
                torch.linalg.vector_norm(residual[feature_positions], dim=1)
            )
            projected_head = _project(
                vertices[head_indices],
                rotations[index],
                translations[index],
                focal,
                principal,
            )
            silhouette_indices = _dynamic_silhouette_indices(projected_head, bins=48)
            silhouette = projected_head[silhouette_indices]
            terms["contour"] = terms["contour"] + torch.mean(
                _sample_distance_field(distances[index], silhouette) / diagonal
            )
            terms["silhouetteProfile"] = terms["silhouetteProfile"] + (
                _silhouette_profile_loss(
                    projected_head,
                    silhouette_profiles[index],
                    diagonal,
                )
            )
            original_silhouette_indices = head_indices[silhouette_indices]
            ear_positions = torch.isin(original_silhouette_indices, all_ear_indices)
            ear_silhouette = silhouette[ear_positions]
            if ear_silhouette.numel():
                terms["earContour"] = terms["earContour"] + torch.mean(
                    _sample_distance_field(distances[index], ear_silhouette) / diagonal
                )
            rotation = _axis_angle_to_matrix(rotations[index])
            camera_landmarks = landmark_points @ rotation.transpose(-1, -2) + translations[index]
            model_depth = camera_landmarks[:, 2]
            model_depth = (model_depth - torch.mean(model_depth)) / torch.clamp(
                torch.std(model_depth),
                min=1e-6,
            )
            target_depth = depth_targets[index]
            target_depth = (target_depth - torch.mean(target_depth)) / torch.clamp(
                torch.std(target_depth),
                min=1e-6,
            )
            terms["relativeDepth"] = terms["relativeDepth"] + torch_functional.smooth_l1_loss(
                model_depth,
                target_depth,
            )

        displacement = vertices - scaled
        edge_displacement = displacement[edges[:, 0]] - displacement[edges[:, 1]]
        laplacian = torch.zeros_like(displacement)
        laplacian.index_add_(0, edges[:, 0], edge_displacement)
        laplacian.index_add_(0, edges[:, 1], -edge_displacement)
        laplacian = laplacian / degree.clamp_min(1.0)[:, None]
        scaled_edge = scaled[edges[:, 0]] - scaled[edges[:, 1]]
        deformed_edge = vertices[edges[:, 0]] - vertices[edges[:, 1]]
        scaled_length = torch.linalg.vector_norm(scaled_edge, dim=1)
        deformed_length = torch.linalg.vector_norm(deformed_edge, dim=1)
        scaled_triangles = scaled[faces]
        deformed_triangles = vertices[faces]
        reference_cross = torch.linalg.cross(
            scaled_triangles[:, 1] - scaled_triangles[:, 0],
            scaled_triangles[:, 2] - scaled_triangles[:, 0],
        )
        deformed_cross = torch.linalg.cross(
            deformed_triangles[:, 1] - deformed_triangles[:, 0],
            deformed_triangles[:, 2] - deformed_triangles[:, 0],
        )
        reference_area = torch.linalg.vector_norm(reference_cross, dim=1).clamp_min(1e-12)
        signed_ratio = torch.sum(deformed_cross * reference_cross, dim=1) / reference_area.square()
        robust_faces = reference_area > torch.median(reference_area) * _STABILITY_AREA_RATIO
        inversion_violation = torch.relu(0.20 - signed_ratio[robust_faces])
        active_inversion_violation = inversion_violation[inversion_violation > 0.0]
        inversion_barrier = (
            torch.mean(active_inversion_violation.square())
            if active_inversion_violation.numel()
            else zero.clone()
        )
        if symmetry_pairs.numel():
            first = displacement[symmetry_pairs[:, 0]]
            second = displacement[symmetry_pairs[:, 1]]
            symmetry_delta = torch.stack(
                (
                    first[:, 0] + second[:, 0],
                    first[:, 1] - second[:, 1],
                    first[:, 2] - second[:, 2],
                ),
                dim=1,
            )
            symmetry = torch.mean(symmetry_delta.square()) / face_width**2
        else:
            symmetry = zero.clone()
        eyelid = zero.clone()
        for (
            original_center,
            original_radius,
            region,
            contact,
            target_gap,
            socket,
            socket_target,
        ) in zip(
            eye_centers,
            eye_radii,
            eyelid_regions,
            eyelid_contacts,
            eyelid_contact_targets,
            eye_socket_contacts,
            eye_socket_targets,
            strict=True,
        ):
            scaled_eye_center = (original_center - center) * scale + center
            moving_center = scaled_eye_center + torch.mean(displacement[region], dim=0)
            moving_radius = original_radius * torch.sqrt(scale[0] * scale[1])
            current_gap = (
                torch.linalg.vector_norm(vertices[contact] - moving_center, dim=1) - moving_radius
            ) / moving_radius.clamp_min(1e-8)
            current_socket_gap = (
                torch.linalg.vector_norm(vertices[socket] - moving_center, dim=1) - moving_radius
            ) / moving_radius.clamp_min(1e-8)
            eyelid = eyelid + torch.mean((current_gap - target_gap).square())
            eyelid = eyelid + torch.mean((current_socket_gap - socket_target).square())
        terms.update(
            {
                "shapePrior": torch.mean(low_coefficients.square())
                + torch.mean((scale - 1.0).square()),
                "detailPrior": torch.mean((displacement / maximum_displacement).square()),
                "laplacian": torch.mean((laplacian / face_width).square()),
                "arap": torch.mean(((deformed_length - scaled_length) / face_width).square()),
                "symmetry": symmetry,
                "inversionBarrier": inversion_barrier,
                "eyelidContact": eyelid,
                "earPrior": torch.mean(ear_raw.square())
                + torch.mean((ear_raw[0] - ear_raw[1]).square()),
                "focalPrior": (log_focal - math.log(initial_focal)).square(),
            }
        )
        return terms, vertices

    def objective(stage: str) -> torch.Tensor:
        terms, _ = loss_terms(stage == "detail")
        camera_data = (
            config.fit.landmark_weight * terms["landmark"]
            + config.fit.dense_landmark_weight * terms["denseLandmark"]
            + 0.002 * terms["focalPrior"]
        )
        if stage == "camera":
            return camera_data
        return (
            camera_data
            + config.fit.contour_weight * terms["contour"]
            + config.fit.contour_weight * 2.0 * terms["silhouetteProfile"]
            + config.fit.relative_depth_weight * terms["relativeDepth"]
            + config.fit.ear_constraint_weight * terms["earContour"]
            + config.fit.shape_prior_weight * terms["earPrior"]
            + config.fit.shape_prior_weight * terms["shapePrior"]
            + config.fit.local_offset_weight * terms["detailPrior"]
            + config.fit.laplacian_weight * terms["laplacian"]
            + config.fit.arap_weight * terms["arap"]
            + config.fit.symmetry_weight * terms["symmetry"]
            + config.fit.inversion_barrier_weight * terms["inversionBarrier"]
            + config.fit.eyelid_constraint_weight * terms["eyelidContact"]
        )

    total_iterations = config.fit.adam_iterations
    if total_iterations:
        camera_iterations = min(80, max(1, total_iterations // 8))
        low_iterations = max(1, round(total_iterations * 0.58))
        detail_iterations = max(1, total_iterations - low_iterations)
    else:
        camera_iterations = low_iterations = detail_iterations = 0
    stage_records: list[dict[str, Any]] = []

    def minimum_orientation(detail_enabled: bool) -> float:
        with torch.no_grad():
            vertices, scaled, _ = current_vertices(detail_enabled)
            scaled_triangles = scaled[faces]
            deformed_triangles = vertices[faces]
            reference_cross = torch.linalg.cross(
                scaled_triangles[:, 1] - scaled_triangles[:, 0],
                scaled_triangles[:, 2] - scaled_triangles[:, 0],
            )
            deformed_cross = torch.linalg.cross(
                deformed_triangles[:, 1] - deformed_triangles[:, 0],
                deformed_triangles[:, 2] - deformed_triangles[:, 0],
            )
            reference_area = torch.linalg.vector_norm(reference_cross, dim=1).clamp_min(1e-12)
            robust = reference_area > torch.median(reference_area) * _STABILITY_AREA_RATIO
            signed_ratio = (
                torch.sum(reference_cross * deformed_cross, dim=1) / reference_area.square()
            )
            return float(torch.min(signed_ratio[robust]))

    def optimize(stage: str, parameters: list[torch.Tensor], iterations: int) -> None:
        if not iterations:
            stage_records.append(
                {
                    "stage": stage,
                    "iterations": 0,
                    "objective": float(objective(stage).detach()),
                }
            )
            return
        initial_value = float(objective(stage).detach())
        if stage == "camera" and initial_value <= 1e-7:
            stage_records.append(
                {
                    "stage": stage,
                    "iterations": 0,
                    "requestedIterations": iterations,
                    "objective": initial_value,
                    "reason": "pnp-already-converged",
                }
            )
            return
        learning_rate = config.fit.learning_rate * (
            1.0 if stage == "camera" else (0.05 if stage == "detail" else 0.10)
        )
        if stage == "low-frequency":
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": [rotations, translations, log_focal],
                        "lr": learning_rate * 0.25,
                    },
                    {"params": [scale_raw], "lr": learning_rate * 0.25},
                    {
                        "params": [low_coefficients],
                        "lr": learning_rate / math.sqrt(max(int(basis.shape[1]), 1)),
                    },
                    {"params": [ear_raw], "lr": learning_rate * 0.50},
                ]
            )
        elif stage == "detail":
            optimizer = torch.optim.Adam(
                [
                    {"params": [detail_raw], "lr": learning_rate},
                    {"params": [ear_raw], "lr": learning_rate * 0.50},
                ]
            )
        else:
            optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        value = torch.zeros((), dtype=dtype)
        backtrack_count = 0
        backtracked_step_count = 0
        rejected_step_count = 0
        maximum_backtrack_exponent = 0
        for _ in range(iterations):
            optimizer.zero_grad()
            value = objective(stage)
            if not torch.isfinite(value):
                fail(
                    "fit-non-finite",
                    f"TemplateHeadV0 {stage} 阶段出现非有限损失",
                    stage="template-fit",
                )
            value.backward()
            previous = [parameter.detach().clone() for parameter in parameters]
            optimizer.step()
            if (
                stage != "camera"
                and minimum_orientation(stage == "detail") <= _MINIMUM_SIGNED_AREA_RATIO
            ):
                candidate = [parameter.detach().clone() for parameter in parameters]
                accepted = False
                for exponent in range(1, 14):
                    fraction = 0.5**exponent
                    with torch.no_grad():
                        for parameter, start, end in zip(
                            parameters,
                            previous,
                            candidate,
                            strict=True,
                        ):
                            parameter.copy_(start + (end - start) * fraction)
                    if minimum_orientation(stage == "detail") > _MINIMUM_SIGNED_AREA_RATIO:
                        accepted = True
                        backtrack_count += exponent
                        backtracked_step_count += 1
                        maximum_backtrack_exponent = max(
                            maximum_backtrack_exponent,
                            exponent,
                        )
                        break
                if not accepted:
                    with torch.no_grad():
                        for parameter, start in zip(parameters, previous, strict=True):
                            parameter.copy_(start)
                    backtrack_count += 13
                    rejected_step_count += 1
                    # Adam's moments describe a step that was not accepted.
                    # Reset them instead of permanently collapsing every group
                    # learning rate to near zero.
                    optimizer.state.clear()
        stage_records.append(
            {
                "stage": stage,
                "iterations": iterations,
                "objective": float(value.detach()),
                "learningRate": learning_rate,
                "finalLearningRates": [
                    float(parameter_group["lr"]) for parameter_group in optimizer.param_groups
                ],
                "inversionBacktrackCount": backtrack_count,
                "backtrackedStepCount": backtracked_step_count,
                "rejectedStepCount": rejected_step_count,
                "maximumBacktrackExponent": maximum_backtrack_exponent,
            }
        )
        del optimizer

    optimize("camera", [rotations, translations, log_focal], camera_iterations)
    optimize(
        "low-frequency",
        [rotations, translations, log_focal, scale_raw, low_coefficients, ear_raw],
        low_iterations,
    )
    optimize(
        "detail",
        [detail_raw, ear_raw],
        detail_iterations,
    )
    if config.fit.lbfgs_iterations:
        parameters = [scale_raw, low_coefficients, detail_raw, ear_raw]
        optimizer = torch.optim.LBFGS(
            parameters,
            max_iter=config.fit.lbfgs_iterations,
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            value = objective("detail")
            if not torch.isfinite(value):
                fail(
                    "fit-non-finite",
                    "TemplateHeadV0 LBFGS 阶段出现非有限损失",
                    stage="template-fit",
                )
            value.backward()
            return value

        previous = [parameter.detach().clone() for parameter in parameters]
        optimizer.step(closure)
        candidate = [parameter.detach().clone() for parameter in parameters]
        accepted_fraction = 1.0
        rejected = minimum_orientation(True) <= _MINIMUM_SIGNED_AREA_RATIO
        if rejected:
            accepted_fraction = 0.0
            with torch.no_grad():
                for exponent in range(1, 14):
                    fraction = 0.5**exponent
                    for parameter, start, end in zip(
                        parameters,
                        previous,
                        candidate,
                        strict=True,
                    ):
                        parameter.copy_(start + (end - start) * fraction)
                    if minimum_orientation(True) > _MINIMUM_SIGNED_AREA_RATIO:
                        accepted_fraction = fraction
                        rejected = False
                        break
                if rejected:
                    for parameter, start in zip(parameters, previous, strict=True):
                        parameter.copy_(start)
        stage_records.append(
            {
                "stage": "lbfgs",
                "iterations": config.fit.lbfgs_iterations,
                "objective": float(objective("detail").detach()),
                "rejectedForInversion": rejected,
                "acceptedFraction": accepted_fraction,
            }
        )
        del optimizer

    with torch.no_grad():
        terms, final_tensor = loss_terms(True)
        final_vertices = final_tensor.cpu().numpy()
        _, scaled_tensor, scale_tensor = current_vertices(True)
        scaled_vertices = scaled_tensor.cpu().numpy()
        scale = scale_tensor.cpu().numpy()
        focal = float(torch.exp(log_focal).cpu())
        rotations_numpy = rotations.cpu().numpy()
        translations_numpy = translations.cpu().numpy()
        coefficients = low_coefficients.cpu().numpy()
        detail_values = (torch.tanh(detail_raw) * maximum_detail * observed).cpu().numpy()
        ear_parameters = ear_raw.cpu().numpy()
        final_terms = {name: float(value.cpu()) for name, value in terms.items()}

    internal_orientation = triangle_orientation_metrics(
        scaled_vertices,
        final_vertices,
        faces_numpy,
    )
    # NPZ and glTF POSITION payloads are float32. Make that representation the
    # only downstream geometry. A local collision line search then attenuates
    # displacement only around intersecting triangles; it never remeshes or
    # replaces the canonical surface.
    final_vertices, collision_finalization = collision_safe_deformation(
        scaled_vertices,
        final_vertices,
        faces_numpy,
    )
    left_eye = _deformed_eye(
        asset.left_eye,
        asset.regions["left_eyelid"],
        template_center,
        scale,
        scaled_vertices,
        final_vertices,
    )
    right_eye = _deformed_eye(
        asset.right_eye,
        asset.regions["right_eyelid"],
        template_center,
        scale,
        scaled_vertices,
        final_vertices,
    )
    left_eye, left_eye_finalization = eyelid_contact_corrected_eye(
        left_eye,
        final_vertices,
        asset.regions["left_eyelid"],
        config.anatomy.eyelid_clearance_ratio_max,
        asset.regions["left_eye"],
    )
    right_eye, right_eye_finalization = eyelid_contact_corrected_eye(
        right_eye,
        final_vertices,
        asset.regions["right_eyelid"],
        config.anatomy.eyelid_clearance_ratio_max,
        asset.regions["right_eye"],
    )
    initial_eye_finalization = {
        "left": left_eye_finalization,
        "right": right_eye_finalization,
    }
    final_vertices, left_eyelid_support = eyelid_support_corrected_vertices(
        final_vertices,
        left_eye,
        asset.regions["left_eyelid"],
        config.anatomy.eyelid_clearance_ratio_max,
    )
    final_vertices, right_eyelid_support = eyelid_support_corrected_vertices(
        final_vertices,
        right_eye,
        asset.regions["right_eyelid"],
        config.anatomy.eyelid_clearance_ratio_max,
    )
    final_vertices, post_contact_collision_finalization = collision_safe_deformation(
        scaled_vertices,
        final_vertices,
        faces_numpy,
    )
    left_eye, left_eye_finalization = eyelid_contact_corrected_eye(
        left_eye,
        final_vertices,
        asset.regions["left_eyelid"],
        config.anatomy.eyelid_clearance_ratio_max,
        asset.regions["left_eye"],
    )
    right_eye, right_eye_finalization = eyelid_contact_corrected_eye(
        right_eye,
        final_vertices,
        asset.regions["right_eyelid"],
        config.anatomy.eyelid_clearance_ratio_max,
        asset.regions["right_eye"],
    )
    orientation = triangle_orientation_metrics(scaled_vertices, final_vertices, faces_numpy)
    delivery_topology = _edge_and_component_metrics(final_vertices, faces_numpy)
    displacement = final_vertices - scaled_vertices
    maximum_displacement_ratio = float(
        np.max(np.linalg.norm(displacement, axis=1), initial=0.0) / face_width
    )
    ear_vertex_mask = np.zeros(len(final_vertices), dtype=bool)
    ear_vertex_mask[np.asarray(asset.regions["left_ear"], dtype=np.int64)] = True
    ear_vertex_mask[np.asarray(asset.regions["right_ear"], dtype=np.int64)] = True
    displacement_norm = np.linalg.norm(displacement, axis=1) / face_width
    maximum_core_displacement_ratio = float(
        np.max(displacement_norm[~ear_vertex_mask], initial=0.0)
    )
    maximum_ear_displacement_ratio = float(np.max(displacement_norm[ear_vertex_mask], initial=0.0))
    fitted_anatomy = copy.deepcopy(asset.anatomy)
    fitted_anatomy["state"] = "template-fitted"
    fitted_anatomy["fit"] = {
        "sourceComputeGeometrySha256": geometry_hash(asset.skin_vertices, asset.skin_faces),
        "sourceDeliveryGeometrySha256": asset.geometry_sha256,
        "basis": "normalized-graph-laplacian",
        "basisSize": basis_numpy.shape[1],
        "stages": stage_records,
        "collisionFinalization": collision_finalization,
        "postContactCollisionFinalization": post_contact_collision_finalization,
        "eyelidSupportFinalization": {
            "left": left_eyelid_support,
            "right": right_eyelid_support,
        },
        "initialEyeContactFinalization": initial_eye_finalization,
        "eyeContactFinalization": {
            "left": left_eye_finalization,
            "right": right_eye_finalization,
        },
        "sdfUsed": False,
    }
    fitted_hash = geometry_hash(final_vertices[asset.render_to_skin], asset.render_faces)
    fitted = UnifiedHeadAsset(
        skin_vertices=final_vertices,
        skin_faces=faces_numpy,
        render_to_skin=asset.render_to_skin,
        render_faces=asset.render_faces,
        uv=asset.uv,
        regions=asset.regions,
        left_eye=left_eye,
        right_eye=right_eye,
        geometry_sha256=fitted_hash,
        anatomy=fitted_anatomy,
    )

    cameras: list[CameraRecord] = []
    per_view: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    final_landmarks = np.einsum(
        "nvc,nv->nc",
        final_vertices[faces_numpy[binding.triangle[valid_landmarks]]],
        binding.barycentric[valid_landmarks],
    )
    feature_indices = np.asarray(
        [valid_position[index] for index in MEDIAPIPE_TO_IBUG68 if index in valid_position],
        dtype=np.int64,
    )
    for index, role in enumerate(REQUIRED_VIEWS):
        view = views[role]
        rotation_matrix, _ = cv2.Rodrigues(rotations_numpy[index])
        angles = cv2.RQDecomp3x3(rotation_matrix)[0]
        camera = CameraRecord(
            role=role,
            width=sizes[index][0],
            height=sizes[index][1],
            focal_length_px=focal,
            principal_point_px=(sizes[index][0] / 2, sizes[index][1] / 2),
            rotation_vector=tuple(float(value) for value in rotations_numpy[index]),
            translation=tuple(float(value) for value in translations_numpy[index]),
            pitch_deg=float(angles[0]),
            yaw_deg=float(angles[1]),
            roll_deg=float(angles[2]),
        )
        cameras.append(camera)
        camera_points = final_landmarks @ rotation_matrix.T + translations_numpy[index]
        projected = camera_points[:, :2] / np.clip(
            camera_points[:, 2:3], 1e-6, None
        ) * focal + np.asarray(camera.principal_point_px)
        feature_target = target_numpy[index][feature_indices]
        diagonal = max(float(np.linalg.norm(np.ptp(feature_target, axis=0))), 1.0)
        dense_nme = float(
            np.mean(np.linalg.norm(projected - target_numpy[index], axis=1)) / diagonal
        )
        feature_nme = float(
            np.mean(
                np.linalg.norm(
                    projected[feature_indices] - feature_target,
                    axis=1,
                )
            )
            / diagonal
        )
        rendered = _render_silhouette(final_vertices, faces_numpy, camera)
        full_silhouette_iou = _silhouette_iou(rendered, masks[index])
        evaluation_bottom = silhouette_bottoms[index]
        evaluated_render = rendered[:evaluation_bottom]
        evaluated_target = masks[index][:evaluation_bottom]
        silhouette_iou = _silhouette_iou(evaluated_render, evaluated_target)
        _save_silhouette_overlay(
            Path(view["normalized_path"]),
            rendered,
            run_dir / "overlays" / f"fit-silhouette-{role.value}.png",
            masks[index],
        )
        nme_limit = (
            config.acceptance.front_landmark_nme_v2_max
            if role == ViewRole.FRONT
            else config.acceptance.side_landmark_nme_v2_max
        )
        iou_limit = (
            config.acceptance.front_silhouette_iou_v2_min
            if role == ViewRole.FRONT
            else config.acceptance.side_silhouette_iou_v2_min
        )
        passed = feature_nme <= nme_limit and dense_nme <= nme_limit and silhouette_iou >= iou_limit
        per_view[role.value] = {
            "landmarkNME": feature_nme,
            "denseLandmarkNME": dense_nme,
            "landmarkErrorPx": feature_nme * diagonal,
            "silhouetteIoU": silhouette_iou,
            "fullBustSilhouetteIoU": full_silhouette_iou,
            "silhouetteEvaluationBottomPx": evaluation_bottom,
            "silhouetteEvaluationRegion": "crown-through-chin",
            "landmarkThreshold": nme_limit,
            "silhouetteThreshold": iou_limit,
            "passed": passed,
        }
        if not passed:
            failures.append({"role": role.value, **per_view[role.value]})

    if not collision_finalization["passed"]:
        failures.append(
            {
                "reason": "self-intersection-finalization-failed",
                **collision_finalization,
            }
        )
    if not post_contact_collision_finalization["passed"]:
        failures.append(
            {
                "reason": "post-contact-self-intersection-finalization-failed",
                **post_contact_collision_finalization,
            }
        )
    for side, eye_finalization in (
        ("left", left_eye_finalization),
        ("right", right_eye_finalization),
    ):
        if not eye_finalization["passed"]:
            failures.append(
                {
                    "reason": "eyelid-contact-finalization-failed",
                    "side": side,
                    **eye_finalization,
                }
            )
    if orientation["flippedTriangleCount"]:
        failures.append({"reason": "flipped-triangles", **orientation})
    if orientation["minimumSignedAreaRatio"] < _MINIMUM_SIGNED_AREA_RATIO:
        failures.append(
            {
                "reason": "triangle-area-safety-margin",
                **orientation,
                "threshold": _MINIMUM_SIGNED_AREA_RATIO,
            }
        )
    invalid_delivery_topology = {
        name: delivery_topology[name]
        for name in (
            "componentCount",
            "boundaryEdgeCount",
            "nonManifoldEdgeCount",
            "degenerateFaceCount",
            "duplicateFaceCount",
            "duplicateVertexCount",
            "watertight",
            "windingConsistent",
        )
        if delivery_topology[name]
        != {
            "componentCount": 1,
            "boundaryEdgeCount": 0,
            "nonManifoldEdgeCount": 0,
            "degenerateFaceCount": 0,
            "duplicateFaceCount": 0,
            "duplicateVertexCount": 0,
            "watertight": True,
            "windingConsistent": True,
        }[name]
    }
    if invalid_delivery_topology:
        failures.append(
            {
                "reason": "float32-delivery-topology-invalid",
                "metrics": invalid_delivery_topology,
            }
        )
    core_displacement_limit = (
        config.fit.maximum_vertex_displacement_face_width
        + config.fit.maximum_normal_offset_face_width
    ) * 1.05
    ear_displacement_limit = core_displacement_limit + 0.12
    if maximum_core_displacement_ratio > core_displacement_limit:
        failures.append(
            {
                "reason": "maximum-displacement",
                "region": "core-head",
                "measured": maximum_core_displacement_ratio,
                "threshold": core_displacement_limit,
            }
        )
    if maximum_ear_displacement_ratio > ear_displacement_limit:
        failures.append(
            {
                "reason": "maximum-displacement",
                "region": "ears",
                "measured": maximum_ear_displacement_ratio,
                "threshold": ear_displacement_limit,
            }
        )
    working = run_dir / "working"
    _save_npz(
        working / "fit.npz",
        vertices=final_vertices.astype(np.float32),
        faces=faces_numpy.astype(np.int32),
        scale=scale.astype(np.float32),
        low_frequency_coefficients=coefficients.astype(np.float32),
        detail_normal_offsets=detail_values.astype(np.float32),
        ear_parameters=ear_parameters.astype(np.float32),
        landmark_index=valid_landmarks.astype(np.int32),
        landmark_triangle=binding.triangle[valid_landmarks].astype(np.int32),
        landmark_barycentric=binding.barycentric[valid_landmarks].astype(np.float32),
        basis_eigenvalues=eigenvalues.astype(np.float32),
    )
    fitted.save(working / "unified-head.npz")
    atomic_write_json(
        working / "cameras.json",
        {
            "schemaVersion": 3,
            "cameras": [camera.model_dump(mode="json") for camera in cameras],
        },
    )
    neutral_atlas = Image.new("RGB", (512, 512), (176, 178, 182))
    fitted.export_head_glb(run_dir / "models" / "fitted-head.glb", neutral_atlas)
    metrics = {
        "schemaVersion": 3,
        "templateId": "TemplateHeadV0",
        "surfaceSource": "template-non-rigid-deformation",
        "sdfUsed": False,
        "sourceComputeGeometrySha256": geometry_hash(asset.skin_vertices, asset.skin_faces),
        "sourceDeliveryGeometrySha256": asset.geometry_sha256,
        "fittedGeometrySha256": fitted_hash,
        "vertexCount": len(final_vertices),
        "triangleCount": len(faces_numpy),
        "perView": per_view,
        "sharedFocalLengthPx": focal,
        "scale": scale.astype(float).tolist(),
        "maximumDisplacementFaceWidth": maximum_displacement_ratio,
        "maximumCoreDisplacementFaceWidth": maximum_core_displacement_ratio,
        "maximumEarDisplacementFaceWidth": maximum_ear_displacement_ratio,
        "earParameters": {
            "order": [
                "outwardTranslation",
                "verticalTranslation",
                "heightScale",
                "depthScale",
                "outwardAngle",
            ],
            "leftRaw": ear_parameters[0].astype(float).tolist(),
            "rightRaw": ear_parameters[1].astype(float).tolist(),
            "maximumAngleDeg": 25.0,
        },
        "orientation": orientation,
        "internalFloat64Orientation": internal_orientation,
        "orientationPrecision": "float32-delivery",
        "collisionFinalization": collision_finalization,
        "postContactCollisionFinalization": post_contact_collision_finalization,
        "eyelidSupportFinalization": {
            "left": left_eyelid_support,
            "right": right_eyelid_support,
        },
        "initialEyeContactFinalization": initial_eye_finalization,
        "eyeContactFinalization": {
            "left": left_eye_finalization,
            "right": right_eye_finalization,
        },
        "deliveryTopology": delivery_topology,
        "regularization": {
            "basis": "normalized-graph-laplacian",
            "basisSize": basis_numpy.shape[1],
            "laplacianWeight": config.fit.laplacian_weight,
            "arapWeight": config.fit.arap_weight,
            "inversionBarrierWeight": config.fit.inversion_barrier_weight,
            "symmetryPairCount": len(symmetry_pairs),
            "stabilityLock": stability_metrics,
        },
        "stages": stage_records,
        "lossTerms": final_terms,
        "passed": not failures,
    }
    atomic_write_json(working / "fit-metrics.json", metrics)
    if failures:
        fail(
            "template-fit-gate-failed",
            "TemplateHeadV0 三视图非刚性拟合未通过门禁",
            stage="template-fit",
            details={
                "failures": failures,
                "metrics": str(working / "fit-metrics.json"),
            },
        )
    return metrics
