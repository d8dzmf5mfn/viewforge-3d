from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import trimesh


@dataclass(slots=True)
class HeightMeshResult:
    raw_mesh: trimesh.Trimesh
    smooth_mesh: trimesh.Trimesh
    metrics: dict[str, Any]
    front_surface_depth: np.ndarray | None = None


def _filled_primary_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8) > 0
    components, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if components <= 1:
        raise ValueError("pixel mask contains no foreground")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    primary = np.where(labels == largest, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(primary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(primary)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled > 0


def _resize_fields(
    depth: np.ndarray,
    rear_depth: np.ndarray,
    mask: np.ndarray,
    feature: np.ndarray,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_height, source_width = mask.shape
    height = max(32, round(width * source_height / source_width))
    resized_depth = cv2.resize(
        depth.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC
    )
    resized_rear = cv2.resize(
        rear_depth.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC
    )
    resized_mask = cv2.resize(
        mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
    resized_feature = cv2.resize(
        feature.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
    clean_mask = _filled_primary_mask(resized_mask)
    return resized_depth, resized_rear, clean_mask, (resized_feature > 0) & clean_mask


def _top_faces(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    index = np.full((height, width), -1, dtype=np.int32)
    coordinates = np.argwhere(mask)
    index[coordinates[:, 0], coordinates[:, 1]] = np.arange(len(coordinates), dtype=np.int32)
    faces: list[tuple[int, int, int]] = []
    for row in range(height - 1):
        for column in range(width - 1):
            a = index[row, column]
            b = index[row, column + 1]
            c = index[row + 1, column]
            d = index[row + 1, column + 1]
            occupied = (a >= 0, b >= 0, d >= 0, c >= 0)
            count = sum(occupied)
            if count == 4:
                faces.extend(((int(a), int(c), int(b)), (int(b), int(c), int(d))))
            elif count == 3:
                polygon = [
                    value for value, present in zip((a, b, d, c), occupied, strict=True) if present
                ]
                faces.append((int(polygon[0]), int(polygon[2]), int(polygon[1])))
    if not faces:
        raise ValueError("pixel mask is too small to triangulate")
    return coordinates, np.asarray(faces, dtype=np.int32)


def _boundary_edges(faces: np.ndarray) -> np.ndarray:
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    return directed[counts[inverse] == 1]


def _closed_height_mesh(
    depth: np.ndarray,
    rear_depth: np.ndarray,
    mask: np.ndarray,
    *,
    pixel_step: float,
) -> trimesh.Trimesh:
    coordinates, top_faces = _top_faces(mask)
    rows = coordinates[:, 0]
    columns = coordinates[:, 1]
    center_x = (mask.shape[1] - 1) / 2
    center_y = (mask.shape[0] - 1) / 2
    top_vertices = np.column_stack(
        (
            (columns - center_x) * pixel_step,
            (center_y - rows) * pixel_step,
            depth[rows, columns],
        )
    ).astype(np.float64)
    used = np.unique(top_faces)
    remap = np.full(len(top_vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    top_vertices = top_vertices[used]
    top_faces = remap[top_faces]
    bottom_vertices = top_vertices.copy()
    bottom_vertices[:, 2] = rear_depth[rows[used], columns[used]]
    count = len(top_vertices)
    bottom_faces = top_faces[:, ::-1] + count
    boundary = _boundary_edges(top_faces)
    side_faces = np.empty((len(boundary) * 2, 3), dtype=np.int32)
    side_faces[0::2] = np.column_stack((boundary[:, 1], boundary[:, 0], boundary[:, 0] + count))
    side_faces[1::2] = np.column_stack(
        (boundary[:, 1], boundary[:, 0] + count, boundary[:, 1] + count)
    )
    mesh = trimesh.Trimesh(
        vertices=np.vstack((top_vertices, bottom_vertices)),
        faces=np.vstack((top_faces, bottom_faces, side_faces)),
        process=False,
    )
    return mesh


def rear_depth_field(
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    pixel_step: float,
    minimum_span: float | None = None,
) -> np.ndarray:
    foreground = np.asarray(mask) > 0
    values = np.asarray(depth, dtype=np.float32)
    sigma = max(values.shape[1] * 0.10, 2.0)
    numerator = cv2.GaussianBlur(values * foreground, (0, 0), sigma)
    denominator = cv2.GaussianBlur(foreground.astype(np.float32), (0, 0), sigma)
    smooth = numerator / np.maximum(denominator, 1e-5)
    distance = cv2.distanceTransform(foreground.astype(np.uint8), cv2.DIST_L2, 5)
    # OpenCV reports roughly one pixel even on the inner silhouette edge. Remove
    # that offset so the inferred rear surface actually meets the observed front
    # silhouette instead of creating a vertical wall around the head and ears.
    distance = np.maximum(distance - 1.0, 0.0)
    distance_scale = max(float(np.quantile(distance[foreground], 0.98)), 1.0)
    interior_weight = np.sqrt(np.clip(distance / distance_scale, 0.0, 1.0))
    model_width = values.shape[1] * pixel_step
    observed_span = float(np.ptp(values[foreground]))
    # A human cranium is substantially deeper than the visible facial relief.
    # The back is not observed by the three input views, so use a conservative
    # rounded template depth and mark it as inferred in the report.
    span = max(model_width * 0.68, observed_span * 1.10, minimum_span or 0.0)
    thickness = span * (0.004 + 0.996 * interior_weight)
    rear = smooth - thickness
    rear = np.minimum(rear, values - max(pixel_step * 0.6, span * 0.012))
    rear[~foreground] = float(np.min(rear[foreground]))
    return rear.astype(np.float32)


def _taubin_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    feature: np.ndarray,
    *,
    iterations: int,
    lamb: float,
    nu: float,
    maximum_delta: float,
) -> np.ndarray:
    original = depth.astype(np.float64)
    result = original.copy()
    kernel = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
    neighbour_count = cv2.filter2D(
        mask.astype(np.float64), -1, kernel, borderType=cv2.BORDER_CONSTANT
    )
    interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    mutable = interior & ~feature & (neighbour_count >= 4)

    def step(values: np.ndarray, factor: float) -> np.ndarray:
        total = cv2.filter2D(
            values * mask,
            -1,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
        )
        mean = total / np.maximum(neighbour_count, 1)
        updated = values.copy()
        updated[mutable] += factor * (mean[mutable] - values[mutable])
        return updated

    for _ in range(iterations):
        result = step(result, lamb)
        result = step(result, nu)
        result = np.clip(result, original - maximum_delta, original + maximum_delta)
        result[feature] = original[feature]
    return result.astype(np.float32)


def _normal_variance(depth: np.ndarray, mask: np.ndarray, feature: np.ndarray) -> float:
    dy, dx = np.gradient(depth.astype(np.float64))
    normals = np.dstack((-dx, dy, np.ones_like(depth, dtype=np.float64)))
    normals /= np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-12)
    local = np.dstack([cv2.GaussianBlur(normals[:, :, axis], (0, 0), 1.1) for axis in range(3)])
    interior = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    sample = interior & ~feature
    if not np.any(sample):
        return 0.0
    return float(np.mean(np.sum((normals[sample] - local[sample]) ** 2, axis=1)))


def _mesh_metrics(mesh: trimesh.Trimesh) -> dict[str, Any]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    direction = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    direction_balance = np.bincount(inverse, weights=direction, minlength=len(counts))
    areas = np.asarray(mesh.area_faces)
    edge_manifold = bool(np.all(counts == 2))
    return {
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "watertight": edge_manifold,
        "edgeManifold": edge_manifold,
        "boundaryEdges": int(np.count_nonzero(counts == 1)),
        "nonManifoldEdges": int(np.count_nonzero(counts > 2)),
        "degenerateTriangles": int(np.count_nonzero(areas <= 1e-14)),
        "finite": bool(np.all(np.isfinite(mesh.vertices)) and np.all(np.isfinite(areas))),
        "windingConsistent": bool(edge_manifold and np.all(direction_balance == 0)),
    }


def build_height_field_mesh(
    depth: np.ndarray,
    mask: np.ndarray,
    feature: np.ndarray,
    *,
    rear_depth: np.ndarray | None = None,
    target_triangles: int,
    minimum_triangles: int,
    maximum_triangles: int,
    pixel_step: float,
    taubin_iterations: int,
    taubin_lambda: float,
    taubin_nu: float,
    hausdorff_voxels_max: float,
) -> HeightMeshResult:
    depth = np.asarray(depth, dtype=np.float32)
    mask = _filled_primary_mask(mask)
    feature = (np.asarray(feature) > 0) & mask
    supplied_rear = rear_depth is not None
    if rear_depth is None:
        rear_depth = depth
    rear_depth = np.asarray(rear_depth, dtype=np.float32)
    if rear_depth.shape != depth.shape:
        raise ValueError("rear depth field must match front depth shape")
    occupancy = max(float(np.mean(mask)), 0.05)
    estimate = int(round(np.sqrt(target_triangles / (4 * occupancy))))
    width = int(np.clip(estimate, 32, depth.shape[1]))

    candidates: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    lower = max(32, width - 30)
    upper = min(depth.shape[1], width + 30)
    for candidate_width in range(lower, upper + 1, 2):
        candidate_depth, candidate_rear, candidate_mask, candidate_feature = _resize_fields(
            depth, rear_depth, mask, feature, candidate_width
        )
        _, top = _top_faces(candidate_mask)
        boundary_count = len(_boundary_edges(top))
        triangles = int(2 * len(top) + 2 * boundary_count)
        candidates.append(
            (
                candidate_width,
                candidate_depth,
                candidate_rear,
                candidate_mask,
                candidate_feature,
                triangles,
            )
        )
    in_budget = [
        candidate
        for candidate in candidates
        if minimum_triangles <= candidate[5] <= maximum_triangles
    ]
    pool = in_budget or candidates
    _, resized_depth, resized_rear, resized_mask, resized_feature, _ = min(
        pool, key=lambda candidate: abs(candidate[5] - target_triangles)
    )
    mesh_step = pixel_step * depth.shape[1] / resized_depth.shape[1]
    maximum_delta = hausdorff_voxels_max * pixel_step * 0.95
    smooth_depth = _taubin_depth(
        resized_depth,
        resized_mask,
        resized_feature,
        iterations=taubin_iterations,
        lamb=taubin_lambda,
        nu=taubin_nu,
        maximum_delta=maximum_delta,
    )
    if supplied_rear:
        minimum_gap = max(mesh_step * 0.6, 1e-4)
        raw_rear = np.minimum(resized_rear, resized_depth - minimum_gap)
        smooth_rear = cv2.GaussianBlur(resized_rear, (0, 0), 1.0)
        smooth_rear = np.minimum(smooth_rear, smooth_depth - minimum_gap)
    else:
        raw_rear = rear_depth_field(
            resized_depth,
            resized_mask,
            pixel_step=mesh_step,
        )
        smooth_rear = rear_depth_field(
            smooth_depth,
            resized_mask,
            pixel_step=mesh_step,
        )
    raw_mesh = _closed_height_mesh(
        resized_depth,
        raw_rear,
        resized_mask,
        pixel_step=mesh_step,
    )
    smooth_mesh = _closed_height_mesh(
        smooth_depth,
        smooth_rear,
        resized_mask,
        pixel_step=mesh_step,
    )
    before_variance = _normal_variance(resized_depth, resized_mask, resized_feature)
    after_variance = _normal_variance(smooth_depth, resized_mask, resized_feature)
    if before_variance <= 1e-15:
        variance_reduction = 1.0 if after_variance <= before_variance + 1e-15 else 0.0
    else:
        variance_reduction = float(np.clip(1 - after_variance / before_variance, 0, 1))
    displacement = np.abs(smooth_depth - resized_depth)
    feature_displacement = displacement[resized_feature]
    metrics = {
        **_mesh_metrics(smooth_mesh),
        "representation": "closed-smoothed-pixel-depth-field",
        "meshGrid": [int(resized_depth.shape[1]), int(resized_depth.shape[0])],
        "pixelStep": float(pixel_step),
        "meshPixelStep": float(mesh_step),
        "rearDepthRange": [
            float(smooth_rear[resized_mask].min()),
            float(smooth_rear[resized_mask].max()),
        ],
        "selfIntersection": False,
        "selfIntersectionMethod": "ordered-front-and-rear-height-fields",
        "featureDriftVoxels": float(feature_displacement.max(initial=0.0) / max(pixel_step, 1e-12)),
        "normalVarianceBefore": before_variance,
        "normalVarianceAfter": after_variance,
        "normalVarianceReduction": variance_reduction,
        "hausdorffVoxels": float(displacement.max(initial=0.0) / max(pixel_step, 1e-12)),
        "maximumSilhouetteIoUDrop": 0.0,
        "smoothing": "feature-locked-taubin-height-field",
        "simplification": "adaptive-regular-grid",
    }
    metrics["passed"] = bool(
        metrics["watertight"]
        and metrics["edgeManifold"]
        and metrics["finite"]
        and metrics["windingConsistent"]
        and metrics["boundaryEdges"] == 0
        and metrics["nonManifoldEdges"] == 0
        and metrics["degenerateTriangles"] == 0
        and minimum_triangles <= metrics["triangles"] <= maximum_triangles
        and metrics["hausdorffVoxels"] <= hausdorff_voxels_max
    )
    return HeightMeshResult(raw_mesh=raw_mesh, smooth_mesh=smooth_mesh, metrics=metrics)
