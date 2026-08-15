from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial import cKDTree
from skimage import measure

from face3d.models import CameraRecord, ViewRole


@dataclass(slots=True)
class MultiViewPixelSurface:
    positions: np.ndarray
    model_uv: np.ndarray
    source_uv: np.ndarray
    pixel_codes: np.ndarray
    feature_class: np.ndarray
    confidence: np.ndarray
    source_bits: np.ndarray
    voxel_size: float
    grid_size: tuple[int, int]
    raw_mesh: trimesh.Trimesh
    smooth_mesh: trimesh.Trimesh
    metrics: dict[str, Any]


@dataclass(slots=True)
class MultiViewPixelCells:
    """Discrete surface cells produced before any continuous-mesh operation."""

    positions: np.ndarray
    normals: np.ndarray
    model_uv: np.ndarray
    source_uv: np.ndarray
    pixel_codes: np.ndarray
    feature_class: np.ndarray
    confidence: np.ndarray
    source_bits: np.ndarray
    voxel_size: float
    grid_size: tuple[int, int]
    raw_mesh: trimesh.Trimesh
    metrics: dict[str, Any]


def foreground_mask_from_background(rgb: np.ndarray) -> np.ndarray:
    """Extract the primary foreground without using hidden geometry or depth."""
    image = np.asarray(rgb, dtype=np.uint8)
    corners = np.concatenate(
        (
            image[:24, :24].reshape(-1, 3),
            image[:24, -24:].reshape(-1, 3),
            image[-24:, :24].reshape(-1, 3),
            image[-24:, -24:].reshape(-1, 3),
        ),
        axis=0,
    )
    background = np.median(corners.astype(np.float32), axis=0)
    distance = np.linalg.norm(image.astype(np.float32) - background, axis=2)
    binary = np.where(distance > 12.0, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if components <= 1:
        raise ValueError("reference image contains no separable foreground")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    primary = np.where(labels == label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(primary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(primary)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def photometric_feature_relief(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    maximum_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate local depth only inside eyes, nose, mouth, and ear image regions."""
    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    foreground = mask > 127
    rows, columns = np.nonzero(foreground)
    if not len(rows):
        raise ValueError("front mask contains no foreground")
    y0, y1 = int(rows.min()), int(rows.max())
    x0, x1 = int(columns.min()), int(columns.max())
    mask_width = max(x1 - x0 + 1, 1)
    mask_height = max(y1 - y0 + 1, 1)
    head_bottom = min(y1 + 1, y0 + round(mask_height * 0.74))
    row_widths: list[int] = []
    row_centers: list[float] = []
    for row in range(y0, head_bottom):
        row_columns = np.flatnonzero(foreground[row])
        if len(row_columns):
            row_widths.append(int(row_columns[-1] - row_columns[0] + 1))
            row_centers.append(float((row_columns[0] + row_columns[-1]) / 2))
    face_width = float(np.quantile(row_widths, 0.70)) if row_widths else float(mask_width)
    face_width = float(np.clip(face_width, mask_width * 0.45, mask_width))
    face_center_x = float(np.median(row_centers)) if row_centers else float((x0 + x1) / 2)
    face_height = max(head_bottom - y0, 1)
    yy, xx = np.indices(foreground.shape, dtype=np.float32)
    normalized_x = (xx - (face_center_x - face_width / 2)) / face_width
    normalized_y = (yy - y0) / face_height
    feature = np.zeros_like(mask, dtype=np.uint8)
    regions = (
        (
            1,
            (
                ((normalized_x >= 0.14) & (normalized_x <= 0.46))
                | ((normalized_x >= 0.54) & (normalized_x <= 0.86))
            )
            & (normalized_y >= 0.25)
            & (normalized_y <= 0.43),
        ),
        (
            2,
            (normalized_x >= 0.37)
            & (normalized_x <= 0.63)
            & (normalized_y >= 0.32)
            & (normalized_y <= 0.62),
        ),
        (
            3,
            (normalized_x >= 0.25)
            & (normalized_x <= 0.75)
            & (normalized_y >= 0.58)
            & (normalized_y <= 0.75),
        ),
        (
            4,
            (
                ((normalized_x >= -0.08) & (normalized_x <= 0.14))
                | ((normalized_x >= 0.86) & (normalized_x <= 1.08))
            )
            & (normalized_y >= 0.30)
            & (normalized_y <= 0.63),
        ),
    )
    smoothed = cv2.GaussianBlur(gray, (0, 0), 3.5)
    low_frequency = cv2.GaussianBlur(smoothed, (0, 0), max(face_height / 24, 5.0))
    local = smoothed - low_frequency
    gradient_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    detail_energy = np.abs(local) + cv2.magnitude(gradient_x, gradient_y) * 0.18
    detail_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for feature_class, region in regions:
        candidate = foreground & region
        values = detail_energy[candidate]
        if not len(values):
            continue
        threshold = float(np.quantile(values, 0.52))
        active = candidate & (detail_energy >= threshold)
        active = cv2.dilate(active.astype(np.uint8), detail_kernel, iterations=1) > 0
        feature[candidate & active] = feature_class
    selected = feature > 0
    scale = max(float(np.quantile(np.abs(local[selected]), 0.92)), 1.0) if np.any(selected) else 1.0
    feature_weight = cv2.GaussianBlur(selected.astype(np.float32), (0, 0), 4.0)
    relief = np.clip(local / scale, -1, 1) * feature_weight * maximum_offset
    relief[~selected] = 0
    return feature, relief.astype(np.float32)


def _project(points: np.ndarray, camera: CameraRecord) -> tuple[np.ndarray, np.ndarray]:
    rotation, _ = cv2.Rodrigues(np.asarray(camera.rotation_vector, dtype=np.float64))
    camera_points = np.asarray(points, dtype=np.float64) @ rotation.T
    camera_points += np.asarray(camera.translation, dtype=np.float64)
    depth = camera_points[:, 2]
    valid = depth > 1e-6
    pixels = np.zeros((len(points), 2), dtype=np.float64)
    pixels[valid, 0] = (
        camera_points[valid, 0] / depth[valid] * camera.focal_length_px
        + camera.principal_point_px[0]
    )
    pixels[valid, 1] = (
        camera_points[valid, 1] / depth[valid] * camera.focal_length_px
        + camera.principal_point_px[1]
    )
    return pixels, valid


def _pixels_on_world_plane(pixels: np.ndarray, camera: CameraRecord) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(camera.rotation_vector, dtype=np.float64))
    x = (pixels[:, 0] - camera.principal_point_px[0]) / camera.focal_length_px
    y = (pixels[:, 1] - camera.principal_point_px[1]) / camera.focal_length_px
    rays = np.column_stack((x, y, np.ones(len(pixels), dtype=np.float64)))
    translation = np.asarray(camera.translation, dtype=np.float64)
    numerator = float(translation @ rotation[:, 2])
    denominator = rays @ rotation[:, 2]
    scale = numerator / np.where(np.abs(denominator) > 1e-8, denominator, 1e-8)
    camera_points = rays * scale[:, None]
    return (camera_points - translation) @ rotation


def _volume_bounds(
    mask: np.ndarray,
    camera: CameraRecord,
    *,
    depth_to_front_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.nonzero(mask > 127)
    if not len(rows):
        raise ValueError("front mask contains no foreground")
    corners = np.asarray(
        [
            [columns.min(), rows.min()],
            [columns.max(), rows.min()],
            [columns.max(), rows.max()],
            [columns.min(), rows.max()],
        ],
        dtype=np.float64,
    )
    world = _pixels_on_world_plane(corners, camera)
    lower_xy = world[:, :2].min(axis=0)
    upper_xy = world[:, :2].max(axis=0)
    span_xy = upper_xy - lower_xy
    padding = span_xy * 0.025
    lower_xy -= padding
    upper_xy += padding
    if not np.isfinite(depth_to_front_width) or depth_to_front_width <= 0:
        raise ValueError("depth_to_front_width must be finite and positive")
    depth_span = max(
        float(span_xy[0] * depth_to_front_width),
        float(span_xy[1] * 0.45),
    )
    lower = np.asarray([lower_xy[0], lower_xy[1], -depth_span / 2], dtype=np.float64)
    upper = np.asarray([upper_xy[0], upper_xy[1], depth_span / 2], dtype=np.float64)
    return lower, upper


def _carve_visual_hull(
    masks: dict[ViewRole, np.ndarray],
    cameras: dict[ViewRole, CameraRecord],
    *,
    resolution: int,
    depth_to_front_width: float,
    additional_constraints: tuple[tuple[np.ndarray, CameraRecord], ...] = (),
) -> tuple[np.ndarray, np.ndarray, float]:
    lower, upper = _volume_bounds(
        masks[ViewRole.FRONT],
        cameras[ViewRole.FRONT],
        depth_to_front_width=depth_to_front_width,
    )
    voxel_size = float(np.max(upper - lower) / max(resolution - 1, 1))
    counts = np.maximum(np.ceil((upper - lower) / voxel_size).astype(int) + 1, 8)
    axes = [
        lower[axis] + np.arange(counts[axis], dtype=np.float64) * voxel_size for axis in range(3)
    ]
    occupancy = np.ones(tuple(int(value) for value in counts), dtype=bool)
    constraints = tuple((masks[role], cameras[role]) for role in ViewRole)
    constraints += additional_constraints
    yz = np.stack(np.meshgrid(axes[1], axes[2], indexing="ij"), axis=-1).reshape(-1, 2)
    for x_index, x_value in enumerate(axes[0]):
        points = np.column_stack(
            (
                np.full(len(yz), x_value, dtype=np.float64),
                yz[:, 0],
                yz[:, 1],
            )
        )
        inside = np.ones(len(points), dtype=bool)
        for constraint_mask, constraint_camera in constraints:
            pixels, valid = _project(points, constraint_camera)
            coordinates = np.rint(pixels).astype(np.int32)
            height, width = constraint_mask.shape
            valid &= (
                (coordinates[:, 0] >= 0)
                & (coordinates[:, 0] < width)
                & (coordinates[:, 1] >= 0)
                & (coordinates[:, 1] < height)
            )
            supported = np.zeros(len(points), dtype=bool)
            supported[valid] = (
                constraint_mask[coordinates[valid, 1], coordinates[valid, 0]] > 127
            )
            inside &= supported
        occupancy[x_index] = inside.reshape(counts[1], counts[2])
    if not np.any(occupancy):
        raise ValueError("three-view silhouette intersection is empty")
    origin = np.asarray([axis[0] for axis in axes], dtype=np.float64)
    return occupancy, origin, voxel_size


def _surface_mask(occupancy: np.ndarray) -> np.ndarray:
    interior = occupancy.copy()
    for axis in range(3):
        forward = np.zeros_like(occupancy)
        backward = np.zeros_like(occupancy)
        source_forward = [slice(None)] * 3
        target_forward = [slice(None)] * 3
        source_forward[axis] = slice(1, None)
        target_forward[axis] = slice(None, -1)
        forward[tuple(target_forward)] = occupancy[tuple(source_forward)]
        source_backward = [slice(None)] * 3
        target_backward = [slice(None)] * 3
        source_backward[axis] = slice(None, -1)
        target_backward[axis] = slice(1, None)
        backward[tuple(target_backward)] = occupancy[tuple(source_backward)]
        interior &= forward & backward
    return occupancy & ~interior


def _surface_normals(occupancy: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Estimate outward cell normals without constructing or smoothing a mesh."""
    density = gaussian_filter(occupancy.astype(np.float32), sigma=0.85, mode="constant")
    gradients = np.gradient(density)
    normals = -np.column_stack(
        tuple(component[tuple(indices.T)] for component in gradients)
    ).astype(np.float32)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    fallback = lengths[:, 0] < 1e-6
    if np.any(fallback):
        center = np.mean(indices.astype(np.float32), axis=0)
        normals[fallback] = indices[fallback].astype(np.float32) - center
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-8)
    return normals


def _point_surface_normals(
    positions: np.ndarray,
    reference_normals: np.ndarray,
) -> np.ndarray:
    """Re-estimate normals after local Pixel depth displacement; geometry is unchanged."""
    neighbor_count = min(12, len(positions))
    _, neighbors = cKDTree(positions).query(positions, k=neighbor_count)
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
    neighborhoods = positions[neighbors]
    centered = neighborhoods - np.mean(neighborhoods, axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered, optimize=True)
    _, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0].astype(np.float32)
    invalid = ~np.all(np.isfinite(normals), axis=1)
    normals[invalid] = reference_normals[invalid]
    orientation = np.sum(normals * reference_normals, axis=1) < 0
    normals[orientation] *= -1
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    return normals


def _isolated_surface_cell_count(occupancy: np.ndarray, surface: np.ndarray) -> int:
    neighbors = np.zeros(occupancy.shape, dtype=np.uint8)
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(None, -1)
        upper[axis] = slice(1, None)
        neighbors[tuple(lower)] += occupancy[tuple(upper)]
        neighbors[tuple(upper)] += occupancy[tuple(lower)]
    return int(np.count_nonzero(surface & (neighbors == 0)))


def _front_projection_metrics(
    positions: np.ndarray,
    front_mask: np.ndarray,
    camera: CameraRecord,
    *,
    maximum_grid_size: int,
) -> dict[str, Any]:
    height, width = front_mask.shape
    scale = maximum_grid_size / max(height, width)
    grid_width = max(8, int(round(width * scale)))
    grid_height = max(8, int(round(height * scale)))
    target = cv2.resize(
        (front_mask > 127).astype(np.uint8),
        (grid_width, grid_height),
        interpolation=cv2.INTER_NEAREST,
    )
    pixels, valid = _project(positions, camera)
    projected = np.rint(pixels[valid] * scale).astype(np.int32)
    valid_projected = (
        (projected[:, 0] >= 0)
        & (projected[:, 0] < grid_width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < grid_height)
    )
    projected = projected[valid_projected]
    raster = np.zeros((grid_height, grid_width), dtype=np.uint8)
    raster[projected[:, 1], projected[:, 0]] = 1
    # A cell centre represents one finite pixel footprint. A one-cell dilation
    # measures that footprint without inventing a continuous surface.
    footprint = cv2.dilate(raster, np.ones((3, 3), dtype=np.uint8), iterations=1)
    foreground_count = int(np.count_nonzero(target))
    covered = int(np.count_nonzero((target > 0) & (footprint > 0)))
    distance = cv2.distanceTransform((1 - raster).astype(np.uint8), cv2.DIST_L2, 5)
    maximum_distance = float(np.max(distance[target > 0], initial=0.0))
    return {
        "gridSize": [grid_width, grid_height],
        "foregroundPixelCount": foreground_count,
        "coveredPixelCount": covered,
        "coverage": covered / max(foreground_count, 1),
        "maximumDistancePixels": maximum_distance,
    }


def _direct_front_pixel_cells(
    positions: np.ndarray,
    normals: np.ndarray,
    front_mask: np.ndarray,
    camera: CameraRecord,
    *,
    maximum_grid_size: int,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create one traceable front cell for every foreground model pixel."""
    height, width = front_mask.shape
    scale = maximum_grid_size / max(height, width)
    grid_width = max(8, int(round(width * scale)))
    grid_height = max(8, int(round(height * scale)))
    target = cv2.resize(
        (front_mask > 127).astype(np.uint8),
        (grid_width, grid_height),
        interpolation=cv2.INTER_NEAREST,
    )
    rows, columns = np.nonzero(target)
    rotation, _ = cv2.Rodrigues(np.asarray(camera.rotation_vector, dtype=np.float64))
    camera_positions = positions @ rotation.T + np.asarray(camera.translation)
    depth = camera_positions[:, 2]
    projected, valid = _project(positions, camera)
    projected_grid = projected[valid] * scale
    valid_indices = np.flatnonzero(valid)
    target_grid = np.column_stack((columns, rows)).astype(np.float64)
    neighbor_count = min(12, len(projected_grid))
    _, neighbors = cKDTree(projected_grid).query(target_grid, k=neighbor_count)
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
    candidate_indices = valid_indices[neighbors]
    candidate_depths = depth[candidate_indices]
    selected = candidate_indices[
        np.arange(len(candidate_indices)), np.argmin(candidate_depths, axis=1)
    ]
    selected_depth = depth[selected]

    source_x = (columns.astype(np.float64) + 0.5) / scale
    source_y = (rows.astype(np.float64) + 0.5) / scale
    camera_x = (
        (source_x - camera.principal_point_px[0])
        / camera.focal_length_px
        * selected_depth
    )
    camera_y = (
        (source_y - camera.principal_point_px[1])
        / camera.focal_length_px
        * selected_depth
    )
    camera_points = np.column_stack((camera_x, camera_y, selected_depth))
    direct_positions = (camera_points - np.asarray(camera.translation)) @ rotation
    direct_normals = normals[selected]
    attachment = cKDTree(positions).query(direct_positions, k=1)[0]
    return (
        direct_positions.astype(np.float32),
        direct_normals.astype(np.float32),
        {
            "gridSize": [grid_width, grid_height],
            "cellCount": int(len(direct_positions)),
            "attachmentMaximumVoxels": float(
                np.max(attachment, initial=0.0) / max(voxel_size, 1e-8)
            ),
            "attachmentP99Voxels": float(
                np.quantile(attachment, 0.99) / max(voxel_size, 1e-8)
            ),
        },
    )


def _regularize_depth_cross_sections(occupancy: np.ndarray) -> np.ndarray:
    """Turn each 2D-supported row envelope into a smooth oval depth section."""
    regularized = occupancy.copy()
    x_grid, z_grid = np.indices((occupancy.shape[0], occupancy.shape[2]), dtype=np.float64)
    bounds = np.full((occupancy.shape[1], 4), np.nan, dtype=np.float64)
    for y_index in range(occupancy.shape[1]):
        section = occupancy[:, y_index, :]
        coordinates = np.argwhere(section)
        if not len(coordinates):
            continue
        x_min, z_min = coordinates.min(axis=0)
        x_max, z_max = coordinates.max(axis=0)
        bounds[y_index] = (x_min, x_max, z_min, z_max)
    valid_rows = np.flatnonzero(np.all(np.isfinite(bounds), axis=1))
    if not len(valid_rows):
        raise ValueError("regularized three-view silhouette volume is empty")
    rows = np.arange(occupancy.shape[1], dtype=np.float64)
    for column in range(bounds.shape[1]):
        bounds[:, column] = np.interp(
            rows,
            valid_rows,
            bounds[valid_rows, column],
        )
    bounds[:, :2] = gaussian_filter1d(bounds[:, :2], sigma=1.2, axis=0)
    bounds[:, 2:] = gaussian_filter1d(bounds[:, 2:], sigma=5.0, axis=0)
    for y_index in valid_rows:
        x_min, x_max, z_min, z_max = bounds[y_index]
        radius_x = max(float(x_max - x_min) / 2 + 0.75, 1.0)
        radius_z = max(float(z_max - z_min) / 2 + 0.75, 1.0)
        center_x = float(x_min + x_max) / 2
        center_z = float(z_min + z_max) / 2
        ellipse = ((x_grid - center_x) / radius_x) ** 2 + (
            (z_grid - center_z) / radius_z
        ) ** 2 <= 1.0
        regularized[:, y_index, :] &= ellipse
    if not np.any(regularized):
        raise ValueError("regularized three-view silhouette volume is empty")
    return regularized


_FACE_DEFINITIONS = (
    ((1, 0, 0), ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1))),
    ((-1, 0, 0), ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1))),
    ((0, 1, 0), ((-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1))),
    ((0, -1, 0), ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1))),
    ((0, 0, 1), ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))),
    ((0, 0, -1), ((-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1))),
)


def _voxel_surface_mesh(
    occupancy: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
) -> trimesh.Trimesh:
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    vertex_indices: dict[tuple[int, int, int], int] = {}
    shape = np.asarray(occupancy.shape, dtype=np.int32)
    for cell in np.argwhere(occupancy):
        for direction, corners in _FACE_DEFINITIONS:
            neighbor = cell + np.asarray(direction, dtype=np.int32)
            if np.all((neighbor >= 0) & (neighbor < shape)) and occupancy[tuple(neighbor)]:
                continue
            quad: list[int] = []
            for corner in corners:
                key = tuple((cell * 2 + np.asarray(corner, dtype=np.int32)).tolist())
                index = vertex_indices.get(key)
                if index is None:
                    index = len(vertices)
                    vertex_indices[key] = index
                    vertices.append(origin + np.asarray(key, dtype=np.float64) * voxel_size / 2)
                quad.append(index)
            faces.extend(((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])))
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int32),
        process=False,
    )
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals(multibody=True)
    return mesh


def _marching_cubes_mesh(
    occupancy: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
) -> trimesh.Trimesh:
    padded = np.pad(occupancy.astype(np.float32), 1, mode="constant")
    scalar_field = gaussian_filter(padded, sigma=0.80, mode="constant")
    vertices, faces, _, _ = measure.marching_cubes(
        scalar_field,
        level=0.5,
        spacing=(voxel_size, voxel_size, voxel_size),
        method="lewiner",
        allow_degenerate=False,
    )
    vertices += origin - voxel_size
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals(multibody=True)
    return mesh


def _apply_relief(
    vertices: np.ndarray,
    camera: CameraRecord,
    feature_map: np.ndarray | None,
    relief_map: np.ndarray | None,
    *,
    depth_tolerance: float,
) -> tuple[np.ndarray, int]:
    if feature_map is None or relief_map is None:
        return vertices, 0
    result = np.asarray(vertices, dtype=np.float64).copy()
    rotation, _ = cv2.Rodrigues(np.asarray(camera.rotation_vector, dtype=np.float64))
    camera_vertices = result @ rotation.T
    camera_vertices += np.asarray(camera.translation, dtype=np.float64)
    pixels, valid = _project(result, camera)
    coordinates = np.rint(pixels).astype(np.int32)
    height, width = feature_map.shape
    valid &= (
        (coordinates[:, 0] >= 0)
        & (coordinates[:, 0] < width)
        & (coordinates[:, 1] >= 0)
        & (coordinates[:, 1] < height)
    )
    selected = np.zeros(len(result), dtype=bool)
    selected[valid] = feature_map[coordinates[valid, 1], coordinates[valid, 0]] > 0
    flat_pixels = coordinates[:, 1] * width + coordinates[:, 0]
    nearest_depth = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(nearest_depth, flat_pixels[valid], camera_vertices[valid, 2])
    visible = np.zeros(len(result), dtype=bool)
    visible[valid] = camera_vertices[valid, 2] <= (
        nearest_depth[flat_pixels[valid]] + depth_tolerance
    )
    front_depth_limit = float(np.quantile(camera_vertices[:, 2], 0.40))
    selected &= visible & (camera_vertices[:, 2] <= front_depth_limit)
    offsets = relief_map[coordinates[selected, 1], coordinates[selected, 0]]
    camera_forward_world = rotation[2]
    result[selected] -= offsets[:, None] * camera_forward_world[None, :]
    return result, int(np.count_nonzero(selected))


def _trace_surface_pixels(
    positions: np.ndarray,
    images: dict[ViewRole, np.ndarray],
    cameras: dict[ViewRole, CameraRecord],
    feature_map: np.ndarray | None,
    *,
    model_grid_size: tuple[int, int],
    inferred_roles: frozenset[ViewRole],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    projected = {role: _project(positions, cameras[role])[0] for role in ViewRole}
    center = positions.mean(axis=0)
    primary = np.full(len(positions), ViewRole.FRONT.value, dtype=object)
    rear = positions[:, 2] < center[2]
    primary[rear & (positions[:, 0] < center[0])] = ViewRole.LEFT45.value
    primary[rear & (positions[:, 0] >= center[0])] = ViewRole.RIGHT45.value
    source_uv = np.zeros((len(positions), 2), dtype=np.uint16)
    colors = np.zeros((len(positions), 3), dtype=np.uint32)
    for role in ViewRole:
        selected = primary == role.value
        coordinates = np.rint(projected[role][selected]).astype(np.int32)
        height, width = images[role].shape[:2]
        coordinates[:, 0] = np.clip(coordinates[:, 0], 0, width - 1)
        coordinates[:, 1] = np.clip(coordinates[:, 1], 0, height - 1)
        source_uv[selected] = coordinates.astype(np.uint16)
        colors[selected] = images[role][coordinates[:, 1], coordinates[:, 0]].astype(np.uint32)
    pixel_codes = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
    front_pixels = np.rint(projected[ViewRole.FRONT]).astype(np.int32)
    front_height, front_width = images[ViewRole.FRONT].shape[:2]
    front_pixels[:, 0] = np.clip(front_pixels[:, 0], 0, front_width - 1)
    front_pixels[:, 1] = np.clip(front_pixels[:, 1], 0, front_height - 1)
    model_uv = np.column_stack(
        (
            front_pixels[:, 0] * (model_grid_size[0] - 1) // max(front_width - 1, 1),
            front_pixels[:, 1] * (model_grid_size[1] - 1) // max(front_height - 1, 1),
        )
    ).astype(np.uint16)
    feature = np.zeros(len(positions), dtype=np.uint8)
    if feature_map is not None:
        feature = feature_map[front_pixels[:, 1], front_pixels[:, 0]].astype(np.uint8)
    confidence = np.where(rear, 0.82, 0.94).astype(np.float32)
    confidence[feature > 0] = 0.96
    # Visual-hull cells are constrained by all three silhouettes. Bit 8 marks
    # that at least one supporting view was inferred rather than independently
    # observed; source_uv still points to the primary raster chosen above.
    source_value = 1 | 2 | 4 | (8 if inferred_roles else 0)
    source_bits = np.full(len(positions), source_value, dtype=np.uint8)
    if inferred_roles:
        confidence = np.minimum(confidence, 0.52)
    return model_uv, source_uv, pixel_codes, feature, confidence, source_bits


def pixel_cells_from_occupancy(
    occupancy: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    images: dict[ViewRole, np.ndarray],
    cameras: dict[ViewRole, CameraRecord],
    front_evidence_mask: np.ndarray,
    *,
    model_grid_size: int,
    maximum_cells: int = 150_000,
    feature_map: np.ndarray | None = None,
    relief_map: np.ndarray | None = None,
    inferred_roles: frozenset[ViewRole] = frozenset(),
    representation: str = "pixel-derived-occupancy-surface",
    metric_details: dict[str, Any] | None = None,
) -> MultiViewPixelCells:
    """Convert a pixel-derived occupancy grid into traceable surface cells only."""
    occupancy = np.asarray(occupancy, dtype=bool)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    if occupancy.ndim != 3 or not np.any(occupancy):
        raise ValueError("occupancy must be a non-empty 3D grid")
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and positive")
    missing = [role.value for role in ViewRole if role not in images or role not in cameras]
    if missing:
        raise ValueError(f"missing calibrated views: {missing}")

    surface = _surface_mask(occupancy)
    surface_indices = np.argwhere(surface)
    positions = origin + surface_indices.astype(np.float64) * voxel_size
    normals = _surface_normals(occupancy, surface_indices)
    positions, refined_cell_count = _apply_relief(
        positions,
        cameras[ViewRole.FRONT],
        feature_map,
        relief_map,
        depth_tolerance=voxel_size * 1.5,
    )
    occupancy_boundary_count = int(len(positions))
    direct_positions, direct_normals, direct_front = _direct_front_pixel_cells(
        positions,
        normals,
        front_evidence_mask,
        cameras[ViewRole.FRONT],
        maximum_grid_size=model_grid_size,
        voxel_size=voxel_size,
    )
    positions = np.vstack((positions, direct_positions)).astype(np.float32)
    normals = np.vstack((normals, direct_normals)).astype(np.float32)
    normals = _point_surface_normals(positions, normals)
    if len(positions) > maximum_cells:
        raise ValueError(f"surface cell count {len(positions)} exceeds limit {maximum_cells}")

    raw_mesh = _voxel_surface_mesh(occupancy, origin, voxel_size)
    front_projection = _front_projection_metrics(
        positions,
        front_evidence_mask,
        cameras[ViewRole.FRONT],
        maximum_grid_size=model_grid_size,
    )
    grid_size = tuple(int(value) for value in front_projection["gridSize"])
    model_uv, source_uv, pixel_codes, feature, confidence, source_bits = _trace_surface_pixels(
        positions,
        images,
        cameras,
        feature_map,
        model_grid_size=grid_size,
        inferred_roles=inferred_roles,
    )
    _, component_count = measure.label(occupancy, connectivity=1, return_num=True)
    metrics = {
        "representation": representation,
        "volumeResolution": [int(value) for value in occupancy.shape],
        "occupiedVoxels": int(np.count_nonzero(occupancy)),
        "surfaceCells": int(len(positions)),
        "occupancyBoundaryCells": occupancy_boundary_count,
        "directFront": direct_front,
        "refinedCells": refined_cell_count,
        "normalsReestimatedAfterRelief": True,
        "isolatedVoxelCount": _isolated_surface_cell_count(occupancy, surface),
        "componentCount": int(component_count),
        "surfaceCellCoverage": 1.0,
        "frontProjection": front_projection,
        "frontSurfaceMaxDistancePixels": front_projection["maximumDistancePixels"],
        "traceabilityComplete": bool(
            len(model_uv) == len(positions) and np.all(source_bits > 0)
        ),
        "rawPixelMeshWatertight": bool(raw_mesh.is_watertight),
        "finite": bool(
            np.all(np.isfinite(positions))
            and np.all(np.isfinite(normals))
            and np.all(np.isfinite(confidence))
        ),
        "smoothingApplied": False,
        "continuousMeshGenerated": False,
        **(metric_details or {}),
    }
    return MultiViewPixelCells(
        positions=positions,
        normals=normals,
        model_uv=model_uv,
        source_uv=source_uv,
        pixel_codes=pixel_codes.astype(np.uint32),
        feature_class=feature,
        confidence=confidence,
        source_bits=source_bits,
        voxel_size=float(voxel_size),
        grid_size=grid_size,
        raw_mesh=raw_mesh,
        metrics=metrics,
    )


def pixel_cells_from_front_grid_and_occupancy(
    occupancy: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    images: dict[ViewRole, np.ndarray],
    cameras: dict[ViewRole, CameraRecord],
    front_evidence_mask: np.ndarray,
    front_positions: np.ndarray,
    front_normals: np.ndarray,
    front_grid_mask: np.ndarray,
    front_source_uv: np.ndarray,
    front_rgb: np.ndarray,
    *,
    maximum_cells: int = 150_000,
    feature_map: np.ndarray | None = None,
    inferred_roles: frozenset[ViewRole] = frozenset(),
    inferred_cell_positions: np.ndarray | None = None,
    inferred_cell_normals: np.ndarray | None = None,
    front_duplicate_distance_voxels: float = 0.45,
    front_duplicate_normal_z_minimum: float | None = None,
    representation: str = "direct-front-grid-plus-pixel-occupancy-shell",
    metric_details: dict[str, Any] | None = None,
) -> MultiViewPixelCells:
    """Join exact front-image cells to a discrete inferred occupancy shell.

    The observed front grid is never snapped to a mesh or resampled from an
    isosurface. Occupancy boundary cells are retained only for lateral and rear
    support, so this function remains a Pixel-stage operation.
    """
    occupancy = np.asarray(occupancy, dtype=bool)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    grid_mask = np.asarray(front_grid_mask) > 127
    if occupancy.ndim != 3 or not np.any(occupancy):
        raise ValueError("occupancy must be a non-empty 3D grid")
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and positive")
    if (
        not np.isfinite(front_duplicate_distance_voxels)
        or front_duplicate_distance_voxels <= 0
    ):
        raise ValueError("front_duplicate_distance_voxels must be finite and positive")
    if front_duplicate_normal_z_minimum is not None and (
        not np.isfinite(front_duplicate_normal_z_minimum)
        or not -1.0 <= front_duplicate_normal_z_minimum <= 1.0
    ):
        raise ValueError("front_duplicate_normal_z_minimum must be between -1 and 1")
    if front_positions.shape != (*grid_mask.shape, 3):
        raise ValueError("front_positions must match front_grid_mask")
    if front_normals.shape != front_positions.shape:
        raise ValueError("front_normals must match front_positions")
    if front_source_uv.shape != (*grid_mask.shape, 2):
        raise ValueError("front_source_uv must match front_grid_mask")
    if front_rgb.shape != (*grid_mask.shape, 3):
        raise ValueError("front_rgb must match front_grid_mask")
    if feature_map is not None and feature_map.shape != grid_mask.shape:
        raise ValueError("feature_map must match front_grid_mask")
    missing = [role.value for role in ViewRole if role not in images or role not in cameras]
    if missing:
        raise ValueError(f"missing calibrated views: {missing}")

    surface = _surface_mask(occupancy)
    surface_indices = np.argwhere(surface)
    occupancy_surface_positions = origin + surface_indices.astype(np.float64) * voxel_size
    if (inferred_cell_positions is None) != (inferred_cell_normals is None):
        raise ValueError("inferred cell positions and normals must be supplied together")
    if inferred_cell_positions is None:
        inferred_positions = occupancy_surface_positions
        inferred_normals = _surface_normals(occupancy, surface_indices)
        inferred_cell_source = "occupancy-boundary"
    else:
        inferred_positions = np.asarray(inferred_cell_positions, dtype=np.float64)
        inferred_normals = np.asarray(inferred_cell_normals, dtype=np.float64)
        if inferred_positions.ndim != 2 or inferred_positions.shape[1] != 3:
            raise ValueError("inferred_cell_positions must be an N x 3 array")
        if inferred_normals.shape != inferred_positions.shape:
            raise ValueError("inferred_cell_normals must match inferred_cell_positions")
        if not np.all(np.isfinite(inferred_positions)) or not np.all(np.isfinite(inferred_normals)):
            raise ValueError("inferred cells must be finite")
        inferred_cell_source = "provided-discrete-scan-shell"

    rows, columns = np.nonzero(grid_mask)
    observed_positions = np.asarray(front_positions[rows, columns], dtype=np.float64)
    observed_normals = np.asarray(front_normals[rows, columns], dtype=np.float64)
    if not len(observed_positions) or not np.all(np.isfinite(observed_positions)):
        raise ValueError("front grid contains no finite observed cells")

    coverage_fallback_cell_count = 0
    if inferred_cell_positions is not None:
        preliminary_positions = np.vstack((observed_positions, inferred_positions))
        preliminary_distance = cKDTree(preliminary_positions).query(
            occupancy_surface_positions,
            k=1,
        )[0]
        uncovered = preliminary_distance > voxel_size * 1.55
        if np.any(uncovered):
            occupancy_normals = _surface_normals(occupancy, surface_indices)
            inferred_positions = np.vstack(
                (inferred_positions, occupancy_surface_positions[uncovered])
            )
            inferred_normals = np.vstack(
                (inferred_normals, occupancy_normals[uncovered])
            )
            coverage_fallback_cell_count = int(np.count_nonzero(uncovered))

    # Remove only the occupancy layer that would duplicate the directly scanned
    # front surface. Deep lateral/rear boundary cells remain untouched.
    # Retain an overlap band so direct front tiles still cover the surface at
    # oblique views. Remove only genuinely coincident cells; broad XY-only
    # deduplication creates a visible and measurable front/lateral gap.
    duplicate_distance = cKDTree(observed_positions).query(inferred_positions, k=1)[0]
    duplicate_front = duplicate_distance <= (
        voxel_size * front_duplicate_distance_voxels
    )
    if front_duplicate_normal_z_minimum is not None:
        inferred_normal_lengths = np.maximum(
            np.linalg.norm(inferred_normals, axis=1),
            1e-8,
        )
        inferred_normal_z = inferred_normals[:, 2] / inferred_normal_lengths
        duplicate_front &= inferred_normal_z >= front_duplicate_normal_z_minimum
    inferred_positions = inferred_positions[~duplicate_front]
    inferred_normals = inferred_normals[~duplicate_front]

    inferred_trace = _trace_surface_pixels(
        inferred_positions,
        images,
        cameras,
        None,
        model_grid_size=(grid_mask.shape[1], grid_mask.shape[0]),
        inferred_roles=inferred_roles,
    )
    _, inferred_source_uv, inferred_codes, _, inferred_confidence, _ = inferred_trace

    source_uv = np.vstack(
        (
            np.asarray(front_source_uv[rows, columns], dtype=np.uint16),
            inferred_source_uv,
        )
    )
    observed_colors = np.asarray(front_rgb[rows, columns], dtype=np.uint32)
    observed_codes = (
        (observed_colors[:, 0] << 16)
        | (observed_colors[:, 1] << 8)
        | observed_colors[:, 2]
    )
    pixel_codes = np.concatenate((observed_codes, inferred_codes)).astype(np.uint32)
    positions = np.vstack((observed_positions, inferred_positions)).astype(np.float32)
    normals = np.vstack((observed_normals, inferred_normals)).astype(np.float32)
    normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(normal_lengths, 1e-8)

    observed_model_uv = np.column_stack((columns, rows)).astype(np.uint16)
    u_axis = np.median(front_source_uv[..., 0], axis=0).astype(np.float64)
    v_axis = np.median(front_source_uv[..., 1], axis=1).astype(np.float64)
    inferred_model_u = np.interp(
        inferred_source_uv[:, 0],
        u_axis,
        np.arange(grid_mask.shape[1], dtype=np.float64),
    )
    inferred_model_v = np.interp(
        inferred_source_uv[:, 1],
        v_axis,
        np.arange(grid_mask.shape[0], dtype=np.float64),
    )
    inferred_model_uv = np.rint(
        np.column_stack((inferred_model_u, inferred_model_v))
    ).astype(np.uint16)
    model_uv = np.vstack((observed_model_uv, inferred_model_uv))

    observed_feature = (
        np.zeros(len(observed_positions), dtype=np.uint8)
        if feature_map is None
        else np.asarray(feature_map[rows, columns], dtype=np.uint8)
    )
    feature = np.concatenate(
        (observed_feature, np.zeros(len(inferred_positions), dtype=np.uint8))
    )
    observed_confidence = np.full(len(observed_positions), 0.98, dtype=np.float32)
    confidence = np.concatenate(
        (observed_confidence, np.minimum(inferred_confidence, 0.52).astype(np.float32))
    )
    observed_source_value = 1 | (2 | 4 | 8 if inferred_roles else 0)
    source_bits = np.concatenate(
        (
            np.full(len(observed_positions), observed_source_value, dtype=np.uint8),
            np.full(
                len(inferred_positions),
                (2 | 4 | 8) if inferred_roles else 1,
                dtype=np.uint8,
            ),
        )
    )
    if len(positions) > maximum_cells:
        raise ValueError(f"surface cell count {len(positions)} exceeds limit {maximum_cells}")

    raw_mesh = _voxel_surface_mesh(occupancy, origin, voxel_size)
    front_projection = _front_projection_metrics(
        positions,
        front_evidence_mask,
        cameras[ViewRole.FRONT],
        maximum_grid_size=max(grid_mask.shape),
    )
    attachment = cKDTree(occupancy_surface_positions).query(
        observed_positions,
        k=1,
    )[0]
    coverage_distance = cKDTree(positions).query(occupancy_surface_positions, k=1)[0]
    surface_cell_coverage = float(
        np.mean(coverage_distance <= voxel_size * 1.80)
    )
    _, component_count = measure.label(occupancy, connectivity=1, return_num=True)
    metrics = {
        "representation": representation,
        "volumeResolution": [int(value) for value in occupancy.shape],
        "occupiedVoxels": int(np.count_nonzero(occupancy)),
        "surfaceCells": int(len(positions)),
        "occupancyBoundaryCells": int(len(surface_indices)),
        "frontMeasuredCells": int(len(observed_positions)),
        "inferredSurfaceCells": int(len(inferred_positions)),
        "inferredCellSource": inferred_cell_source,
        "coverageFallbackCellCount": coverage_fallback_cell_count,
        "removedDuplicateFrontBoundaryCells": int(np.count_nonzero(duplicate_front)),
        "frontDuplicateDistanceVoxels": float(front_duplicate_distance_voxels),
        "frontDuplicateNormalZMinimum": (
            None
            if front_duplicate_normal_z_minimum is None
            else float(front_duplicate_normal_z_minimum)
        ),
        "directFront": {
            "gridSize": [int(grid_mask.shape[1]), int(grid_mask.shape[0])],
            "cellCount": int(len(observed_positions)),
            "attachmentMaximumVoxels": float(
                np.max(attachment, initial=0.0) / max(voxel_size, 1e-8)
            ),
            "attachmentP99Voxels": float(
                np.quantile(attachment, 0.99) / max(voxel_size, 1e-8)
            ),
        },
        "isolatedVoxelCount": _isolated_surface_cell_count(occupancy, surface),
        "componentCount": int(component_count),
        "surfaceCellCoverage": surface_cell_coverage,
        "frontProjection": front_projection,
        "frontSurfaceMaxDistancePixels": front_projection["maximumDistancePixels"],
        "traceabilityComplete": bool(
            len(model_uv) == len(positions) and np.all(source_bits > 0)
        ),
        "rawPixelMeshWatertight": bool(raw_mesh.is_watertight),
        "finite": bool(
            np.all(np.isfinite(positions))
            and np.all(np.isfinite(normals))
            and np.all(np.isfinite(confidence))
        ),
        "frontGridResampledFromMesh": False,
        "smoothingApplied": False,
        "continuousMeshGenerated": False,
        **(metric_details or {}),
    }
    return MultiViewPixelCells(
        positions=positions,
        normals=normals,
        model_uv=model_uv,
        source_uv=source_uv,
        pixel_codes=pixel_codes,
        feature_class=feature,
        confidence=confidence,
        source_bits=source_bits,
        voxel_size=float(voxel_size),
        grid_size=(int(grid_mask.shape[1]), int(grid_mask.shape[0])),
        raw_mesh=raw_mesh,
        metrics=metrics,
    )


def _reconstruct_multiview_pixel_cells(
    images: dict[ViewRole, np.ndarray],
    masks: dict[ViewRole, np.ndarray],
    cameras: dict[ViewRole, CameraRecord],
    *,
    resolution: int = 96,
    maximum_cells: int = 150_000,
    feature_map: np.ndarray | None = None,
    relief_map: np.ndarray | None = None,
    depth_to_front_width: float = 0.78,
    inferred_roles: frozenset[ViewRole] = frozenset(),
    additional_constraints: tuple[tuple[np.ndarray, CameraRecord], ...] = (),
    direct_front_mask: np.ndarray | None = None,
) -> tuple[MultiViewPixelCells, np.ndarray, np.ndarray]:
    """Build the discrete Pixel result and retain internals for an optional later stage."""
    missing = [role.value for role in ViewRole if role not in images or role not in masks]
    if missing:
        raise ValueError(f"missing calibrated views: {missing}")
    occupancy, origin, voxel_size = _carve_visual_hull(
        masks,
        cameras,
        resolution=resolution,
        depth_to_front_width=depth_to_front_width,
        additional_constraints=additional_constraints,
    )
    occupancy = _regularize_depth_cross_sections(occupancy)
    front_evidence_mask = (
        masks[ViewRole.FRONT] if direct_front_mask is None else direct_front_mask
    )
    if front_evidence_mask.shape != masks[ViewRole.FRONT].shape:
        raise ValueError("direct_front_mask must match the calibrated front mask shape")
    cells = pixel_cells_from_occupancy(
        occupancy,
        origin,
        voxel_size,
        images,
        cameras,
        front_evidence_mask,
        model_grid_size=resolution,
        maximum_cells=maximum_cells,
        feature_map=feature_map,
        relief_map=relief_map,
        inferred_roles=inferred_roles,
        representation="three-view-2d-pixel-occupancy-surface",
        metric_details={
            "depthSearchToFrontWidth": float(depth_to_front_width),
            "silhouetteConstraintCount": 3 + len(additional_constraints),
        },
    )
    return cells, occupancy, origin


def reconstruct_multiview_pixel_cells(
    images: dict[ViewRole, np.ndarray],
    masks: dict[ViewRole, np.ndarray],
    cameras: dict[ViewRole, CameraRecord],
    *,
    resolution: int = 96,
    maximum_cells: int = 150_000,
    feature_map: np.ndarray | None = None,
    relief_map: np.ndarray | None = None,
    depth_to_front_width: float = 0.78,
    inferred_roles: frozenset[ViewRole] = frozenset(),
    additional_constraints: tuple[tuple[np.ndarray, CameraRecord], ...] = (),
    direct_front_mask: np.ndarray | None = None,
) -> MultiViewPixelCells:
    """Reconstruct traceable 3D Pixel cells without any smooth-mesh operation."""
    cells, _, _ = _reconstruct_multiview_pixel_cells(
        images,
        masks,
        cameras,
        resolution=resolution,
        maximum_cells=maximum_cells,
        feature_map=feature_map,
        relief_map=relief_map,
        depth_to_front_width=depth_to_front_width,
        inferred_roles=inferred_roles,
        additional_constraints=additional_constraints,
        direct_front_mask=direct_front_mask,
    )
    return cells


def reconstruct_multiview_pixel_surface(
    images: dict[ViewRole, np.ndarray],
    masks: dict[ViewRole, np.ndarray],
    cameras: dict[ViewRole, CameraRecord],
    *,
    resolution: int = 96,
    maximum_cells: int = 150_000,
    target_triangles: int = 80_000,
    feature_map: np.ndarray | None = None,
    relief_map: np.ndarray | None = None,
    depth_to_front_width: float = 0.78,
    inferred_roles: frozenset[ViewRole] = frozenset(),
    additional_constraints: tuple[tuple[np.ndarray, CameraRecord], ...] = (),
    direct_front_mask: np.ndarray | None = None,
) -> MultiViewPixelSurface:
    """Run the optional continuous stage after the discrete Pixel result exists."""
    cells, occupancy, origin = _reconstruct_multiview_pixel_cells(
        images,
        masks,
        cameras,
        resolution=resolution,
        maximum_cells=maximum_cells,
        feature_map=feature_map,
        relief_map=relief_map,
        depth_to_front_width=depth_to_front_width,
        inferred_roles=inferred_roles,
        additional_constraints=additional_constraints,
        direct_front_mask=direct_front_mask,
    )
    smooth_mesh = _marching_cubes_mesh(occupancy, origin, cells.voxel_size)
    trimesh.smoothing.filter_taubin(
        smooth_mesh,
        lamb=0.43,
        nu=0.50,
        iterations=28,
    )
    smooth_vertices, refined_vertex_count = _apply_relief(
        np.asarray(smooth_mesh.vertices),
        cameras[ViewRole.FRONT],
        feature_map,
        relief_map,
        depth_tolerance=cells.voxel_size * 1.5,
    )
    smooth_mesh.vertices = smooth_vertices
    if len(smooth_mesh.faces) > target_triangles:
        smooth_mesh = smooth_mesh.simplify_quadric_decimation(face_count=target_triangles)
    smooth_mesh.remove_unreferenced_vertices()
    smooth_mesh.fix_normals(multibody=True)
    metrics = {
        **cells.metrics,
        "representation": "three-view-2d-silhouette-depth-envelope",
        "refinedVertices": refined_vertex_count,
        "vertices": int(len(smooth_mesh.vertices)),
        "triangles": int(len(smooth_mesh.faces)),
        "watertight": bool(smooth_mesh.is_watertight),
        "edgeManifold": bool(smooth_mesh.is_watertight),
        "boundaryEdges": 0 if smooth_mesh.is_watertight else -1,
        "finite": bool(np.all(np.isfinite(smooth_mesh.vertices))),
        "smoothingApplied": True,
        "continuousMeshGenerated": True,
    }
    return MultiViewPixelSurface(
        positions=cells.positions,
        model_uv=cells.model_uv,
        source_uv=cells.source_uv,
        pixel_codes=cells.pixel_codes,
        feature_class=cells.feature_class,
        confidence=cells.confidence,
        source_bits=cells.source_bits,
        voxel_size=cells.voxel_size,
        grid_size=cells.grid_size,
        raw_mesh=cells.raw_mesh,
        smooth_mesh=smooth_mesh,
        metrics=metrics,
    )
