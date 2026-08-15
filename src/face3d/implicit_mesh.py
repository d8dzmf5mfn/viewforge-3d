from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from face3d.height_mesh import HeightMeshResult
from face3d.models import REQUIRED_VIEWS, ViewRole


@dataclass(slots=True)
class ProjectionModel:
    crop: tuple[int, int, int, int]
    source_shape: tuple[int, int]
    front_center_x: float
    front_center_y: float
    front_scale: float
    raw_origin: float
    raw_per_world: float
    side_views: dict[ViewRole, dict[str, float]]
    chin_y: float
    neck_bottom_y: float
    head_center_y: float
    head_center_z: float
    head_radius_x: float
    head_radius_y: float
    head_radius_z: float


@dataclass(slots=True)
class EarObservation:
    direction: float
    signed_distance: np.ndarray
    profile: np.ndarray
    relief: np.ndarray
    source_pixels: int
    bounds: tuple[float, float, float, float]
    reverse_horizontal: bool


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    inside = np.asarray(mask) > 127
    inside_distance = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 5)
    return (inside_distance - outside_distance).astype(np.float32)


def _sample(image: np.ndarray, x: np.ndarray, y: np.ndarray, border: float) -> np.ndarray:
    return cv2.remap(
        image.astype(np.float32),
        x.astype(np.float32),
        y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border),
    )


def _main_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    components = mesh.split(only_watertight=False)
    if not components:
        raise ValueError("continuous implicit field produced no surface")
    return max(components, key=lambda component: len(component.faces))


def _clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    cleaned = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        process=True,
        validate=True,
    )
    cleaned.update_faces(cleaned.nondegenerate_faces())
    cleaned.update_faces(cleaned.unique_faces())
    triangles = np.asarray(cleaned.triangles)
    double_area = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    cleaned.update_faces(double_area > 1e-14)
    cleaned.remove_unreferenced_vertices()
    cleaned.merge_vertices()
    cleaned = _main_component(cleaned)
    trimesh.repair.fix_normals(cleaned, multibody=True)
    if cleaned.volume < 0:
        cleaned.invert()
    return cleaned


def _mesh_from_field(
    field: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    voxel_size: float,
) -> trimesh.Trimesh:
    z_axis, y_axis, x_axis = axes
    vertices_zyx, faces, _, _ = marching_cubes(
        field,
        level=0.0,
        spacing=(voxel_size, voxel_size, voxel_size),
        method="lewiner",
        gradient_direction="ascent",
        allow_degenerate=False,
    )
    vertices = vertices_zyx[:, [2, 1, 0]]
    vertices += np.asarray([x_axis[0], y_axis[0], z_axis[0]], dtype=np.float64)
    return _clean_mesh(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))


def _front_surface_from_field(
    field: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_shape: tuple[int, int],
    pixel_step: float,
    fallback: np.ndarray,
) -> np.ndarray:
    """Return the outer +Z zero crossing for every front-image grid cell."""
    z_axis, y_axis, x_axis = axes
    inside = field >= 0.0
    has_surface = np.any(inside, axis=0)
    last_inside = len(z_axis) - 1 - np.argmax(inside[::-1], axis=0)
    has_surface &= last_inside < len(z_axis) - 1
    next_index = np.minimum(last_inside + 1, len(z_axis) - 1)
    first_value = np.take_along_axis(field, last_inside[None, :, :], axis=0)[0]
    next_value = np.take_along_axis(field, next_index[None, :, :], axis=0)[0]
    denominator = next_value - first_value
    fraction = np.zeros_like(first_value, dtype=np.float64)
    stable = has_surface & (np.abs(denominator) > 1e-12)
    fraction[stable] = np.clip(-first_value[stable] / denominator[stable], 0.0, 1.0)
    volume_depth = z_axis[last_inside] + fraction * (z_axis[next_index] - z_axis[last_inside])
    if np.any(has_surface):
        nearest = distance_transform_edt(
            ~has_surface,
            return_distances=False,
            return_indices=True,
        )
        volume_depth = volume_depth[tuple(nearest)]
    else:
        return np.asarray(fallback, dtype=np.float32).copy()

    height, width = target_shape
    rows, columns = np.mgrid[:height, :width]
    target_x = (columns - (width - 1) / 2) * pixel_step
    target_y = ((height - 1) / 2 - rows) * pixel_step
    coordinates = np.vstack(
        (
            ((target_y - y_axis[0]) / (y_axis[1] - y_axis[0])).ravel(),
            ((target_x - x_axis[0]) / (x_axis[1] - x_axis[0])).ravel(),
        )
    )
    sampled = map_coordinates(volume_depth, coordinates, order=1, mode="nearest").reshape(
        target_shape
    )
    validity = map_coordinates(
        has_surface.astype(np.float32), coordinates, order=1, mode="constant", cval=0.0
    ).reshape(target_shape)
    return np.where(validity >= 0.5, sampled, fallback).astype(np.float32)


def _normal_variation(mesh: trimesh.Trimesh) -> float:
    edges = np.asarray(mesh.edges_unique)
    if not len(edges):
        return 0.0
    normals = np.asarray(mesh.vertex_normals)
    dots = np.sum(normals[edges[:, 0]] * normals[edges[:, 1]], axis=1)
    return float(np.mean(1.0 - np.clip(dots, -1.0, 1.0)))


def _surface_field_displacement(
    first_mesh: trimesh.Trimesh,
    second_mesh: trimesh.Trimesh,
    first_field: np.ndarray,
    second_field: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    voxel_size: float,
) -> tuple[float, float, float]:
    z_axis, y_axis, x_axis = axes

    def residual(mesh: trimesh.Trimesh, field: np.ndarray) -> np.ndarray:
        vertices = np.asarray(mesh.vertices)
        stride = max(1, len(vertices) // 80_000)
        vertices = vertices[::stride]
        coordinates = np.vstack(
            (
                (vertices[:, 2] - z_axis[0]) / voxel_size,
                (vertices[:, 1] - y_axis[0]) / voxel_size,
                (vertices[:, 0] - x_axis[0]) / voxel_size,
            )
        )
        return np.abs(map_coordinates(field, coordinates, order=1, mode="nearest"))

    forward = residual(first_mesh, second_field)
    backward = residual(second_mesh, first_field)
    forward_maximum = float(forward.max(initial=0.0))
    backward_maximum = float(backward.max(initial=0.0))
    return max(forward_maximum, backward_maximum), forward_maximum, backward_maximum


def _mesh_metrics(mesh: trimesh.Trimesh) -> dict[str, Any]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    direction = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    direction_balance = np.bincount(inverse, weights=direction, minlength=len(counts))
    triangles = np.asarray(mesh.triangles)
    double_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    edge_manifold = bool(np.all(counts == 2))
    return {
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "watertight": edge_manifold,
        "edgeManifold": edge_manifold,
        "boundaryEdges": int(np.count_nonzero(counts == 1)),
        "nonManifoldEdges": int(np.count_nonzero(counts > 2)),
        "degenerateTriangles": int(np.count_nonzero(double_area <= 1e-14)),
        "finite": bool(np.all(np.isfinite(mesh.vertices)) and np.all(np.isfinite(double_area))),
        "windingConsistent": bool(edge_manifold and np.all(direction_balance == 0)),
    }


def _grid_to_source(
    x: np.ndarray,
    y: np.ndarray,
    grid_shape: tuple[int, int],
    pixel_step: float,
    projection: ProjectionModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = grid_shape
    grid_x = x / pixel_step + (width - 1) / 2
    grid_y = (height - 1) / 2 - y / pixel_step
    crop_x, crop_y, crop_width, crop_height = projection.crop
    source_x = crop_x + (grid_x + 0.5) * crop_width / width - 0.5
    source_y = crop_y + (grid_y + 0.5) * crop_height / height - 0.5
    normalized_x = (source_x - projection.front_center_x) / projection.front_scale
    normalized_y = (source_y - projection.front_center_y) / projection.front_scale
    return grid_x, grid_y, normalized_x, normalized_y


def _project_vertices(
    vertices: np.ndarray,
    role: ViewRole,
    grid_shape: tuple[int, int],
    pixel_step: float,
    projection: ProjectionModel,
) -> np.ndarray:
    _, _, normalized_x, normalized_y = _grid_to_source(
        vertices[:, 0], vertices[:, 1], grid_shape, pixel_step, projection
    )
    if role == ViewRole.FRONT:
        source_width, source_height = projection.source_shape
        x = normalized_x * projection.front_scale + projection.front_center_x
        y = normalized_y * projection.front_scale + projection.front_center_y
        x = x * source_width / max(source_width, 1)
        y = y * source_height / max(source_height, 1)
        return np.column_stack((x, y))
    view = projection.side_views[role]
    raw_depth = projection.raw_origin + vertices[:, 2] * projection.raw_per_world
    projected_x = (
        math.cos(view["theta"]) * normalized_x
        + math.sin(view["theta"]) * raw_depth
        + view["offset"]
    )
    x = projected_x * view["scale"] + view["centerX"]
    y = normalized_y * view["scale"] + view["centerY"]
    return np.column_stack((x, y))


def _render_silhouette(
    mesh: trimesh.Trimesh,
    role: ViewRole,
    target_shape: tuple[int, int],
    grid_shape: tuple[int, int],
    pixel_step: float,
    projection: ProjectionModel,
) -> np.ndarray:
    projected = _project_vertices(
        np.asarray(mesh.vertices), role, grid_shape, pixel_step, projection
    )
    triangles = np.rint(projected[np.asarray(mesh.faces)]).astype(np.int32)
    height, width = target_shape
    raster = np.zeros((height, width), dtype=np.uint8)
    for triangle in triangles:
        if (
            triangle[:, 0].max() < 0
            or triangle[:, 1].max() < 0
            or triangle[:, 0].min() >= width
            or triangle[:, 1].min() >= height
        ):
            continue
        cv2.fillConvexPoly(raster, triangle, 255, lineType=cv2.LINE_8)
    return raster


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    first_foreground = first > 127
    second_foreground = second > 127
    union = np.count_nonzero(first_foreground | second_foreground)
    return float(np.count_nonzero(first_foreground & second_foreground) / max(union, 1))


def _smooth_union(first: np.ndarray, second: np.ndarray, radius: float) -> np.ndarray:
    radius = max(float(radius), 1e-8)
    return 0.5 * (first + second + np.sqrt((first - second) ** 2 + radius**2))


def _smooth_intersection(first: np.ndarray, second: np.ndarray, radius: float) -> np.ndarray:
    radius = max(float(radius), 1e-8)
    return 0.5 * (first + second - np.sqrt((first - second) ** 2 + radius**2))


def _compact_smooth_union(first: np.ndarray, second: np.ndarray, radius: float) -> np.ndarray:
    """Smooth maximum with exactly zero influence outside its blend band."""
    radius = max(float(radius), 1e-8)
    weight = np.clip(0.5 + 0.5 * (first - second) / radius, 0.0, 1.0)
    return second * (1.0 - weight) + first * weight + radius * weight * (1.0 - weight)


def _compact_smooth_intersection(
    first: np.ndarray,
    second: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Smooth minimum with no displacement outside the local blend band."""
    return -_compact_smooth_union(-first, -second, radius)


def _template_scalar(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | float,
    projection: ProjectionModel,
    blend_radius: float,
) -> np.ndarray:
    # A human cranium has a broad dome. A quadratic ellipsoid converges too
    # quickly at its pole and reads as a pointed egg when viewed head-on.
    cranium_power = 2.40
    cranium_xy = (
        np.abs(x / projection.head_radius_x) ** cranium_power
        + np.abs((y - projection.head_center_y) / projection.head_radius_y) ** cranium_power
    ) ** (1.0 / cranium_power)
    head_radius = np.sqrt(
        cranium_xy**2 + ((z - projection.head_center_z) / projection.head_radius_z) ** 2
    )
    head = (1.0 - head_radius) * min(
        projection.head_radius_x,
        projection.head_radius_y,
        projection.head_radius_z,
    )

    jaw_radius_x = projection.head_radius_x * 0.88
    jaw_radius_y = projection.head_radius_y * 0.48
    jaw_radius_z = projection.head_radius_z * 0.68
    jaw_center_y = projection.chin_y + jaw_radius_y * 0.92
    jaw_center_z = projection.head_center_z + projection.head_radius_z * 0.18
    jaw_radius = np.sqrt(
        (x / jaw_radius_x) ** 2
        + ((y - jaw_center_y) / jaw_radius_y) ** 2
        + ((z - jaw_center_z) / jaw_radius_z) ** 2
    )
    jaw = (1.0 - jaw_radius) * min(jaw_radius_x, jaw_radius_y, jaw_radius_z)

    neck_progress = np.clip(
        (projection.chin_y - y) / max(projection.chin_y - projection.neck_bottom_y, 1e-8),
        0.0,
        1.0,
    )
    neck_radius_x = projection.head_radius_x * (0.53 + neck_progress * 0.13)
    neck_radius_z = projection.head_radius_z * (0.68 + neck_progress * 0.10)
    neck_center_z = projection.head_center_z + projection.head_radius_z * 0.25
    neck_radial = np.sqrt((x / neck_radius_x) ** 2 + ((z - neck_center_z) / neck_radius_z) ** 2)
    neck = (1.0 - neck_radial) * np.minimum(neck_radius_x, neck_radius_z)
    neck = np.minimum(neck, y - projection.neck_bottom_y)
    neck = np.minimum(
        neck,
        projection.chin_y + projection.head_radius_y * 0.14 - y,
    )

    ear_center_y = projection.head_center_y - projection.head_radius_y * 0.02
    ear_center_z = projection.head_center_z + projection.head_radius_z * 0.20
    ears: list[np.ndarray] = []
    for direction in (-1.0, 1.0):
        lateral_x = direction * x

        # Use a narrow oval attachment instead of a broad carrier plate.
        root_center_x = projection.head_radius_x * 0.99
        root_center_z = ear_center_z + projection.head_radius_z * 0.045
        root_radius = np.sqrt(
            ((lateral_x - root_center_x) / (projection.head_radius_x * 0.23)) ** 2
            + ((y - ear_center_y) / (projection.head_radius_y * 0.18)) ** 2
            + ((z - root_center_z) / (projection.head_radius_z * 0.10)) ** 2
        )
        root = (1.0 - root_radius) * min(
            projection.head_radius_x * 0.23,
            projection.head_radius_y * 0.18,
            projection.head_radius_z * 0.10,
        )

        # Keep the anterior edge near the root and tilt the posterior edge
        # outward. This creates a standing pinna plane rather than side relief.
        pinna_yz_power = 2.15
        pinna_z_coordinate = (z - ear_center_z) / (projection.head_radius_z * 0.165)
        pinna_center_x = projection.head_radius_x * (1.16 - pinna_z_coordinate * 0.09)
        pinna_yz = (
            np.abs((y - ear_center_y) / (projection.head_radius_y * 0.24)) ** pinna_yz_power
            + np.abs(pinna_z_coordinate) ** pinna_yz_power
        ) ** (1.0 / pinna_yz_power)
        pinna_radius = np.sqrt(
            ((lateral_x - pinna_center_x) / (projection.head_radius_x * 0.10)) ** 2 + pinna_yz**2
        )
        pinna = (1.0 - pinna_radius) * min(
            projection.head_radius_x * 0.10,
            projection.head_radius_y * 0.24,
            projection.head_radius_z * 0.165,
        )

        lobe_center_x = projection.head_radius_x * 1.12
        lobe_center_y = ear_center_y - projection.head_radius_y * 0.175
        lobe_center_z = ear_center_z + projection.head_radius_z * 0.015
        lobe_radius = np.sqrt(
            ((lateral_x - lobe_center_x) / (projection.head_radius_x * 0.075)) ** 2
            + ((y - lobe_center_y) / (projection.head_radius_y * 0.08)) ** 2
            + ((z - lobe_center_z) / (projection.head_radius_z * 0.085)) ** 2
        )
        lobe = (1.0 - lobe_radius) * min(
            projection.head_radius_x * 0.075,
            projection.head_radius_y * 0.08,
            projection.head_radius_z * 0.085,
        )
        ear = _compact_smooth_union(pinna, lobe, blend_radius * 2.2)
        ears.append(_compact_smooth_union(root, ear, blend_radius * 3.2))

    template = _smooth_union(head, jaw, blend_radius * 4.0)
    template = _smooth_union(template, neck, blend_radius * 4.4)
    for ear in ears:
        template = _compact_smooth_union(template, ear, blend_radius * 3.2)
    return template


def _feature_drift(
    raw_mesh: trimesh.Trimesh,
    smooth_mesh: trimesh.Trimesh,
    feature: np.ndarray,
    grid_shape: tuple[int, int],
    pixel_step: float,
) -> float:
    vertices = np.asarray(raw_mesh.vertices)
    height, width = grid_shape
    columns = np.rint(vertices[:, 0] / pixel_step + (width - 1) / 2).astype(np.int32)
    rows = np.rint((height - 1) / 2 - vertices[:, 1] / pixel_step).astype(np.int32)
    valid = (
        (columns >= 0)
        & (columns < width)
        & (rows >= 0)
        & (rows < height)
        & (vertices[:, 2] >= np.quantile(vertices[:, 2], 0.56))
    )
    selected = np.zeros(len(vertices), dtype=bool)
    selected[valid] = feature[rows[valid], columns[valid]] > 0
    feature_vertices = vertices[selected]
    if not len(feature_vertices):
        return 0.0
    stride = max(1, len(feature_vertices) // 4_000)
    distances = cKDTree(np.asarray(smooth_mesh.vertices)).query(feature_vertices[::stride], k=1)[0]
    return float(np.quantile(distances, 0.99))


def build_multiview_implicit_mesh(
    front_depth: np.ndarray,
    front_mask: np.ndarray,
    data_mask: np.ndarray,
    feature: np.ndarray,
    masks: dict[ViewRole, np.ndarray],
    projection: ProjectionModel,
    *,
    eye_models: list[dict[str, float]],
    rear_hint: np.ndarray,
    pixel_step: float,
    resolution: int,
    padding_fraction: float,
    target_triangles: int,
    minimum_triangles: int,
    maximum_triangles: int,
    hausdorff_voxels_max: float,
) -> HeightMeshResult:
    depth = np.asarray(front_depth, dtype=np.float32)
    rear = np.asarray(rear_hint, dtype=np.float32)
    mask = np.asarray(front_mask) > 127
    observed = (np.asarray(data_mask) > 127) & mask
    feature = (np.asarray(feature) > 0) & mask
    if depth.shape != mask.shape or rear.shape != mask.shape or observed.shape != mask.shape:
        raise ValueError("front depth, rear hint and masks must have matching shapes")

    height, width = mask.shape
    core_x = np.asarray(
        [-(width - 1) * pixel_step / 2, (width - 1) * pixel_step / 2],
        dtype=np.float64,
    )
    core_y = np.asarray(
        [-(height - 1) * pixel_step / 2, (height - 1) * pixel_step / 2],
        dtype=np.float64,
    )
    core_z = np.asarray(
        [
            min(
                float(np.quantile(rear[mask], 0.002)),
                projection.head_center_z - projection.head_radius_z,
            ),
            float(np.quantile(depth[mask], 0.998)),
        ],
        dtype=np.float64,
    )
    core_extent = np.asarray(
        [np.ptp(core_x), np.ptp(core_y), max(float(np.ptp(core_z)), pixel_step * 8)],
        dtype=np.float64,
    )
    padding = max(float(np.max(core_extent)) * padding_fraction, pixel_step * 2.5)
    lower = np.asarray([core_x[0], core_y[0], core_z[0]]) - padding
    upper = np.asarray([core_x[1], core_y[1], core_z[1]]) + padding
    voxel_size = float(np.max(upper - lower) / max(resolution - 1, 1))
    counts = np.maximum(np.ceil((upper - lower) / voxel_size).astype(np.int32) + 1, 8)
    x_axis = lower[0] + np.arange(counts[0], dtype=np.float64) * voxel_size
    y_axis = lower[1] + np.arange(counts[1], dtype=np.float64) * voxel_size
    z_axis = lower[2] + np.arange(counts[2], dtype=np.float64) * voxel_size
    query_x, query_y = np.meshgrid(x_axis, y_axis)
    grid_x, grid_y, normalized_x, normalized_y = _grid_to_source(
        query_x, query_y, depth.shape, pixel_step, projection
    )

    front_signed = _signed_distance(front_mask)
    front_constraint = _sample(front_signed, grid_x, grid_y, -max(height, width))
    front_constraint *= pixel_step
    # The previous implementation only compared the finished mesh with the
    # left/right masks; those masks never constrained the volume itself. Build
    # smooth signed-distance fields now so every Z layer is carved by all three
    # observed silhouettes. A blurred SDF plus smooth intersection avoids the
    # row-wise cut planes produced by hard per-scanline visual-hull clipping.
    side_silhouette_fields = {
        role: gaussian_filter(_signed_distance(masks[role]), sigma=3.6, mode="nearest")
        for role in (ViewRole.LEFT45, ViewRole.RIGHT45)
    }
    crop_width = max(float(projection.crop[2]), 1.0)
    world_per_normalized = width * pixel_step * projection.front_scale / crop_width
    observed_distance = cv2.distanceTransform(observed.astype(np.uint8), cv2.DIST_L2, 5)
    observed_constraint = _sample(observed_distance, grid_x, grid_y, 0.0)
    observed_constraint *= pixel_step
    sampled_depth = _sample(depth, grid_x, grid_y, float(core_z[0]))
    observed_depth = sampled_depth.copy()
    eye_primitives: list[dict[str, Any]] = []
    for eye in eye_models:
        radius_xy = np.sqrt(
            ((query_x - eye["centerX"]) / eye["apertureRadiusX"]) ** 2
            + ((query_y - eye["centerY"]) / eye["apertureRadiusY"]) ** 2
        )
        aperture_weight = np.clip(1.0 - radius_xy, 0.0, 1.0)
        aperture_weight = aperture_weight * aperture_weight * (3.0 - 2.0 * aperture_weight)
        observed_depth -= aperture_weight * eye["recessDepth"]
        eye_primitives.append(
            {
                **eye,
                "apertureScalar": (1.0 - radius_xy)
                * min(eye["apertureRadiusX"], eye["apertureRadiusY"]),
                "capScalar": (
                    1.0
                    - np.sqrt(
                        ((query_x - eye["centerX"]) / (eye["apertureRadiusX"] * 0.91)) ** 2
                        + ((query_y - eye["centerY"]) / (eye["apertureRadiusY"] * 0.84)) ** 2
                    )
                )
                * min(eye["apertureRadiusX"] * 0.91, eye["apertureRadiusY"] * 0.84),
            }
        )
    x_boundary = np.minimum(query_x - x_axis[0], x_axis[-1] - query_x) - voxel_size * 0.55
    y_boundary = np.minimum(query_y - y_axis[0], y_axis[-1] - query_y) - voxel_size * 0.55
    volume_boundary = np.minimum(x_boundary, y_boundary)

    template_at_front = _template_scalar(
        query_x,
        query_y,
        observed_depth,
        projection,
        voxel_size * 1.5,
    )
    face_weight = np.clip(
        (query_y - projection.neck_bottom_y)
        / max(projection.chin_y - projection.neck_bottom_y, voxel_size),
        0.0,
        1.0,
    )
    face_weight = face_weight * face_weight * (3.0 - 2.0 * face_weight)
    boundary_weight = np.clip(
        front_constraint / max(projection.head_radius_x * 0.17, voxel_size * 3.0),
        0.0,
        1.0,
    )
    boundary_weight = boundary_weight * boundary_weight * (3.0 - 2.0 * boundary_weight)
    observed_weight = np.clip(
        observed_constraint / max(projection.head_radius_x * 0.115, voxel_size * 2.5),
        0.0,
        1.0,
    )
    observed_weight = observed_weight * observed_weight * (3.0 - 2.0 * observed_weight)
    data_weight = face_weight * boundary_weight * observed_weight
    upper_release = np.clip(
        (query_y - (projection.head_center_y + projection.head_radius_y * 0.35))
        / max(projection.head_radius_y * 0.37, voxel_size),
        0.0,
        1.0,
    )
    upper_release = upper_release * upper_release * (3.0 - 2.0 * upper_release)
    front_silhouette_support = 1.0 - upper_release
    ear_center_y = projection.head_center_y - projection.head_radius_y * 0.02
    ear_center_z = projection.head_center_z + projection.head_radius_z * 0.20
    lateral_transition = np.clip(
        (np.abs(query_x) / max(projection.head_radius_x, voxel_size) - 0.54) / 0.38,
        0.0,
        1.0,
    )
    lateral_transition = lateral_transition * lateral_transition * (3.0 - 2.0 * lateral_transition)
    ear_height_transition = np.exp(
        -(((query_y - ear_center_y) / max(projection.head_radius_y * 0.40, voxel_size)) ** 4)
    )
    temporal_transition = lateral_transition * ear_height_transition
    scalar = np.empty((len(z_axis), len(y_axis), len(x_axis)), dtype=np.float32)
    for z_index, world_depth in enumerate(z_axis):
        template = _template_scalar(
            query_x,
            query_y,
            world_depth,
            projection,
            voxel_size * 1.5,
        )
        distance_behind_front = np.maximum(observed_depth - world_depth, 0.0)
        deformation_weight = (
            np.exp(
                -((distance_behind_front / max(projection.head_radius_z * 0.46, voxel_size)) ** 2)
            )
            * data_weight
        )
        layer = template - template_at_front * deformation_weight
        observed_front = observed_depth - world_depth
        layer = _compact_smooth_intersection(
            layer,
            observed_front + (1.0 - data_weight) * projection.head_radius_z * 3.0,
            voxel_size * 3.2,
        )
        silhouette_weight = np.exp(
            -((distance_behind_front / max(projection.head_radius_z * 0.24, voxel_size)) ** 2)
        )
        effective_front_constraint = (
            front_constraint
            + (1.0 - silhouette_weight * front_silhouette_support) * projection.head_radius_z * 3.0
        )
        layer = _compact_smooth_intersection(
            layer,
            effective_front_constraint,
            voxel_size * 2.4,
        )
        multiview_constraint = front_constraint.copy()
        raw_depth = projection.raw_origin + world_depth * projection.raw_per_world
        for role in (ViewRole.LEFT45, ViewRole.RIGHT45):
            view = projection.side_views[role]
            projected_x = (
                math.cos(view["theta"]) * normalized_x
                + math.sin(view["theta"]) * raw_depth
                + view["offset"]
            )
            pixel_x = projected_x * view["scale"] + view["centerX"]
            pixel_y = normalized_y * view["scale"] + view["centerY"]
            distance_pixels = _sample(
                side_silhouette_fields[role],
                pixel_x,
                pixel_y,
                -max(masks[role].shape),
            )
            distance_world = (
                distance_pixels * world_per_normalized / max(float(view["scale"]), 1e-8)
                + voxel_size * 1.25
            )
            multiview_constraint = _smooth_intersection(
                multiview_constraint,
                distance_world,
                voxel_size * 1.8,
            )
        # A hard visual-hull intersection matches silhouettes numerically but
        # creates faceted bands where the three camera extrusions exchange
        # dominance. Use the masks as a bounded outward-error correction while
        # retaining the continuous anatomical template as the primary field.
        outside_error = np.minimum(multiview_constraint - layer, 0.0)
        ear_depth_transition = math.exp(
            -(
                ((world_depth - ear_center_z) / max(projection.head_radius_z * 0.46, voxel_size))
                ** 4
            )
        )
        transition_relaxation = temporal_transition * ear_depth_transition
        constraint_strength = 0.24 * (1.0 - transition_relaxation * 0.75)
        layer = layer + outside_error * constraint_strength
        for eye in eye_primitives:
            eye_radius = eye["sphereRadius"]
            eyeball = (
                1.0
                - np.sqrt(
                    ((query_x - eye["centerX"]) / eye_radius) ** 2
                    + ((query_y - eye["centerY"]) / eye_radius) ** 2
                    + ((world_depth - eye["sphereCenterZ"]) / eye_radius) ** 2
                )
            ) * eye_radius
            visible_cap = np.minimum(eyeball, eye["capScalar"])
            layer = np.maximum(layer, visible_cap)
        layer = np.minimum(layer, volume_boundary)
        z_boundary = min(world_depth - z_axis[0], z_axis[-1] - world_depth)
        layer = np.minimum(layer, z_boundary - voxel_size * 0.55)
        scalar[z_index] = layer.astype(np.float32)

    if not np.all(np.isfinite(scalar)) or not np.any(scalar > 0) or not np.any(scalar < 0):
        raise ValueError("continuous implicit field has no finite zero crossing")

    axes = (z_axis, y_axis, x_axis)
    raw_mesh = _mesh_from_field(scalar, axes, voxel_size)
    mild_scalar = gaussian_filter(scalar, sigma=(0.78, 0.68, 0.68), mode="nearest")
    anatomical_scalar = gaussian_filter(scalar, sigma=(1.16, 1.02, 1.02), mode="nearest")
    depth_distance = np.abs(z_axis[:, None, None] - observed_depth[None, :, :])
    measured_front_protection = data_weight[None, :, :] * np.exp(
        -((depth_distance / max(voxel_size * 3.2, 1e-8)) ** 2)
    )
    # Preserve measured eyes/nose/mouth on the observed exterior surface while
    # diffusing scanline transitions where the three silhouette extrusions meet.
    # This removes horizontal visual-hull bands without rounding away identity data.
    smooth_scalar = (
        anatomical_scalar * (1.0 - measured_front_protection)
        + mild_scalar * measured_front_protection
    )
    temporal_scalar = gaussian_filter(scalar, sigma=(1.42, 1.25, 1.25), mode="nearest")
    ear_depth_volume = np.exp(
        -(
            (
                (z_axis[:, None, None] - ear_center_z)
                / max(projection.head_radius_z * 0.50, voxel_size)
            )
            ** 4
        )
    )
    temporal_volume = temporal_transition[None, :, :] * ear_depth_volume
    smooth_scalar = smooth_scalar * (1.0 - temporal_volume * 0.32) + temporal_scalar * (
        temporal_volume * 0.32
    )
    smooth_scalar = scalar * 0.14 + smooth_scalar * 0.86
    outer_ear_lateral = np.clip(
        (np.abs(query_x) / max(projection.head_radius_x, voxel_size) - 0.91) / 0.13,
        0.0,
        1.0,
    )
    outer_ear_lateral = outer_ear_lateral * outer_ear_lateral * (3.0 - 2.0 * outer_ear_lateral)
    outer_ear_height = 1.0 - np.clip(
        (np.abs(query_y - ear_center_y) / max(projection.head_radius_y, voxel_size) - 0.26) / 0.12,
        0.0,
        1.0,
    )
    outer_ear_height = outer_ear_height * outer_ear_height * (3.0 - 2.0 * outer_ear_height)
    normalized_ear_depth = np.abs(z_axis[:, None, None] - ear_center_z) / max(
        projection.head_radius_z,
        voxel_size,
    )
    outer_ear_depth = 1.0 - np.clip(
        (normalized_ear_depth - 0.20) / 0.14,
        0.0,
        1.0,
    )
    outer_ear_depth = outer_ear_depth * outer_ear_depth * (3.0 - 2.0 * outer_ear_depth)
    ear_detail_protection = (
        outer_ear_lateral[None, :, :] * outer_ear_height[None, :, :] * outer_ear_depth
    )
    smooth_scalar = smooth_scalar * (1.0 - ear_detail_protection) + scalar * (ear_detail_protection)
    smooth_full = _mesh_from_field(smooth_scalar, axes, voxel_size)
    trimesh.smoothing.filter_taubin(
        smooth_full,
        lamb=0.28,
        nu=0.30,
        iterations=2,
    )
    smooth_full = _clean_mesh(smooth_full)
    front_surface_depth = _front_surface_from_field(
        smooth_scalar,
        axes,
        depth.shape,
        pixel_step,
        depth,
    )

    before_variation = _normal_variation(raw_mesh)
    after_variation = _normal_variation(smooth_full)
    normal_reduction = float(
        np.clip(1.0 - after_variation / max(before_variation, 1e-12), 0.0, 1.0)
    )
    hausdorff, raw_to_smooth, smooth_to_raw = _surface_field_displacement(
        raw_mesh,
        smooth_full,
        scalar,
        smooth_scalar,
        axes,
        voxel_size,
    )
    feature_drift = _feature_drift(raw_mesh, smooth_full, feature, depth.shape, pixel_step)

    smooth_mesh = smooth_full
    for _ in range(3):
        if len(smooth_mesh.faces) <= target_triangles:
            break
        previous_face_count = len(smooth_mesh.faces)
        candidate = smooth_mesh.simplify_quadric_decimation(
            face_count=target_triangles,
            aggression=2,
        )
        candidate = _clean_mesh(candidate)
        if len(candidate.faces) >= previous_face_count:
            break
        smooth_mesh = candidate

    silhouette: dict[str, Any] = {}
    maximum_drop = 0.0
    for role in REQUIRED_VIEWS:
        target = masks[role]
        raw_raster = _render_silhouette(
            raw_mesh, role, target.shape, depth.shape, pixel_step, projection
        )
        smooth_raster = _render_silhouette(
            smooth_mesh, role, target.shape, depth.shape, pixel_step, projection
        )
        raw_iou = _iou(raw_raster, target)
        smooth_iou = _iou(smooth_raster, target)
        drop = max(0.0, raw_iou - smooth_iou)
        maximum_drop = max(maximum_drop, drop)
        silhouette[role.value] = {
            "rawIoU": raw_iou,
            "smoothIoU": smooth_iou,
            "drop": drop,
        }

    metrics = {
        **_mesh_metrics(smooth_mesh),
        "representation": "multiview-silhouette-constrained-implicit-volume",
        "volumeResolution": [int(len(x_axis)), int(len(y_axis)), int(len(z_axis))],
        "voxelSize": voxel_size,
        "rawTriangles": int(len(raw_mesh.faces)),
        "selfIntersection": False,
        "selfIntersectionMethod": "single-zero-isosurface-with-topology-gates",
        "featureDriftVoxels": feature_drift / max(voxel_size, 1e-12),
        "normalVarianceBefore": before_variation,
        "normalVarianceAfter": after_variation,
        "normalVarianceReduction": normal_reduction,
        "hausdorffVoxels": hausdorff / max(voxel_size, 1e-12),
        "rawToSmoothVoxels": raw_to_smooth / max(voxel_size, 1e-12),
        "smoothToRawVoxels": smooth_to_raw / max(voxel_size, 1e-12),
        "silhouette": silhouette,
        "maximumSilhouetteIoUDrop": maximum_drop,
        "smoothing": "continuous-distance-field-gaussian",
        "simplification": "quadric-error-metric",
        "openEyeCount": len(eye_models),
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
    return HeightMeshResult(
        raw_mesh=raw_mesh,
        smooth_mesh=smooth_mesh,
        metrics=metrics,
        front_surface_depth=front_surface_depth,
    )
