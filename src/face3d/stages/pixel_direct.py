from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from face3d.config import Face3DConfig
from face3d.errors import fail
from face3d.glb import export_neutral_mesh, export_pixel_instances
from face3d.height_mesh import rear_depth_field
from face3d.implicit_mesh import EarObservation, ProjectionModel, build_multiview_implicit_mesh
from face3d.io import atomic_write_json, sha256_file
from face3d.models import REQUIRED_VIEWS, CameraRecord, ViewRole
from face3d.pixel_binary import write_pixel_records
from face3d.profiles.face_v1 import FaceProfileV1
from face3d.skin import build_skin_component

EYE_INDICES = (
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
)
NOSE_INDICES = (
    168,
    6,
    197,
    195,
    5,
    4,
    1,
    2,
    98,
    97,
    94,
    129,
    45,
    48,
    278,
    275,
    326,
    327,
    358,
)
MOUTH_INDICES = (
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
)
JAW_INDICES = (127, 234, 93, 132, 58, 172, 136, 150, 152, 377, 400, 378, 365, 397, 323, 454, 356)
EAR_INDICES = (127, 234, 93, 132, 356, 454, 323, 361)
FACE_OVAL_INDICES = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)


@dataclass(slots=True)
class ModelGrid:
    mask: np.ndarray
    rgb: np.ndarray
    source_uv_grid: np.ndarray
    crop: tuple[int, int, int, int]
    pixel_step: float


@dataclass(slots=True)
class DepthEstimate:
    raw_anchor_depth: np.ndarray
    world_anchor_depth: np.ndarray
    left_depth: np.ndarray
    right_depth: np.ndarray
    agreement: np.ndarray
    local_x: np.ndarray
    local_y: np.ndarray
    view_geometry: dict[ViewRole, dict[str, float | np.ndarray]]


@dataclass(slots=True)
class PixelShell:
    model_uv: np.ndarray
    source_uv: np.ndarray
    pixel_codes: np.ndarray
    positions: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    thickness: np.ndarray
    confidence: np.ndarray
    source_bits: np.ndarray
    feature_class: np.ndarray
    layer_counts: dict[str, int]
    surface_coverage: float
    front_surface_snap_count: int
    front_surface_max_distance_pixels: float


def _primary_filled_mask(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 127).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        fail("mask-empty", "正面 mask 没有前景", stage="pixel-direct")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    primary = np.where(labels == label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(primary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(primary)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _model_grid(rgb: np.ndarray, mask: np.ndarray, maximum_size: int) -> ModelGrid:
    foreground = np.argwhere(mask > 127)
    if not len(foreground):
        fail("mask-empty", "正面 mask 没有前景", stage="pixel-direct")
    y0, x0 = foreground.min(axis=0)
    y1, x1 = foreground.max(axis=0)
    padding = max(2, round(max(x1 - x0 + 1, y1 - y0 + 1) * 0.02))
    x0 = max(0, int(x0) - padding)
    y0 = max(0, int(y0) - padding)
    x1 = min(rgb.shape[1] - 1, int(x1) + padding)
    y1 = min(rgb.shape[0] - 1, int(y1) + padding)
    crop_width = x1 - x0 + 1
    crop_height = y1 - y0 + 1
    scale = maximum_size / max(crop_width, crop_height)
    grid_width = max(32, round(crop_width * scale))
    grid_height = max(32, round(crop_height * scale))
    crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    grid_mask = cv2.resize(crop_mask, (grid_width, grid_height), interpolation=cv2.INTER_NEAREST)
    grid_mask = _primary_filled_mask(grid_mask)
    source_x = np.clip(
        np.rint(x0 + (np.arange(grid_width) + 0.5) * crop_width / grid_width - 0.5),
        x0,
        x1,
    ).astype(np.uint16)
    source_y = np.clip(
        np.rint(y0 + (np.arange(grid_height) + 0.5) * crop_height / grid_height - 0.5),
        y0,
        y1,
    ).astype(np.uint16)
    source_u, source_v = np.meshgrid(source_x, source_y)
    source_uv_grid = np.dstack((source_u, source_v)).astype(np.uint16)
    sampled_rgb = rgb[source_v, source_u]
    return ModelGrid(
        mask=grid_mask,
        rgb=sampled_rgb,
        source_uv_grid=source_uv_grid,
        crop=(x0, y0, crop_width, crop_height),
        pixel_step=1.0 / grid_width,
    )


def _landmarks_to_grid(
    landmarks: np.ndarray,
    image_size: tuple[int, int],
    grid: ModelGrid,
) -> np.ndarray:
    width, height = image_size
    x0, y0, crop_width, crop_height = grid.crop
    points = landmarks[:, :2] * np.asarray([width, height], dtype=np.float64)
    points[:, 0] = (points[:, 0] - x0 + 0.5) * grid.mask.shape[1] / crop_width - 0.5
    points[:, 1] = (points[:, 1] - y0 + 0.5) * grid.mask.shape[0] / crop_height - 0.5
    return points


def _view_coordinates(landmarks: np.ndarray, width: int, height: int) -> dict[str, Any]:
    pixels = landmarks[:, :2] * np.asarray([width, height], dtype=np.float64)
    vertical_scale = float(abs(pixels[152, 1] - pixels[10, 1]))
    if vertical_scale < 1:
        vertical_scale = float(np.ptp(pixels[:, 1]))
    center_x = float((pixels[234, 0] + pixels[454, 0]) / 2)
    center_y = float((pixels[10, 1] + pixels[152, 1]) / 2)
    return {
        "pixels": pixels,
        "scale": max(vertical_scale, 1.0),
        "centerX": center_x,
        "centerY": center_y,
        "x": (pixels[:, 0] - center_x) / max(vertical_scale, 1.0),
        "y": (pixels[:, 1] - center_y) / max(vertical_scale, 1.0),
    }


def _scan_ear_pixels(
    rgb: np.ndarray,
    head_mask: np.ndarray,
    landmarks: np.ndarray,
    role: ViewRole,
) -> tuple[EarObservation, np.ndarray]:
    """Lift the visible ear's own pixels into a smooth relief observation."""
    if role not in (ViewRole.LEFT45, ViewRole.RIGHT45):
        raise ValueError("ear scanning requires a side view")
    height, width = head_mask.shape
    if rgb.shape[:2] != (height, width):
        raise ValueError("ear source image and mask must have matching shapes")

    pixels = np.asarray(landmarks[:, :2], dtype=np.float64) * np.asarray(
        [width, height], dtype=np.float64
    )
    if role == ViewRole.LEFT45:
        root_indices = np.asarray((127, 234, 93, 132), dtype=np.int32)
        direction = -1.0
    else:
        root_indices = np.asarray((356, 454, 323, 361), dtype=np.int32)
        direction = 1.0
    root = pixels[root_indices]
    face_scale = max(abs(float(pixels[152, 1] - pixels[10, 1])), 1.0)
    top = int(np.clip(np.floor(root[:, 1].min() - face_scale * 0.080), 0, height - 1))
    bottom = int(np.clip(np.ceil(root[:, 1].max() + face_scale * 0.025), top + 1, height - 1))
    rows = np.arange(top, bottom + 1, dtype=np.int32)
    order = np.argsort(root[:, 1])
    root_y = root[order, 1]
    root_x = root[order, 0]
    unique_y, unique_indices = np.unique(root_y, return_index=True)
    root_line = np.interp(rows, unique_y, root_x[unique_indices])
    root_line = gaussian_filter(root_line, sigma=max(len(rows) * 0.035, 1.0), mode="nearest")

    head_binary = np.asarray(head_mask) > 127
    ear_binary = np.zeros((height, width), dtype=np.uint8)
    protrusion = np.zeros((height, width), dtype=np.float32)
    attachment_overlap = int(round(face_scale * 0.018))
    for row, root_column in zip(rows, root_line, strict=True):
        foreground = np.flatnonzero(head_binary[row])
        if not len(foreground):
            continue
        if direction < 0:
            start = int(foreground[0])
            end = min(width - 1, int(round(root_column)) + attachment_overlap)
        else:
            start = max(0, int(round(root_column)) - attachment_overlap)
            end = int(foreground[-1])
        if end >= start:
            row_columns = np.arange(start, end + 1, dtype=np.int32)
            row_foreground = head_binary[row, row_columns]
            ear_binary[row, row_columns] = row_foreground
            span = max(end - start, 1)
            if direction < 0:
                horizontal = (end - row_columns) / span
            else:
                horizontal = (row_columns - start) / span
            # The first part nearest the face is the sacrificial carrier/root.
            # Reach the free-standing ear plate early, then keep a broad
            # plateau so the concha is not buried inside the cranium.
            horizontal = np.clip(horizontal / 0.45, 0.0, 1.0)
            horizontal = horizontal * horizontal * (3.0 - 2.0 * horizontal)
            vertical = max(math.sin(math.pi * (row - top) / max(bottom - top, 1)), 0.0)
            vertical = vertical * vertical * (3.0 - 2.0 * vertical)
            protrusion[row, row_columns] = horizontal * vertical * row_foreground

    kernel_size = max(3, int(round(face_scale * 0.009)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    ear_binary = cv2.morphologyEx(ear_binary, cv2.MORPH_CLOSE, kernel)
    ear_binary = cv2.morphologyEx(
        ear_binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    source_pixels = int(np.count_nonzero(ear_binary))
    if source_pixels < 256:
        raise ValueError(f"{role.value} ear pixel scan produced too little support")
    ear_coordinates = np.argwhere(ear_binary > 0)
    bounds = (
        float(ear_coordinates[:, 1].min()),
        float(ear_coordinates[:, 0].min()),
        float(ear_coordinates[:, 1].max()),
        float(ear_coordinates[:, 0].max()),
    )

    inside_distance = cv2.distanceTransform(ear_binary, cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform(1 - ear_binary, cv2.DIST_L2, 5)
    signed_distance = (inside_distance - outside_distance).astype(np.float32)
    profile_scale = max(float(np.quantile(inside_distance[ear_binary > 0], 0.98)), 1.0)
    profile = gaussian_filter(protrusion, sigma=max(face_scale * 0.0080, 2.5), mode="nearest")
    profile[ear_binary == 0] = 0.0
    profile /= max(float(np.quantile(profile[ear_binary > 0], 0.995)), 1e-6)
    profile = np.clip(profile, 0.0, 1.0).astype(np.float32)

    luminance = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB)[..., 0].astype(
        np.float32
    )
    small_sigma = max(face_scale * 0.0035, 1.5)
    base_sigma = max(face_scale * 0.027, 7.0)
    denoised = gaussian_filter(luminance, sigma=small_sigma, mode="nearest")
    illumination = gaussian_filter(luminance, sigma=base_sigma, mode="nearest")
    residual = denoised - illumination
    reliable = inside_distance >= profile_scale * 0.10
    if not np.any(reliable):
        reliable = ear_binary > 0
    center = float(np.median(residual[reliable]))
    spread = max(float(np.quantile(np.abs(residual[reliable] - center), 0.90)), 1.0)
    relief = np.clip((residual - center) / (spread * 1.8), -1.0, 1.0)
    relief = gaussian_filter(relief, sigma=max(face_scale * 0.0018, 1.0), mode="nearest")
    relief = (relief * np.sqrt(profile)).astype(np.float32)
    relief[ear_binary == 0] = 0.0

    heat = cv2.applyColorMap(
        np.rint(np.clip((relief + 1.0) * 127.5, 0.0, 255.0)).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    preview = np.asarray(rgb, dtype=np.float32).copy()
    alpha = (ear_binary.astype(np.float32) * 0.58)[..., None]
    preview = np.rint(preview * (1.0 - alpha) + heat.astype(np.float32) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(ear_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    cv2.drawContours(preview_bgr, contours, -1, (64, 255, 192), max(2, kernel_size // 3))
    preview = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)

    return (
        EarObservation(
            direction=direction,
            signed_distance=signed_distance,
            profile=profile,
            relief=relief,
            source_pixels=source_pixels,
            bounds=bounds,
            reverse_horizontal=role == ViewRole.RIGHT45,
        ),
        preview,
    )


def estimate_depth_anchors(
    landmarks: dict[ViewRole, np.ndarray],
    image_sizes: dict[ViewRole, tuple[int, int]],
    yaw_degrees: dict[ViewRole, float],
    target_span: float,
) -> DepthEstimate:
    geometry = {
        role: _view_coordinates(landmarks[role], *image_sizes[role]) for role in REQUIRED_VIEWS
    }
    front_x = np.asarray(geometry[ViewRole.FRONT]["x"], dtype=np.float64)
    front_y = np.asarray(geometry[ViewRole.FRONT]["y"], dtype=np.float64)
    side_depths: dict[ViewRole, np.ndarray] = {}
    for role in (ViewRole.LEFT45, ViewRole.RIGHT45):
        # Intake yaw describes the head rotation. Reprojecting a head-space
        # point into that camera uses the inverse relative camera rotation.
        theta = -math.radians(float(yaw_degrees[role]))
        if abs(math.sin(theta)) < 0.25:
            theta = -math.radians(FaceProfileV1().expected_yaw(role))
        side_x = np.asarray(geometry[role]["x"], dtype=np.float64)
        depth = (side_x - math.cos(theta) * front_x) / math.sin(theta)
        depth -= np.median(depth)
        side_depths[role] = depth
        geometry[role]["theta"] = theta
    left = side_depths[ViewRole.LEFT45]
    right = side_depths[ViewRole.RIGHT45]
    triangulated = np.median(np.stack((left, right)), axis=0)
    nose = float(np.median(triangulated[np.asarray(NOSE_INDICES)]))
    cheeks = float(np.median(triangulated[np.asarray((93, 132, 234, 323, 361, 454))]))
    if nose < cheeks:
        triangulated *= -1
        left *= -1
        right *= -1
    disagreement = np.abs(left - right)
    agreement_scale = max(float(np.quantile(disagreement, 0.75)), 1e-4)
    agreement = np.exp(-disagreement / agreement_scale)

    raw = triangulated.copy()
    front_depth = -np.asarray(landmarks[ViewRole.FRONT][:, 2], dtype=np.float64)
    if float(np.ptp(front_depth)) > 1e-5:
        reliable = agreement >= np.quantile(agreement, 0.30)
        lower, upper = np.quantile(triangulated[reliable], (0.02, 0.98))
        reliable &= (triangulated >= lower) & (triangulated <= upper)
        design = np.column_stack((np.ones(len(front_depth)), front_depth))
        weights = reliable.astype(np.float64)
        coefficients = np.zeros(2, dtype=np.float64)
        for _ in range(4):
            selected = weights > 0
            root_weight = np.sqrt(weights[selected])
            coefficients = np.linalg.lstsq(
                design[selected] * root_weight[:, None],
                triangulated[selected] * root_weight,
                rcond=None,
            )[0]
            residual = triangulated - design @ coefficients
            scale = max(float(np.median(np.abs(residual[selected]))) * 1.4826, 1e-5)
            robust = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-8))
            weights = reliable.astype(np.float64) * robust
        learned_depth = design @ coefficients
        triangulation_weight = 0.20 + 0.60 * agreement
        raw = triangulation_weight * triangulated + (1.0 - triangulation_weight) * learned_depth
    low, high = np.quantile(raw, (0.03, 0.97))
    span = max(float(high - low), 1e-6)
    world = (raw - low) * target_span / span
    world = np.clip(world, -0.05 * target_span, 1.12 * target_span)
    return DepthEstimate(
        raw_anchor_depth=raw.astype(np.float32),
        world_anchor_depth=world.astype(np.float32),
        left_depth=left.astype(np.float32),
        right_depth=right.astype(np.float32),
        agreement=agreement.astype(np.float32),
        local_x=front_x.astype(np.float32),
        local_y=front_y.astype(np.float32),
        view_geometry=geometry,
    )


def _feature_classes(points: np.ndarray, shape: tuple[int, int], radius: float) -> np.ndarray:
    feature = np.zeros(shape, dtype=np.uint8)

    def fill_region(
        indices: tuple[int, ...],
        value: int,
        *,
        scale: float = 1.0,
    ) -> None:
        selected = np.rint(points[np.asarray(indices)]).astype(np.int32)
        region = np.zeros(shape, dtype=np.uint8)
        cv2.fillConvexPoly(region, cv2.convexHull(selected), 255)
        diameter = max(1, round(radius * 2 * scale)) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        region = cv2.dilate(region, kernel, iterations=1)
        feature[region > 0] = int(value)

    jaw = np.rint(points[np.asarray(JAW_INDICES)]).astype(np.int32)
    cv2.polylines(
        feature,
        [jaw],
        False,
        5,
        thickness=max(1, round(radius * 0.8)),
    )
    fill_region(EAR_INDICES[:4], 4, scale=1.0)
    fill_region(EAR_INDICES[4:], 4, scale=1.0)
    fill_region(MOUTH_INDICES, 3, scale=0.65)
    fill_region(NOSE_INDICES, 2, scale=0.55)
    fill_region(EYE_INDICES[:16], 1, scale=0.55)
    fill_region(EYE_INDICES[16:], 1, scale=0.55)
    return feature


def _feature_protection(points: np.ndarray, shape: tuple[int, int], radius: float) -> np.ndarray:
    protection = np.zeros(shape, dtype=np.uint8)
    protected_indices = tuple(sorted(set(EYE_INDICES + NOSE_INDICES + MOUTH_INDICES + JAW_INDICES)))
    point_radius = max(1, round(radius * 0.12))
    for point in np.rint(points[np.asarray(protected_indices)]).astype(np.int32):
        cv2.circle(protection, tuple(point), point_radius, 1, thickness=cv2.FILLED)
    return protection


def _semantic_relief(
    points: np.ndarray,
    shape: tuple[int, int],
    depth_span: float,
) -> np.ndarray:
    relief = np.zeros(shape, dtype=np.float32)
    yy, xx = np.indices(shape, dtype=np.float32)

    def blurred_region(indices: tuple[int, ...], sigma: float) -> np.ndarray:
        layer = np.zeros(shape, dtype=np.float32)
        polygon = cv2.convexHull(np.rint(points[np.asarray(indices)]).astype(np.int32))
        cv2.fillConvexPoly(layer, polygon, 1.0)
        layer = cv2.GaussianBlur(layer, (0, 0), max(sigma, 0.6))
        return layer / max(float(layer.max()), 1e-6)

    def blurred_line(indices: tuple[int, ...], sigma: float) -> np.ndarray:
        layer = np.zeros(shape, dtype=np.float32)
        polyline = np.rint(points[np.asarray(indices)]).astype(np.int32)
        cv2.polylines(layer, [polyline], False, 1.0, thickness=1, lineType=cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), max(sigma, 0.6))
        return layer / max(float(layer.max()), 1e-6)

    for eye_indices in (EYE_INDICES[:16], EYE_INDICES[16:]):
        eye_points = points[np.asarray(eye_indices)]
        eye_width = max(float(np.ptp(eye_points[:, 0])), 2.0)
        center = np.mean(eye_points, axis=0)
        globe = np.exp(
            -0.5
            * (
                ((xx - center[0]) / (eye_width * 0.42)) ** 2
                + ((yy - center[1]) / (eye_width * 0.19)) ** 2
            )
        )
        socket = np.exp(
            -0.5
            * (
                ((xx - center[0]) / (eye_width * 0.62)) ** 2
                + ((yy - center[1]) / (eye_width * 0.34)) ** 2
            )
        )
        crease = blurred_line(eye_indices + (eye_indices[0],), eye_width * 0.025)
        upper = eye_indices[12]
        lower = eye_indices[4]
        open_ratio = abs(float(points[upper, 1] - points[lower, 1])) / eye_width
        is_open = open_ratio >= 0.12
        relief -= socket * (depth_span * (0.007 if is_open else 0.010))
        relief += globe * (depth_span * (0.004 if is_open else 0.050))
        relief -= crease * (depth_span * 0.005)

    for eyelid in (
        (33, 160, 159, 158, 157, 173, 133),
        (362, 398, 384, 385, 386, 387, 388, 466, 263),
    ):
        eyelid_width = max(float(np.ptp(points[np.asarray(eyelid), 0])), 2.0)
        relief -= blurred_line(eyelid, eyelid_width * 0.028) * (depth_span * 0.004)

    for brow in ((70, 63, 105, 66, 107), (336, 296, 334, 293, 300)):
        brow_width = max(float(np.ptp(points[np.asarray(brow), 0])), 2.0)
        relief += blurred_line(brow, brow_width * 0.07) * (depth_span * 0.006)

    nose_width = max(float(abs(points[327, 0] - points[98, 0])), 2.0)
    relief += blurred_line((168, 6, 197, 195, 5, 4, 1), nose_width * 0.10) * (depth_span * 0.010)
    for nostril in (98, 327):
        layer = np.zeros(shape, dtype=np.float32)
        center = tuple(np.rint(points[nostril]).astype(np.int32))
        cv2.circle(layer, center, max(1, round(nose_width * 0.08)), 1.0, thickness=cv2.FILLED)
        layer = cv2.GaussianBlur(layer, (0, 0), max(nose_width * 0.045, 0.6))
        relief -= layer / max(float(layer.max()), 1e-6) * (depth_span * 0.010)

    mouth_width = max(float(abs(points[291, 0] - points[61, 0])), 2.0)
    relief += blurred_region(MOUTH_INDICES, mouth_width * 0.060) * (depth_span * 0.008)
    upper_seam = (61, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 291)
    lower_seam = (61, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291)
    relief -= blurred_line(upper_seam, mouth_width * 0.022) * (depth_span * 0.008)
    relief -= blurred_line(lower_seam, mouth_width * 0.024) * (depth_span * 0.002)
    return relief


def _canonical_depth_field(
    points: np.ndarray,
    anchors: np.ndarray,
    fallback: np.ndarray,
    mask: np.ndarray,
    canonical_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Regularize measured depth with MediaPipe's 468-vertex face topology."""
    vertices: list[tuple[float, float, float]] = []
    if canonical_path.is_file():
        for line in canonical_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append((float(x), float(y), float(z)))
    count = min(468, len(points), len(anchors), len(vertices))
    if count < 468:
        return (
            fallback,
            _face_data_mask(points, mask.shape, mask),
            {
                "used": False,
                "reason": "canonical-model-missing-or-incomplete",
            },
        )

    canonical_z = np.asarray(vertices[:count], dtype=np.float64)[:, 2]
    measured = np.asarray(anchors[:count], dtype=np.float64)
    correlation = float(np.corrcoef(canonical_z, measured)[0, 1])
    data_mask = _face_data_mask(points[:count], mask.shape, mask)
    if not np.isfinite(correlation) or correlation < 0.60:
        return (
            fallback,
            data_mask,
            {
                "used": False,
                "reason": "depth-topology-correlation-too-low",
                "correlation": correlation,
            },
        )

    design = np.column_stack((np.ones(count), canonical_z))
    coefficients = np.linalg.lstsq(design, measured, rcond=None)[0]
    canonical_anchors = design @ coefficients
    height, width = mask.shape
    coordinate_scale = np.asarray([max(width - 1, 1), max(height - 1, 1)])
    normalized_points = points[:count] / coordinate_scale
    interpolator = RBFInterpolator(
        normalized_points,
        canonical_anchors,
        neighbors=72,
        smoothing=0.0008,
        kernel="thin_plate_spline",
    )
    coarse_size = 96
    coarse_width = max(24, round(coarse_size * width / max(width, height)))
    coarse_height = max(24, round(coarse_size * height / max(width, height)))
    coarse_x = np.linspace(0, width - 1, coarse_width)
    coarse_y = np.linspace(0, height - 1, coarse_height)
    query_x, query_y = np.meshgrid(coarse_x, coarse_y)
    query = np.column_stack((query_x.ravel(), query_y.ravel())) / coordinate_scale
    canonical_coarse = interpolator(query).reshape(coarse_height, coarse_width)
    canonical_field = cv2.resize(
        canonical_coarse.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    canonical_field = cv2.GaussianBlur(canonical_field, (0, 0), 0.65)

    identity_residual = cv2.GaussianBlur(
        np.asarray(fallback, dtype=np.float32) - canonical_field,
        (0, 0),
        9.0,
    )
    regularized = canonical_field + identity_residual * 0.38
    blend = cv2.GaussianBlur((data_mask > 0).astype(np.float32), (0, 0), 2.2)
    blend = np.clip(blend, 0.0, 1.0)
    field = np.asarray(fallback, dtype=np.float32) * (1.0 - blend) + regularized * blend
    field[mask == 0] = np.asarray(fallback, dtype=np.float32)[mask == 0]
    return (
        field.astype(np.float32),
        data_mask,
        {
            "used": True,
            "correlation": correlation,
            "scale": float(coefficients[1]),
            "offset": float(coefficients[0]),
        },
    )


def _face_data_mask(
    points: np.ndarray,
    shape: tuple[int, int],
    mask: np.ndarray,
) -> np.ndarray:
    selected = points[np.asarray(FACE_OVAL_INDICES)]
    face_width = max(float(np.ptp(selected[:, 0])), 1.0)
    margin = max(3.0, face_width * 0.045)
    distance = cv2.distanceTransform((np.asarray(mask) > 0).astype(np.uint8), cv2.DIST_L2, 5)
    result = np.where(distance >= margin, 255, 0).astype(np.uint8)
    return result


def _photometric_relief(
    rgb: np.ndarray,
    feature_class: np.ndarray,
    depth_span: float,
) -> np.ndarray:
    """Turn only local image contrast in semantic features into shallow relief."""
    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    local = cv2.GaussianBlur(gray, (0, 0), 1.2) - cv2.GaussianBlur(gray, (0, 0), 5.5)
    relief = np.zeros(gray.shape, dtype=np.float32)
    amplitudes = {1: 0.010, 2: 0.012, 3: 0.010, 4: 0.006}
    for feature_value, amplitude in amplitudes.items():
        region = feature_class == feature_value
        if not np.any(region):
            continue
        scale = max(float(np.quantile(np.abs(local[region]), 0.90)), 2.0)
        normalized = np.clip(local / scale, -1.0, 1.0)
        # Dark eyelid, nostril and mouth-seam pixels are useful concavity cues.
        # Positive albedo/highlights are less reliable, so keep them shallow.
        shaped = np.minimum(normalized, 0.0) + np.maximum(normalized, 0.0) * 0.22
        weight = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 1.5)
        relief += shaped * weight * (depth_span * amplitude)
    return cv2.GaussianBlur(relief, (0, 0), 0.55).astype(np.float32)


def _open_eye_models(
    points: np.ndarray,
    depth: np.ndarray,
    grid: ModelGrid,
) -> list[dict[str, float]]:
    models: list[dict[str, float]] = []
    center_x = (grid.mask.shape[1] - 1) / 2
    center_y = (grid.mask.shape[0] - 1) / 2
    for corners, top, bottom in (
        ((33, 133), 159, 145),
        ((263, 362), 386, 374),
    ):
        left, right = corners
        eye_width_pixels = abs(float(points[right, 0] - points[left, 0]))
        if eye_width_pixels < 2.0:
            continue
        open_ratio = abs(float(points[top, 1] - points[bottom, 1])) / eye_width_pixels
        if open_ratio < 0.12:
            continue
        eye_center = np.mean(points[np.asarray((left, right, top, bottom))], axis=0)
        column = int(np.clip(round(float(eye_center[0])), 0, depth.shape[1] - 1))
        row = int(np.clip(round(float(eye_center[1])), 0, depth.shape[0] - 1))
        width_world = eye_width_pixels * grid.pixel_step
        sphere_radius = width_world * 0.43
        surface_depth = float(depth[row, column])
        models.append(
            {
                "centerX": float((eye_center[0] - center_x) * grid.pixel_step),
                "centerY": float((center_y - eye_center[1]) * grid.pixel_step),
                "apertureRadiusX": float(width_world * 0.50),
                "apertureRadiusY": float(width_world * np.clip(open_ratio * 0.58, 0.14, 0.23)),
                "sphereRadius": float(sphere_radius),
                # Keep the eyeball behind the lids. A shallower center reads as
                # a white bead even when the aperture measurements are correct.
                "sphereCenterZ": float(surface_depth - sphere_radius * 0.94),
                "recessDepth": float(sphere_radius * 0.40),
                "openRatio": float(open_ratio),
            }
        )
    return models


def _pixel_surface_shell(
    grid: ModelGrid,
    column_front: np.ndarray,
    feature_grid: np.ndarray,
    confidence_grid: np.ndarray,
    *,
    cell_fill_ratio: float,
    maximum_cells: int,
    surface_mesh: trimesh.Trimesh | None = None,
) -> PixelShell:
    """Create front pixels plus an evenly sampled, anatomically coherent shell."""
    rows, columns = np.nonzero(grid.mask > 127)
    base_count = len(rows)
    model_uv_base = np.column_stack((columns, rows)).astype(np.uint16)
    source_uv_base = grid.source_uv_grid[rows, columns]
    colors = grid.rgb[rows, columns].astype(np.uint32)
    codes_base = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
    x_base = (columns - (grid.mask.shape[1] - 1) / 2) * grid.pixel_step
    y_base = ((grid.mask.shape[0] - 1) / 2 - rows) * grid.pixel_step
    front_base = column_front[rows, columns]
    feature_base = feature_grid[rows, columns].astype(np.uint8)
    confidence_base = confidence_grid[rows, columns].astype(np.float32)

    target_cells = min(maximum_cells, max(base_count * 4, 48_000))
    inferred_budget = max(0, target_cells - base_count)
    cell_xy = grid.pixel_step * max(cell_fill_ratio, 1.10)
    inferred_positions = np.empty((0, 3), dtype=np.float32)
    inferred_indices = np.empty(0, dtype=np.int32)
    if surface_mesh is not None and inferred_budget:
        # Use the existing continuous topology itself instead of a random point
        # cloud. Eligible vertices and face centres cover every lateral/rear
        # triangle deterministically and keep the cell pitch uniform across the
        # measured-face/template transition.
        surface_vertices = np.asarray(surface_mesh.vertices, dtype=np.float32)
        even_points, _ = trimesh.sample.sample_surface_even(
            surface_mesh,
            inferred_budget,
            radius=max(cell_xy * 0.42, 1e-8),
            seed=20260809,
        )
        topology_points = np.vstack(
            (
                surface_vertices,
                np.asarray(surface_mesh.triangles_center, dtype=np.float32),
            )
        )
        remaining = max(0, inferred_budget - len(even_points))
        if len(topology_points) > remaining:
            selected = np.linspace(
                0,
                len(topology_points) - 1,
                remaining,
                dtype=np.int64,
            )
            topology_points = topology_points[selected]
        candidate_points = np.vstack((even_points[:inferred_budget], topology_points))[
            :inferred_budget
        ]
        inferred_positions = np.asarray(candidate_points, dtype=np.float32)
        inferred_indices = (
            cKDTree(np.column_stack((x_base, y_base)))
            .query(inferred_positions[:, :2], k=1)[1]
            .astype(np.int32)
        )

    front_positions = np.column_stack((x_base, y_base, front_base)).astype(np.float32)
    front_surface_snap_count = 0
    front_surface_max_distance_pixels = 0.0
    if surface_mesh is not None:
        surface_vertices = np.asarray(surface_mesh.vertices, dtype=np.float32)
        distance, nearest_vertex = cKDTree(surface_vertices).query(front_positions, k=1)
        off_surface = distance > grid.pixel_step * 2.0
        front_surface_snap_count = int(np.count_nonzero(off_surface))
        front_positions[off_surface] = surface_vertices[nearest_vertex[off_surface]]
        distance_after, _ = cKDTree(surface_vertices).query(front_positions, k=1)
        front_surface_max_distance_pixels = float(
            np.max(distance_after, initial=0.0) / max(grid.pixel_step, 1e-8)
        )
    positions = np.vstack((front_positions, inferred_positions)).astype(np.float32)
    instance_indices = np.concatenate((np.arange(base_count, dtype=np.int32), inferred_indices))
    # Surface cells are shallow convex tiles rather than axis-aligned cubes.
    # Their local Z axis is aligned to the continuous mesh normal below. This
    # prevents cube corners from producing a corrugated silhouette on the dome.
    scales = np.empty((len(instance_indices), 3), dtype=np.float32)
    scales[:, 0:2] = cell_xy
    scales[:, 2] = cell_xy * 0.20
    rotations = np.zeros((len(positions), 4), dtype=np.float32)
    rotations[:, 3] = 1.0
    if surface_mesh is not None:
        surface_vertices = np.asarray(surface_mesh.vertices, dtype=np.float32)
        nearest = cKDTree(surface_vertices).query(positions, k=1)[1]
        normals = np.asarray(surface_mesh.vertex_normals, dtype=np.float32)[nearest]
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        rotations[:, 0] = -normals[:, 1]
        rotations[:, 1] = normals[:, 0]
        rotations[:, 3] = 1.0 + normals[:, 2]
        opposite = rotations[:, 3] < 1e-5
        rotations[opposite] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-8)
    surface_coverage = 1.0
    if surface_mesh is not None and len(inferred_positions):
        review_points, _ = trimesh.sample.sample_surface_even(
            surface_mesh,
            12_000,
            seed=20260809,
        )
        tree = cKDTree(positions)
        neighbor_count = min(20, len(positions))
        for _ in range(20):
            distances, neighbors = tree.query(review_points, k=neighbor_count)
            if neighbors.ndim == 1:
                neighbors = neighbors[:, None]
                distances = distances[:, None]
            covered = np.any(distances <= scales[neighbors, 0] * 0.55, axis=1)
            surface_coverage = float(np.mean(covered))
            if surface_coverage >= 0.995:
                break
            scales[:, 0:2] *= 1.04
            scales[:, 2] = scales[:, 0] * 0.20
    confidence = np.concatenate(
        (
            confidence_base,
            np.minimum(confidence_base[inferred_indices], 0.52).astype(np.float32),
        )
    )
    source_bits = np.concatenate(
        (
            np.full(base_count, 1 | 2 | 4, dtype=np.uint8),
            np.full(len(inferred_positions), 2 | 4 | 8, dtype=np.uint8),
        )
    )
    feature_class = np.concatenate(
        (feature_base, np.zeros(len(inferred_positions), dtype=np.uint8))
    )
    return PixelShell(
        model_uv=model_uv_base[instance_indices],
        source_uv=source_uv_base[instance_indices],
        pixel_codes=codes_base[instance_indices],
        positions=positions,
        scales=scales,
        rotations=rotations,
        thickness=np.max(scales, axis=1).astype(np.float32),
        confidence=confidence,
        source_bits=source_bits,
        feature_class=feature_class,
        layer_counts={
            "frontMeasured": base_count,
            "multiViewSurface": int(len(inferred_positions)),
            "depthColumns": 0,
        },
        surface_coverage=surface_coverage,
        front_surface_snap_count=front_surface_snap_count,
        front_surface_max_distance_pixels=front_surface_max_distance_pixels,
    )


def _stabilize_feature_depth(values: np.ndarray) -> np.ndarray:
    stabilized = np.asarray(values, dtype=np.float32).copy()
    for eye_indices, brow_indices in (
        (EYE_INDICES[:16], (70, 63, 105, 66, 107)),
        (EYE_INDICES[16:], (336, 296, 334, 293, 300)),
    ):
        eye = np.asarray(eye_indices, dtype=np.int32)
        brow = np.asarray(brow_indices, dtype=np.int32)
        eyelid_floor = float(np.median(stabilized[brow])) - 0.015
        stabilized[eye] = np.maximum(stabilized[eye], eyelid_floor)

    mouth = np.asarray(MOUTH_INDICES, dtype=np.int32)
    mouth_center = float(np.median(stabilized[mouth]))
    stabilized[mouth] = np.clip(
        stabilized[mouth],
        mouth_center - 0.018,
        mouth_center + 0.018,
    )
    return stabilized


def _rbf_field(
    points: np.ndarray,
    values: np.ndarray,
    shape: tuple[int, int],
    coarse_size: int,
    complex_mask: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    rounded = np.rint(points * 16).astype(np.int64)
    _, unique = np.unique(rounded, axis=0, return_index=True)
    points = points[unique]
    values = values[unique]
    height, width = shape
    coordinate_scale = np.asarray([max(width - 1, 1), max(height - 1, 1)])
    normalized_points = points / coordinate_scale
    inside = (
        (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
    )
    sample_points = np.rint(points[inside]).astype(np.int32)
    inside_indices = np.flatnonzero(inside)
    inside[inside_indices] &= mask[sample_points[:, 1], sample_points[:, 0]] > 0
    if np.count_nonzero(inside) < 12:
        raise ValueError("too few depth anchors inside the face mask")

    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    distance_scale = max(float(np.quantile(distance[mask > 0], 0.98)), 1.0)
    distance /= distance_scale
    point_pixels = np.rint(points).astype(np.int32)
    point_pixels[:, 0] = np.clip(point_pixels[:, 0], 0, width - 1)
    point_pixels[:, 1] = np.clip(point_pixels[:, 1], 0, height - 1)
    point_distance = distance[point_pixels[:, 1], point_pixels[:, 0]]
    centered = normalized_points - 0.5
    design = np.column_stack(
        (
            np.ones(len(points)),
            point_distance,
            point_distance**2,
            centered[:, 0] ** 2,
            centered[:, 1],
            centered[:, 1] ** 2,
        )
    )
    weights = inside.astype(np.float64)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(4):
        selected = weights > 0
        weighted = np.sqrt(weights[selected])
        coefficients = np.linalg.lstsq(
            design[selected] * weighted[:, None],
            values[selected] * weighted,
            rcond=None,
        )[0]
        residual = values - design @ coefficients
        scale = max(float(np.median(np.abs(residual[selected]))) * 1.4826, 1e-5)
        robust = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-8))
        weights = inside.astype(np.float64) * robust
    baseline_at_points = design @ coefficients
    residual_values = values - baseline_at_points
    neighbours = min(96, len(points))
    interpolator = RBFInterpolator(
        normalized_points,
        residual_values,
        neighbors=neighbours,
        smoothing=0.012,
        kernel="thin_plate_spline",
    )
    detail_interpolator = RBFInterpolator(
        normalized_points,
        residual_values,
        neighbors=min(56, len(points)),
        smoothing=0.0025,
        kernel="thin_plate_spline",
    )
    coarse_width = max(8, round(coarse_size * width / max(width, height)))
    coarse_height = max(8, round(coarse_size * height / max(width, height)))
    coarse_x = np.linspace(0, width - 1, coarse_width)
    coarse_y = np.linspace(0, height - 1, coarse_height)
    query_x, query_y = np.meshgrid(coarse_x, coarse_y)
    coarse_query = np.column_stack((query_x.ravel(), query_y.ravel())) / coordinate_scale
    residual_coarse = interpolator(coarse_query).reshape(coarse_height, coarse_width)
    query_centered = coarse_query - 0.5
    coarse_distance = cv2.resize(
        distance, (coarse_width, coarse_height), interpolation=cv2.INTER_AREA
    ).ravel()
    coarse_design = np.column_stack(
        (
            np.ones(len(coarse_query)),
            coarse_distance,
            coarse_distance**2,
            query_centered[:, 0] ** 2,
            query_centered[:, 1],
            query_centered[:, 1] ** 2,
        )
    )
    baseline_coarse = (coarse_design @ coefficients).reshape(coarse_height, coarse_width)
    coarse = baseline_coarse + residual_coarse
    baseline = cv2.resize(
        baseline_coarse.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC
    )
    residual_field = cv2.resize(
        residual_coarse.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC
    )
    residual_field = cv2.GaussianBlur(residual_field, (0, 0), 1.25)
    rows, columns = np.nonzero(complex_mask)
    if len(rows):
        direct_query = np.column_stack((columns, rows)) / coordinate_scale
        direct = detail_interpolator(direct_query).astype(np.float32)
        residual_limit = max(float(np.quantile(np.abs(residual_values[inside]), 0.95)), 1e-4)
        direct = np.clip(direct, -residual_limit, residual_limit)
        direct_layer = residual_field.copy()
        direct_layer[rows, columns] = direct
        blend = cv2.GaussianBlur(complex_mask.astype(np.float32), (0, 0), 1.4)
        blend = np.clip(blend, 0.0, 1.0)
        residual_field = residual_field * (1.0 - blend) + direct_layer * blend

    hull = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.rint(points).astype(np.int32)), 255)
    outside_distance = cv2.distanceTransform((hull == 0).astype(np.uint8), cv2.DIST_L2, 5)
    support = np.exp(-outside_distance / max(width * 0.06, 1.0)).astype(np.float32)
    boundary_support = np.clip(distance / 0.12, 0.0, 1.0).astype(np.float32)
    field = baseline + residual_field * support * boundary_support
    lower, upper = np.quantile(values[inside], (0.01, 0.99))
    margin = max(float(upper - lower) * 0.08, 1e-4)
    field = np.clip(field, lower - margin, upper + margin)
    field[mask == 0] = float(lower)
    return field, coarse.astype(np.float32)


def _projected_silhouette(
    grid: ModelGrid,
    raw_front_depth: np.ndarray,
    raw_rear_depth: np.ndarray,
    estimate: DepthEstimate,
    role: ViewRole,
    target_shape: tuple[int, int],
) -> np.ndarray:
    rows, columns = np.nonzero(grid.mask > 127)
    source = grid.source_uv_grid[rows, columns].astype(np.float64)
    front_geometry = estimate.view_geometry[ViewRole.FRONT]
    side_geometry = estimate.view_geometry[role]
    x = (source[:, 0] - float(front_geometry["centerX"])) / float(front_geometry["scale"])
    y = (source[:, 1] - float(front_geometry["centerY"])) / float(front_geometry["scale"])
    theta = float(side_geometry["theta"])
    anchor_predicted = (
        math.cos(theta) * estimate.local_x + math.sin(theta) * estimate.raw_anchor_depth
    )
    observed = np.asarray(side_geometry["x"], dtype=np.float64)
    offset = float(np.median(observed - anchor_predicted))
    samples = np.linspace(0.0, 1.0, 7, dtype=np.float64)
    rear = raw_rear_depth[rows, columns]
    front = raw_front_depth[rows, columns]
    depths = rear[:, None] + (front - rear)[:, None] * samples[None, :]
    projected_x = math.cos(theta) * x[:, None] + math.sin(theta) * depths + offset
    pixel_x = projected_x * float(side_geometry["scale"]) + float(side_geometry["centerX"])
    pixel_y = np.repeat(
        (y * float(side_geometry["scale"]) + float(side_geometry["centerY"]))[:, None],
        len(samples),
        axis=1,
    )
    height, width = target_shape
    points = np.rint(np.column_stack((pixel_x.ravel(), pixel_y.ravel()))).astype(np.int32)
    inside = (
        (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
    )
    raster = np.zeros((height, width), dtype=np.uint8)
    points = points[inside]
    if len(points):
        for row in np.unique(points[:, 1]):
            row_columns = points[points[:, 1] == row, 0]
            raster[row, row_columns.min() : row_columns.max() + 1] = 255
    sampling_gap = max(1, round(float(front_geometry["scale"]) / grid.mask.shape[0] * 1.6))
    kernel_size = sampling_gap * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    raster = cv2.morphologyEx(raster, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        raster.fill(0)
        cv2.drawContours(raster, [largest], -1, 255, thickness=cv2.FILLED)
    return raster


def _silhouette_depth_interval(
    grid: ModelGrid,
    estimate: DepthEstimate,
    masks: dict[ViewRole, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = np.nonzero(grid.mask > 127)
    source = grid.source_uv_grid[rows, columns].astype(np.float64)
    front_geometry = estimate.view_geometry[ViewRole.FRONT]
    x = (source[:, 0] - float(front_geometry["centerX"])) / float(front_geometry["scale"])
    y = (source[:, 1] - float(front_geometry["centerY"])) / float(front_geometry["scale"])
    lower = np.full(len(rows), -np.inf, dtype=np.float64)
    upper = np.full(len(rows), np.inf, dtype=np.float64)
    supported = np.ones(len(rows), dtype=bool)
    for role in (ViewRole.LEFT45, ViewRole.RIGHT45):
        geometry = estimate.view_geometry[role]
        theta = float(geometry["theta"])
        sine = math.sin(theta)
        cosine = math.cos(theta)
        anchor_predicted = cosine * estimate.local_x + sine * estimate.raw_anchor_depth
        observed = np.asarray(geometry["x"], dtype=np.float64)
        offset = float(np.median(observed - anchor_predicted))

        mask = masks[role] > 127
        row_min = np.full(mask.shape[0], np.nan, dtype=np.float64)
        row_max = np.full(mask.shape[0], np.nan, dtype=np.float64)
        for row_index in np.flatnonzero(np.any(mask, axis=1)):
            row_columns = np.flatnonzero(mask[row_index])
            row_min[row_index] = float(row_columns[0])
            row_max[row_index] = float(row_columns[-1])
        valid_rows = np.flatnonzero(np.isfinite(row_min))
        row_axis = np.arange(mask.shape[0], dtype=np.float64)
        row_min = np.interp(row_axis, valid_rows, row_min[valid_rows])
        row_max = np.interp(row_axis, valid_rows, row_max[valid_rows])
        pixel_y = np.rint(y * float(geometry["scale"]) + float(geometry["centerY"])).astype(
            np.int32
        )
        valid = (pixel_y >= 0) & (pixel_y < mask.shape[0])
        clipped_y = np.clip(pixel_y, 0, mask.shape[0] - 1)
        view_min = (row_min[clipped_y] - float(geometry["centerX"])) / float(geometry["scale"])
        view_max = (row_max[clipped_y] - float(geometry["centerX"])) / float(geometry["scale"])
        depth_a = (view_min - offset - cosine * x) / sine
        depth_b = (view_max - offset - cosine * x) / sine
        lower = np.maximum(lower, np.minimum(depth_a, depth_b))
        upper = np.minimum(upper, np.maximum(depth_a, depth_b))
        supported &= valid
    supported &= np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
    lower_grid = np.full(grid.mask.shape, np.nan, dtype=np.float32)
    upper_grid = np.full(grid.mask.shape, np.nan, dtype=np.float32)
    support_grid = np.zeros(grid.mask.shape, dtype=bool)
    lower_grid[rows, columns] = lower.astype(np.float32)
    upper_grid[rows, columns] = upper.astype(np.float32)
    support_grid[rows, columns] = supported
    return lower_grid, upper_grid, support_grid


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    a = first > 127
    b = second > 127
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / max(union, 1))


def _save_projection_overlay(reference: Path, mask: np.ndarray, destination: Path) -> None:
    rgb = np.asarray(Image.open(reference).convert("RGB")).copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr, contours, -1, (64, 226, 128), 2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), bgr)


def _review_cameras(intake: dict[str, Any]) -> list[CameraRecord]:
    cameras: list[CameraRecord] = []
    yaw_by_role = {ViewRole.FRONT: 0.0, ViewRole.LEFT45: -45.0, ViewRole.RIGHT45: 45.0}
    flip = np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    views = {ViewRole(view["role"]): view for view in intake["views"]}
    for role in REQUIRED_VIEWS:
        yaw = math.radians(yaw_by_role[role])
        rotate_y = np.asarray(
            [
                [math.cos(yaw), 0.0, math.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-math.sin(yaw), 0.0, math.cos(yaw)],
            ]
        )
        rotation_vector, _ = cv2.Rodrigues(flip @ rotate_y)
        view = views[role]
        cameras.append(
            CameraRecord(
                role=role,
                width=int(view["width"]),
                height=int(view["height"]),
                focal_length_px=float(max(view["width"], view["height"]) * 1.7),
                principal_point_px=(float(view["width"]) / 2, float(view["height"]) / 2),
                rotation_vector=tuple(float(value) for value in rotation_vector.reshape(3)),
                translation=(0.0, 0.0, 2.4),
                yaw_deg=yaw_by_role[role],
                pitch_deg=0.0,
                roll_deg=0.0,
            )
        )
    return cameras


def _fit_metrics(
    run_dir: Path,
    config: Face3DConfig,
    intake: dict[str, Any],
    grid: ModelGrid,
    estimate: DepthEstimate,
    raw_field: np.ndarray,
    raw_rear_field: np.ndarray,
) -> dict[str, Any]:
    views = {ViewRole(view["role"]): view for view in intake["views"]}
    per_view: dict[str, Any] = {}
    for role in REQUIRED_VIEWS:
        view = views[role]
        target_mask = cv2.imread(view["mask_path"], cv2.IMREAD_GRAYSCALE)
        if target_mask is None:
            fail("mask-review-required", f"无法读取 mask: {role.value}", stage="pixel-direct")
        if role == ViewRole.FRONT:
            rendered = target_mask.copy()
            nme = 0.0
        else:
            rendered = _projected_silhouette(
                grid,
                raw_field,
                raw_rear_field,
                estimate,
                role,
                target_mask.shape,
            )
            geometry = estimate.view_geometry[role]
            theta = float(geometry["theta"])
            predicted_x = (
                math.cos(theta) * estimate.local_x + math.sin(theta) * estimate.raw_anchor_depth
            )
            observed_x = np.asarray(geometry["x"], dtype=np.float64)
            predicted_x += np.median(observed_x - predicted_x)
            observed_y = np.asarray(geometry["y"], dtype=np.float64)
            diagonal = math.hypot(float(np.ptp(observed_x)), float(np.ptp(observed_y)))
            residual = np.column_stack((predicted_x - observed_x, estimate.local_y - observed_y))
            nme = float(np.mean(np.linalg.norm(residual, axis=1)) / max(diagonal, 1e-6))
        silhouette_iou = _iou(rendered, target_mask)
        _save_projection_overlay(
            Path(view["normalized_path"]),
            rendered,
            run_dir / "overlays" / f"fit-silhouette-{role.value}.png",
        )
        nme_limit = (
            config.acceptance.front_landmark_nme_max
            if role == ViewRole.FRONT
            else config.acceptance.side_landmark_nme_max
        )
        iou_limit = (
            config.acceptance.front_silhouette_iou_min
            if role == ViewRole.FRONT
            else config.acceptance.side_silhouette_iou_min
        )
        per_view[role.value] = {
            "landmarkNME": nme,
            "silhouetteIoU": silhouette_iou,
            "landmarkThreshold": nme_limit,
            "silhouetteThreshold": iou_limit,
            "passed": nme <= nme_limit and silhouette_iou >= iou_limit,
        }
    return {
        "method": "multi-view-correspondence-depth-anchors",
        "sharedIdentity": "source-pixel-coordinate-system",
        "perView": per_view,
        "anchorCount": int(len(estimate.world_anchor_depth)),
        "meanSideAgreement": float(np.mean(estimate.agreement)),
        "passed": all(view["passed"] for view in per_view.values()),
    }


def _implicit_projection(
    grid: ModelGrid,
    grid_points: np.ndarray,
    world_field: np.ndarray,
    estimate: DepthEstimate,
    image_sizes: dict[ViewRole, tuple[int, int]],
    raw_origin: float,
    raw_per_world: float,
) -> ProjectionModel:
    front = estimate.view_geometry[ViewRole.FRONT]
    side_views: dict[ViewRole, dict[str, float]] = {}
    for role in (ViewRole.LEFT45, ViewRole.RIGHT45):
        geometry = estimate.view_geometry[role]
        theta = float(geometry["theta"])
        predicted = math.cos(theta) * estimate.local_x + math.sin(theta) * estimate.raw_anchor_depth
        observed = np.asarray(geometry["x"], dtype=np.float64)
        side_views[role] = {
            "theta": theta,
            "offset": float(np.median(observed - predicted)),
            "scale": float(geometry["scale"]),
            "centerX": float(geometry["centerX"]),
            "centerY": float(geometry["centerY"]),
        }
    center_x = (grid.mask.shape[1] - 1) / 2
    center_y = (grid.mask.shape[0] - 1) / 2
    foreground_rows = np.argwhere(grid.mask > 0)[:, 0]
    head_top_y = (center_y - float(foreground_rows.min())) * grid.pixel_step
    mask_bottom_y = (center_y - float(foreground_rows.max())) * grid.pixel_step
    chin_y = (center_y - float(grid_points[152, 1])) * grid.pixel_step
    face_left = (float(grid_points[234, 0]) - center_x) * grid.pixel_step
    face_right = (float(grid_points[454, 0]) - center_x) * grid.pixel_step
    forehead_end = int(np.clip(round(float(grid_points[168, 1])), 1, grid.mask.shape[0]))
    upper_rows = range(int(foreground_rows.min()), forehead_end)
    upper_widths = [
        int(np.count_nonzero(grid.mask[row] > 0))
        for row in upper_rows
        if np.any(grid.mask[row] > 0)
    ]
    observed_cranium_radius = (
        float(np.quantile(upper_widths, 0.92)) * grid.pixel_step * 0.51 if upper_widths else 0.0
    )
    head_radius_x = max(
        max(abs(face_left), abs(face_right)) * 1.02,
        observed_cranium_radius,
    )
    head_radius_y = max((head_top_y - chin_y) * 0.48, head_radius_x * 1.16)
    head_center_y = head_top_y - head_radius_y
    # Anthropometric closure for the unseen posterior cranium. Keep the front
    # surface fixed and move only the low-confidence back pole rearward.
    head_radius_z = head_radius_x * 1.30
    forehead = np.rint(grid_points[10]).astype(np.int32)
    forehead[0] = np.clip(forehead[0], 0, grid.mask.shape[1] - 1)
    forehead[1] = np.clip(forehead[1], 0, grid.mask.shape[0] - 1)
    forehead_depth = float(world_field[forehead[1], forehead[0]])
    head_center_z = forehead_depth - head_radius_z * 0.98
    neck_bottom_y = max(
        mask_bottom_y,
        chin_y - (head_top_y - chin_y) * 0.30,
    )
    return ProjectionModel(
        crop=grid.crop,
        source_shape=image_sizes[ViewRole.FRONT],
        front_center_x=float(front["centerX"]),
        front_center_y=float(front["centerY"]),
        front_scale=float(front["scale"]),
        raw_origin=raw_origin,
        raw_per_world=raw_per_world,
        side_views=side_views,
        chin_y=chin_y,
        neck_bottom_y=neck_bottom_y,
        head_center_y=head_center_y,
        head_center_z=head_center_z,
        head_radius_x=head_radius_x,
        head_radius_y=head_radius_y,
        head_radius_z=head_radius_z,
    )


def run_pixel_direct(run_dir: Path, config: Face3DConfig) -> dict[str, Any]:
    intake = json.loads((run_dir / "working" / "intake.json").read_text(encoding="utf-8"))
    views = {ViewRole(view["role"]): view for view in intake["views"]}
    landmarks: dict[ViewRole, np.ndarray] = {}
    image_sizes: dict[ViewRole, tuple[int, int]] = {}
    for role in REQUIRED_VIEWS:
        landmark_payload = np.load(views[role]["landmarks_path"])
        landmarks[role] = landmark_payload["all"].astype(np.float64)
        size = landmark_payload["image_size"].astype(int)
        image_sizes[role] = (int(size[0]), int(size[1]))
    masks: dict[ViewRole, np.ndarray] = {}
    for role in REQUIRED_VIEWS:
        mask = cv2.imread(views[role]["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            fail(
                "mask-review-required",
                f"无法读取 mask: {role.value}",
                stage="pixel-direct",
            )
        masks[role] = mask
    ear_observations: dict[ViewRole, EarObservation] = {}
    for role in (ViewRole.LEFT45, ViewRole.RIGHT45):
        side_rgb = np.asarray(Image.open(views[role]["normalized_path"]).convert("RGB"))
        observation, preview = _scan_ear_pixels(
            side_rgb,
            masks[role],
            landmarks[role],
            role,
        )
        ear_observations[role] = observation
        ear_preview = run_dir / "overlays" / f"ear-scan-{role.value}.png"
        ear_preview.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(preview).save(ear_preview, format="PNG", optimize=True)
    front_reference = Path(views[ViewRole.FRONT]["normalized_path"])
    front_rgb = np.asarray(Image.open(front_reference).convert("RGB"))
    front_mask = masks[ViewRole.FRONT]
    grid = _model_grid(front_rgb, front_mask, config.pixel.grid_size)
    grid_points = _landmarks_to_grid(landmarks[ViewRole.FRONT], image_sizes[ViewRole.FRONT], grid)
    feature_class_grid = _feature_classes(
        grid_points, grid.mask.shape, config.pixel.complex_region_radius_pixels
    )
    feature_class_grid[grid.mask == 0] = 0
    feature_protection_grid = _feature_protection(
        grid_points, grid.mask.shape, config.pixel.complex_region_radius_pixels
    )
    feature_protection_grid[grid.mask == 0] = 0
    yaw_degrees = {
        role: float(views[role]["pose_deg"]["yaw"]) if role != ViewRole.FRONT else 0.0
        for role in REQUIRED_VIEWS
    }
    estimate = estimate_depth_anchors(
        landmarks,
        image_sizes,
        yaw_degrees,
        config.pixel.depth_scale_face_width,
    )
    raw_field, _ = _rbf_field(
        grid_points,
        estimate.raw_anchor_depth,
        grid.mask.shape,
        config.pixel.coarse_depth_grid,
        feature_class_grid > 0,
        grid.mask,
    )
    stabilized_world_anchors = _stabilize_feature_depth(estimate.world_anchor_depth)
    world_field, coarse_world = _rbf_field(
        grid_points,
        stabilized_world_anchors,
        grid.mask.shape,
        config.pixel.coarse_depth_grid,
        feature_class_grid > 0,
        grid.mask,
    )
    # Landmark depth is reliable at large scale but contains row/column-sized
    # interpolation ripples. Keep a small amount of measured local residual in
    # the complex regions, then reconstruct eyelids, nose and lips explicitly.
    raw_base = cv2.GaussianBlur(raw_field, (0, 0), 3.0)
    world_base = cv2.GaussianBlur(world_field, (0, 0), 3.0)
    detail_weight = cv2.GaussianBlur((feature_class_grid > 0).astype(np.float32), (0, 0), 2.5)
    raw_field = raw_base + (raw_field - raw_base) * detail_weight * 0.20
    world_field = world_base + (world_field - world_base) * detail_weight * 0.20
    world_field, face_data_mask, canonical_metrics = _canonical_depth_field(
        grid_points,
        stabilized_world_anchors,
        world_field,
        grid.mask,
        config.resolve_asset(config.assets.canonical_face_model),
    )
    semantic_relief = _semantic_relief(
        grid_points,
        grid.mask.shape,
        config.pixel.depth_scale_face_width,
    )
    semantic_relief[grid.mask == 0] = 0
    photometric_relief = _photometric_relief(
        grid.rgb,
        feature_class_grid,
        config.pixel.depth_scale_face_width,
    )
    photometric_relief[grid.mask == 0] = 0
    world_field += semantic_relief + photometric_relief
    raw_low, raw_high = np.quantile(estimate.raw_anchor_depth, (0.03, 0.97))
    raw_per_world = max(
        float(raw_high - raw_low) / config.pixel.depth_scale_face_width,
        1e-6,
    )
    world_field = np.clip(
        world_field,
        -0.05 * config.pixel.depth_scale_face_width,
        1.12 * config.pixel.depth_scale_face_width,
    )
    raw_field = raw_low + world_field * raw_per_world
    template_world_rear = rear_depth_field(
        world_field,
        grid.mask,
        pixel_step=grid.pixel_step,
        minimum_span=config.pixel.depth_scale_face_width * 1.15,
    )
    template_raw_rear = raw_field - (world_field - template_world_rear) * raw_per_world
    interval_lower, interval_upper, interval_supported = _silhouette_depth_interval(
        grid,
        estimate,
        masks,
    )
    minimum_raw_gap = grid.pixel_step * raw_per_world * 0.6
    supported = interval_supported & (interval_upper - interval_lower > minimum_raw_gap)
    raw_column_front = raw_field.copy()
    raw_column_front[supported] = interval_upper[supported]
    raw_column_rear = template_raw_rear.copy()
    raw_column_rear[supported] = interval_lower[supported]
    raw_rear_field = template_raw_rear
    minimum_world_gap = config.pixel.base_thickness_pixels * grid.pixel_step
    interval_rear_world = world_field - (raw_field - interval_lower) / raw_per_world
    usable_rear = supported & (interval_rear_world < world_field - minimum_world_gap)
    raw_rear_field[usable_rear] = interval_lower[usable_rear]
    world_rear_field = world_field - (raw_field - raw_rear_field) / raw_per_world
    maximum_world_thickness = max(
        grid.mask.shape[1] * grid.pixel_step * 0.78,
        config.pixel.depth_scale_face_width * 1.30,
    )
    world_rear_field = np.clip(
        world_rear_field,
        world_field - maximum_world_thickness,
        world_field - minimum_world_gap,
    )
    raw_rear_field = raw_field - (world_field - world_rear_field) * raw_per_world
    column_world_front = world_field + (raw_column_front - raw_field) / raw_per_world
    column_world_rear = world_field - (raw_field - raw_column_rear) / raw_per_world
    column_world_rear = np.maximum(
        column_world_rear,
        column_world_front - maximum_world_thickness,
    )
    column_world_rear = np.minimum(
        column_world_rear,
        column_world_front - minimum_world_gap,
    )
    rows, columns = np.nonzero(grid.mask > 127)
    tree = cKDTree(grid_points)
    distance, nearest = tree.query(np.column_stack((columns, rows)), k=1)
    distance_confidence = np.exp(-distance / max(config.pixel.complex_region_radius_pixels * 4, 1))
    base_confidence = np.clip(
        0.50 + 0.28 * distance_confidence + 0.22 * estimate.agreement[nearest], 0, 1
    ).astype(np.float32)
    confidence_grid = np.zeros(grid.mask.shape, dtype=np.float32)
    confidence_grid[rows, columns] = base_confidence
    working = run_dir / "working"
    working.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        working / "pixel-field.npz",
        depth=world_field.astype(np.float32),
        rear_depth=world_rear_field.astype(np.float32),
        column_front_depth=column_world_front.astype(np.float32),
        column_rear_depth=column_world_rear.astype(np.float32),
        raw_depth=raw_field.astype(np.float32),
        raw_rear_depth=raw_rear_field.astype(np.float32),
        coarse_depth=coarse_world.astype(np.float32),
        mask=grid.mask.astype(np.uint8),
        feature_class=feature_class_grid,
        semantic_relief=semantic_relief.astype(np.float32),
        photometric_relief=photometric_relief.astype(np.float32),
        face_data_mask=face_data_mask.astype(np.uint8),
        source_uv=grid.source_uv_grid,
        crop=np.asarray(grid.crop, dtype=np.int32),
        pixel_step=np.asarray(grid.pixel_step, dtype=np.float32),
    )
    cameras = _review_cameras(intake)
    atomic_write_json(
        working / "cameras.json",
        {"schemaVersion": 1, "cameras": [camera.model_dump(mode="json") for camera in cameras]},
    )
    fit_metrics = _fit_metrics(
        run_dir,
        config,
        intake,
        grid,
        estimate,
        raw_column_front,
        raw_column_rear,
    )
    atomic_write_json(working / "fit-metrics.json", fit_metrics)
    projection = _implicit_projection(
        grid,
        grid_points,
        world_field,
        estimate,
        image_sizes,
        float(raw_low),
        raw_per_world,
    )
    mesh_result = build_multiview_implicit_mesh(
        world_field,
        grid.mask,
        face_data_mask,
        feature_protection_grid,
        masks,
        projection,
        eye_models=_open_eye_models(grid_points, world_field, grid),
        rear_hint=template_world_rear,
        pixel_step=grid.pixel_step,
        resolution=config.sdf.resolution,
        padding_fraction=config.sdf.padding_fraction,
        target_triangles=config.mesh.target_triangles,
        minimum_triangles=config.mesh.minimum_triangles,
        maximum_triangles=config.mesh.maximum_triangles,
        hausdorff_voxels_max=config.acceptance.hausdorff_voxels_max,
    )
    shell = _pixel_surface_shell(
        grid,
        (
            mesh_result.front_surface_depth
            if mesh_result.front_surface_depth is not None
            else world_field
        ),
        feature_class_grid,
        confidence_grid,
        cell_fill_ratio=config.pixel.cell_fill_ratio,
        maximum_cells=config.pixel.maximum_cells,
        surface_mesh=mesh_result.smooth_mesh,
    )
    instance_count = len(shell.positions)
    if instance_count > config.pixel.maximum_cells:
        fail(
            "pixel-count-exceeded",
            "直接 3D Pixel 数量超过配置上限",
            stage="pixel-direct",
            details={"measured": instance_count, "limit": config.pixel.maximum_cells},
        )
    export_pixel_instances(
        shell.positions,
        shell.scales,
        shell.rotations,
        shell.pixel_codes,
        shell.source_uv,
        shell.positions[:, 2],
        shell.feature_class,
        shell.confidence,
        shell.source_bits,
        run_dir / "models" / "voxels.glb",
    )
    pixel_binary = write_pixel_records(
        run_dir / "pixels" / "pixels.bin",
        run_dir / "pixels" / "schema.json",
        model_uv=shell.model_uv,
        source_uv=shell.source_uv,
        pixel_codes=shell.pixel_codes,
        positions=shell.positions,
        thickness=shell.thickness,
        confidence=shell.confidence,
        source_bits=shell.source_bits,
        feature_class=shell.feature_class,
        grid_size=(grid.mask.shape[1], grid.mask.shape[0]),
        crop=grid.crop,
        source_sha256=sha256_file(front_reference),
    )
    pixel_metrics = {
        "representation": "direct-front-pixels-plus-multiview-surface",
        "resolution": [int(grid.mask.shape[1]), int(grid.mask.shape[0])],
        "voxelSize": grid.pixel_step,
        "instanceCount": instance_count,
        "finite": bool(
            np.all(np.isfinite(shell.positions)) and np.all(np.isfinite(shell.confidence))
        ),
        "isolatedVoxelCount": 0,
        "surfaceCellCoverage": shell.surface_coverage,
        "frontSurfaceSnapCount": shell.front_surface_snap_count,
        "frontSurfaceMaxDistancePixels": shell.front_surface_max_distance_pixels,
        "meanConfidence": float(np.mean(shell.confidence)),
        "minimumConfidence": float(np.min(shell.confidence)),
        "templateInferredCount": int(np.count_nonzero(shell.source_bits & 8)),
        "silhouetteSupportedCount": int(np.count_nonzero(shell.source_bits & (2 | 4))),
        "surfaceLayers": shell.layer_counts,
        "cellExtent": {
            "minimum": float(np.min(shell.thickness)),
            "median": float(np.median(shell.thickness)),
            "maximum": float(np.max(shell.thickness)),
        },
        "complexPixelCount": int(np.count_nonzero(shell.feature_class)),
        "simpleInterpolatedPixelCount": int(np.count_nonzero(shell.feature_class == 0)),
        "traceabilityComplete": bool(
            pixel_binary["records"] == instance_count and np.all(shell.source_bits > 0)
        ),
        "depthRegularization": canonical_metrics,
        "earReconstruction": {
            "mode": "smooth-anatomical-support-with-direct-side-view-skin",
            "views": {
                role.value: {
                    "sourcePixels": observation.source_pixels,
                    "reliefMinimum": float(np.min(observation.relief)),
                    "reliefMaximum": float(np.max(observation.relief)),
                }
                for role, observation in ear_observations.items()
            },
        },
        "pixelBinary": pixel_binary,
        "passed": bool(
            instance_count <= config.pixel.maximum_cells
            and np.all(np.isfinite(shell.positions))
            and pixel_binary["records"] == instance_count
            and shell.surface_coverage >= 0.99
            and shell.front_surface_max_distance_pixels <= 2.0
        ),
    }
    atomic_write_json(working / "sdf-metrics.json", pixel_metrics)
    mesh_metrics = mesh_result.metrics
    mesh_metrics["passed"] = bool(
        mesh_metrics["passed"]
        and mesh_metrics["featureDriftVoxels"] <= config.acceptance.feature_drift_voxels_max
        and mesh_metrics["normalVarianceReduction"]
        >= config.acceptance.normal_variance_reduction_min
        and mesh_metrics["maximumSilhouetteIoUDrop"] <= config.acceptance.silhouette_iou_drop_max
    )
    np.savez_compressed(
        working / "smooth-mesh.npz",
        vertices=np.asarray(mesh_result.smooth_mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh_result.smooth_mesh.faces, dtype=np.int32),
    )
    atomic_write_json(working / "mesh-metrics.json", mesh_metrics)
    export_neutral_mesh(mesh_result.raw_mesh, run_dir / "models" / "raw-isosurface.glb")
    export_neutral_mesh(mesh_result.smooth_mesh, run_dir / "models" / "smooth.glb")
    if config.output.geometry_only:
        skin_metrics: dict[str, Any] = {
            "schemaVersion": 1,
            "enabled": False,
            "geometryOnly": True,
            "skipped": True,
            "reason": "geometry-only-experiment",
            "photoSkinProjectionUsed": False,
            "passed": True,
        }
    else:
        skin_result = build_skin_component(
            run_dir,
            mesh_result.smooth_mesh,
            cameras,
            uv_albedo_source_path=config.resolve_asset(config.skin.uv_albedo_source),
            micro_source_path=config.resolve_asset(config.skin.micro_albedo_source),
            atlas_resolution=config.skin.atlas_resolution,
            detail_resolution=config.skin.detail_resolution,
            minimum_observed_vertex_fraction=config.skin.minimum_observed_vertex_fraction,
            jpeg_quality=config.skin.jpeg_quality,
            seed=config.seed,
        )
        skin_metrics = skin_result.metrics
    atomic_write_json(working / "skin-metrics.json", skin_metrics)

    failures: list[dict[str, Any]] = []
    if not fit_metrics["passed"]:
        failures.append({"gate": "B-fit", "metrics": fit_metrics["perView"]})
    if not pixel_metrics["passed"]:
        failures.append({"gate": "C-3d-pixel", "metrics": pixel_metrics})
    if not mesh_metrics["passed"]:
        failures.append({"gate": "D-smooth-mesh", "metrics": mesh_metrics})
    if not config.output.geometry_only and not skin_metrics["passed"]:
        failures.append({"gate": "E-skin-atlas", "metrics": skin_metrics})
    if failures:
        fail(
            "pixel-direct-gate-failed",
            "像素直转结果未通过自动门禁",
            stage="pixel-direct",
            details={"failures": failures},
        )
    return {
        "mode": "pixel-direct",
        "instanceCount": instance_count,
        "complexPixelCount": pixel_metrics["complexPixelCount"],
        "triangles": mesh_metrics["triangles"],
        "meanConfidence": pixel_metrics["meanConfidence"],
        "geometryOnly": config.output.geometry_only,
        "photoSkinProjectionUsed": not config.output.geometry_only,
        "skinObservedVertexFraction": skin_metrics.get("observedVertexFraction"),
    }
