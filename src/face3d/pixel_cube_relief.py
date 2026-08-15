from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from face3d.errors import fail
from face3d.glb import export_instanced_voxels
from face3d.io import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from face3d.pixel_cube import (
    FACE_BITS,
    TEMPLATE_INFERRED_SOURCE_BIT,
    PixelCubeSpec,
    PixelCuboidSpec,
    cell_centers_xyz,
    cell_face_masks_xyz,
    surface_cell_indices_xyz,
)
from face3d.stages.intake import FACE_OVAL, _detect, face_landmarker
from face3d.stages.pixel_direct import (
    EYE_INDICES,
    MOUTH_INDICES,
    _canonical_depth_field,
    _feature_classes,
    _photometric_relief,
    _rbf_field,
    _semantic_relief,
    _stabilize_feature_depth,
)

FRONT_OBSERVED_SOURCE_BIT = 1
FRONT_RELIEF_SOURCE_BITS = FRONT_OBSERVED_SOURCE_BIT | TEMPLATE_INFERRED_SOURCE_BIT

@dataclass(frozen=True, slots=True)
class FrontFaceReliefSpec:
    cube: PixelCubeSpec | PixelCuboidSpec = field(default_factory=PixelCuboidSpec)
    front_cells_xy: tuple[int, int] = (344, 512)
    max_inset_m: float = 0.03
    border_rim_cells_xy: tuple[float, float] = (4.0, 4.0)
    observed_confidence: float = 0.42
    coarse_depth_grid: int = 80
    complex_region_radius_pixels: float = 20.0
    depth_scale_face_width: float = 0.38
    maximum_cells: int = 360_000
    scan_config_path: Path | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_inset_m) or self.max_inset_m <= 0:
            raise ValueError("max_inset_m must be finite and positive")
        maximum_safe_inset = self.dimensions_m[2] - self.front_cell_pitch_m
        if self.max_inset_m > maximum_safe_inset:
            raise ValueError("max_inset_m would cross the opposite cuboid surface")
        coarse_x, coarse_y, _ = self.coarse_cells_xyz
        front_x, front_y = self.front_cells_xy
        if front_x < coarse_x or front_y < coarse_y:
            raise ValueError("front_cells_xy must not be smaller than the cuboid front grid")
        if front_x % coarse_x or front_y % coarse_y:
            raise ValueError("front_cells_xy must be integer subdivisions of cuboid cells")
        for rim, subdivision, resolution in zip(
            self.border_rim_cells_xy,
            self.front_subdivision_xy,
            self.front_cells_xy,
            strict=True,
        ):
            if (
                not math.isfinite(rim)
                or rim < subdivision
                or rim >= resolution / 2
            ):
                raise ValueError("border_rim_cells_xy must preserve the coarse outer frame")
        if (
            not math.isfinite(self.observed_confidence)
            or not 0 <= self.observed_confidence <= 1
        ):
            raise ValueError("observed_confidence must be in [0, 1]")
        if not 8 <= self.coarse_depth_grid < max(self.front_cells_xy):
            raise ValueError("coarse_depth_grid must be smaller than the front scan grid")
        if (
            not math.isfinite(self.complex_region_radius_pixels)
            or self.complex_region_radius_pixels <= 0
        ):
            raise ValueError("complex_region_radius_pixels must be finite and positive")
        if (
            not math.isfinite(self.depth_scale_face_width)
            or not 0 < self.depth_scale_face_width <= 1
        ):
            raise ValueError("depth_scale_face_width must be in (0, 1]")
        if self.maximum_cells < self.hybrid_surface_cell_count:
            raise ValueError("hybrid surface exceeds maximum_cells")

    @property
    def coarse_cells_xyz(self) -> tuple[int, int, int]:
        if isinstance(self.cube, PixelCuboidSpec):
            return self.cube.cells_xyz
        return (self.cube.cells_per_edge,) * 3

    @property
    def dimensions_m(self) -> tuple[float, float, float]:
        if isinstance(self.cube, PixelCuboidSpec):
            return self.cube.dimensions_m
        return (self.cube.side_length_m,) * 3

    @property
    def coarse_cell_pitch_m(self) -> float:
        return self.cube.cell_pitch_m

    @property
    def front_subdivision_xy(self) -> tuple[int, int]:
        coarse_x, coarse_y, _ = self.coarse_cells_xyz
        front_x, front_y = self.front_cells_xy
        return front_x // coarse_x, front_y // coarse_y

    @property
    def front_cell_pitch_m(self) -> float:
        width, height, _ = self.dimensions_m
        front_x, front_y = self.front_cells_xy
        pitch_x = width / front_x
        pitch_y = height / front_y
        if not math.isclose(pitch_x, pitch_y, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("front scan pixels must have the same X/Y pitch")
        return pitch_x

    @property
    def front_shape_yx(self) -> tuple[int, int]:
        front_x, front_y = self.front_cells_xy
        return front_y, front_x

    @property
    def hybrid_surface_cell_count(self) -> int:
        if self.front_subdivision_xy == (1, 1):
            return self.cube.surface_cell_count
        coarse_x, coarse_y, _ = self.coarse_cells_xyz
        front_x, front_y = self.front_cells_xy
        subdivision_x, subdivision_y = self.front_subdivision_xy
        coarse_front_interior = (coarse_x - 2) * (coarse_y - 2)
        fine_front_interior = (front_x - 2 * subdivision_x) * (
            front_y - 2 * subdivision_y
        )
        return self.cube.surface_cell_count - coarse_front_interior + fine_front_interior


@dataclass(slots=True)
class FrontReliefScan:
    indentation_m: np.ndarray
    feature_relief_m: np.ndarray
    face_mask: np.ndarray
    feature_class: np.ndarray
    curvature_inset_fraction: np.ndarray
    source_uv_normalized: np.ndarray
    landmarks_normalized: np.ndarray
    mapping: dict[str, Any]
    frontal_metrics: dict[str, float]
    feature_metrics: dict[str, float]
    depth_statistics: dict[str, float]


@dataclass(slots=True)
class FrontReliefGeometry:
    grid_xyz: np.ndarray
    grid_shape_xyz: np.ndarray
    cell_pitch_m: np.ndarray
    face_masks: np.ndarray
    base_positions: np.ndarray
    positions: np.ndarray
    confidence: np.ndarray
    source_bits: np.ndarray
    front_scan_layer: np.ndarray


def _validate_landmarks(landmarks: np.ndarray) -> np.ndarray:
    result = np.asarray(landmarks, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 455 or result.shape[1] != 3:
        raise ValueError("landmarks must contain at least 455 normalized XYZ points")
    if not np.all(np.isfinite(result)):
        raise ValueError("landmarks must be finite")
    return result


def _frontal_metrics(landmarks: np.ndarray) -> dict[str, float]:
    oval = landmarks[np.asarray(FACE_OVAL), :2]
    face_width = max(float(np.ptp(oval[:, 0])), 1e-8)
    face_height = max(float(np.ptp(oval[:, 1])), 1e-8)
    cheek_midpoint_x = float((landmarks[234, 0] + landmarks[454, 0]) / 2)
    return {
        "noseHorizontalOffsetFaceWidths": float(
            abs(landmarks[1, 0] - cheek_midpoint_x) / face_width
        ),
        "eyeTiltFaceHeights": float(abs(landmarks[33, 1] - landmarks[263, 1]) / face_height),
        "faceWidthNormalized": face_width,
        "faceHeightNormalized": face_height,
    }


def _target_mapping(
    landmarks: np.ndarray,
    image_shape: tuple[int, int],
    spec: FrontFaceReliefSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image_height, image_width = image_shape
    oval = landmarks[np.asarray(FACE_OVAL), :2]
    source_min = oval.min(axis=0)
    source_max = oval.max(axis=0)
    source_span = np.maximum(source_max - source_min, 1e-8)

    face_width_pixels = source_span[0] * image_width
    face_height_pixels = source_span[1] * image_height
    front_width, front_height = spec.front_cells_xy
    rim_x, rim_y = spec.border_rim_cells_xy
    available_x = front_width - 2 * rim_x
    available_y = front_height - 2 * rim_y
    scale = min(available_x / face_width_pixels, available_y / face_height_pixels)
    target_width = face_width_pixels * scale
    target_height = face_height_pixels * scale
    target_min = np.asarray(
        [
            (front_width - target_width) / 2,
            (front_height - target_height) / 2,
        ],
        dtype=np.float64,
    )
    target_span = np.asarray([target_width, target_height], dtype=np.float64)
    mapped = target_min + (landmarks[:, :2] - source_min) / source_span * target_span
    mapping = {
        "sourceFaceBoundsNormalized": [
            round(float(source_min[0]), 9),
            round(float(source_min[1]), 9),
            round(float(source_max[0]), 9),
            round(float(source_max[1]), 9),
        ],
        "targetFaceBoundsCells": [
            round(float(target_min[0]), 9),
            round(float(target_min[1]), 9),
            round(float(target_min[0] + target_span[0]), 9),
            round(float(target_min[1] + target_span[1]), 9),
        ],
        "preserveAspectRatio": True,
        "targetYAxis": "image-down; converted to cube +Y during sampling",
        "borderRimCellsXY": [rim_x, rim_y],
    }
    return mapped, np.column_stack((source_min, source_span)), mapping


def _gaussian_feature(
    shape: tuple[int, int],
    center: np.ndarray,
    radius_x: float,
    radius_y: float,
) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float32)
    radius_x = max(float(radius_x), 0.75)
    radius_y = max(float(radius_y), 0.75)
    return np.exp(
        -0.5
        * (
            ((xx - float(center[0])) / radius_x) ** 2
            + ((yy - float(center[1])) / radius_y) ** 2
        )
    ).astype(np.float32)


def _blurred_feature_polygon(
    points: np.ndarray,
    indices: tuple[int, ...],
    shape: tuple[int, int],
    sigma: float,
) -> np.ndarray:
    layer = np.zeros(shape, dtype=np.float32)
    selected = np.rint(points[np.asarray(indices, dtype=np.int32)]).astype(np.int32)
    if len(selected) >= 3:
        cv2.fillConvexPoly(layer, cv2.convexHull(selected), 1.0)
    layer = cv2.GaussianBlur(layer, (0, 0), max(float(sigma), 0.45))
    return layer / max(float(layer.max()), 1e-6)


def _blurred_feature_line(
    points: np.ndarray,
    indices: tuple[int, ...],
    shape: tuple[int, int],
    sigma: float,
    *,
    closed: bool = False,
) -> np.ndarray:
    layer = np.zeros(shape, dtype=np.float32)
    selected = np.rint(points[np.asarray(indices, dtype=np.int32)]).astype(np.int32)
    if len(selected) >= 2:
        cv2.polylines(
            layer,
            [selected],
            closed,
            1.0,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    layer = cv2.GaussianBlur(layer, (0, 0), max(float(sigma), 0.45))
    return layer / max(float(layer.max()), 1e-6)


def _strong_feature_relief(
    points: np.ndarray,
    grid_rgb: np.ndarray,
    face_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Create explicit millimetre-scale eye, nose and mouth relief.

    Positive values move a front pixel deeper into the cube; negative values
    preserve a projected form nearer the original Z+ plane.
    """
    shape = face_mask.shape
    relief = np.zeros(shape, dtype=np.float32)
    gray = cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    masks: dict[str, np.ndarray] = {}

    for side, eye_indices in zip(
        ("left", "right"),
        (EYE_INDICES[:16], EYE_INDICES[16:]),
        strict=True,
    ):
        eye_points = points[np.asarray(eye_indices)]
        eye_width = max(float(np.ptp(eye_points[:, 0])), 3.0)
        eye_height = max(float(np.ptp(eye_points[:, 1])), eye_width * 0.12)
        center = (points[eye_indices[0]] + points[eye_indices[8]]) / 2
        socket = _gaussian_feature(
            shape,
            center,
            eye_width * 0.60,
            max(eye_width * 0.31, eye_height * 1.55),
        )
        globe = _gaussian_feature(
            shape,
            center,
            eye_width * 0.31,
            max(eye_width * 0.12, eye_height * 0.60),
        )
        socket_ring = np.clip(socket - globe * 0.90, 0.0, 1.0)
        aperture = _blurred_feature_polygon(
            points,
            eye_indices,
            shape,
            eye_width * 0.012,
        )
        outline = _blurred_feature_line(
            points,
            eye_indices,
            shape,
            eye_width * 0.018,
            closed=True,
        )
        selected = aperture > 0.28
        if np.count_nonzero(selected) >= 6:
            bright = float(np.quantile(gray[selected], 0.85))
            dark = float(np.quantile(gray[selected], 0.10))
            darkness = np.clip((bright - gray) / max(bright - dark, 0.08), 0.0, 1.0)
        else:
            darkness = np.zeros(shape, dtype=np.float32)
        iris = aperture * darkness

        relief += socket_ring * 0.0040
        relief -= globe * aperture * 0.0024
        relief += outline * 0.0016
        relief += iris * 0.00055
        masks[f"eyeSocket{side.title()}"] = socket_ring
        masks[f"eyeGlobe{side.title()}"] = globe * aperture
        masks[f"iris{side.title()}"] = iris

    for brow in ((70, 63, 105, 66, 107), (336, 296, 334, 293, 300)):
        brow_width = max(float(np.ptp(points[np.asarray(brow), 0])), 3.0)
        relief -= _blurred_feature_line(
            points,
            brow,
            shape,
            brow_width * 0.055,
        ) * 0.0008

    nose_width = max(float(abs(points[327, 0] - points[98, 0])), 3.0)
    nose_bridge = _blurred_feature_line(
        points,
        (168, 6, 197, 195, 5, 4, 1),
        shape,
        nose_width * 0.10,
    )
    nose_tip = _gaussian_feature(
        shape,
        points[1],
        nose_width * 0.30,
        nose_width * 0.25,
    )
    nose_wing = _blurred_feature_polygon(
        points,
        (98, 97, 2, 326, 327, 4),
        shape,
        nose_width * 0.035,
    )
    relief -= nose_bridge * 0.0030
    relief -= nose_tip * 0.0014
    relief -= nose_wing * 0.0009

    nostril_mask = np.zeros(shape, dtype=np.float32)
    nostril_core = np.zeros(shape, dtype=np.float32)
    for nostril in (98, 327):
        local = _gaussian_feature(
            shape,
            points[nostril],
            nose_width * 0.11,
            nose_width * 0.075,
        )
        local *= 0.70 + np.clip(0.62 - gray, 0.0, 0.42) / 0.42 * 0.55
        nostril_mask = np.maximum(nostril_mask, local)
        nostril_core = np.maximum(nostril_core, local**3)
    relief += nostril_mask * 0.0042
    masks["noseProjection"] = nose_tip**2
    masks["noseWing"] = np.clip(nose_wing - nostril_mask, 0.0, 1.0)
    masks["nostril"] = nostril_core

    face_width = max(float(np.ptp(points[np.asarray(FACE_OVAL), 0])), 12.0)
    face_height = max(float(np.ptp(points[np.asarray(FACE_OVAL), 1])), 12.0)
    cheek_left = _gaussian_feature(
        shape,
        points[205],
        face_width * 0.115,
        face_height * 0.085,
    )
    cheek_right = _gaussian_feature(
        shape,
        points[425],
        face_width * 0.115,
        face_height * 0.085,
    )
    cheek = np.maximum(cheek_left, cheek_right)
    relief -= cheek * 0.0010
    masks["cheekLeft"] = cheek_left
    masks["cheekRight"] = cheek_right

    mouth_width = max(float(abs(points[291, 0] - points[61, 0])), 3.0)
    lip = _blurred_feature_polygon(
        points,
        MOUTH_INDICES,
        shape,
        mouth_width * 0.030,
    )
    upper_seam = (61, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 291)
    lower_seam = (61, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291)
    seam = np.maximum(
        _blurred_feature_line(
            points,
            upper_seam,
            shape,
            mouth_width * 0.016,
        ),
        _blurred_feature_line(
            points,
            lower_seam,
            shape,
            mouth_width * 0.016,
        ),
    )
    corners = np.maximum(
        _gaussian_feature(shape, points[61], mouth_width * 0.045, mouth_width * 0.040),
        _gaussian_feature(shape, points[291], mouth_width * 0.045, mouth_width * 0.040),
    )
    relief -= lip * 0.0017
    relief += seam * 0.0030
    relief += corners * 0.0015
    masks["lip"] = np.clip(lip - seam * 0.82, 0.0, 1.0)
    masks["mouthSeam"] = seam

    chin = _gaussian_feature(
        shape,
        points[152],
        face_width * 0.10,
        face_height * 0.065,
    )
    relief -= chin * 0.0007

    relief = cv2.GaussianBlur(relief, (0, 0), max(shape[0] / 1024.0, 0.45))
    relief[~face_mask] = 0.0
    for layer in masks.values():
        layer[~face_mask] = 0.0
    return relief.astype(np.float32), masks


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 1e-8:
        return 0.0
    return float(np.sum(values * weights) / total)


def _feature_metrics(
    indentation: np.ndarray,
    feature_relief: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, float]:
    left_socket = _weighted_mean(indentation, masks["eyeSocketLeft"])
    right_socket = _weighted_mean(indentation, masks["eyeSocketRight"])
    left_globe = _weighted_mean(indentation, masks["eyeGlobeLeft"])
    right_globe = _weighted_mean(indentation, masks["eyeGlobeRight"])
    cheek_left = _weighted_mean(indentation, masks["cheekLeft"])
    cheek_right = _weighted_mean(indentation, masks["cheekRight"])
    nose = _weighted_mean(indentation, masks["noseProjection"])
    nostril = _weighted_mean(indentation, masks["nostril"])
    nose_wing = _weighted_mean(indentation, masks["noseWing"])
    lip = _weighted_mean(indentation, masks["lip"])
    mouth_seam = _weighted_mean(indentation, masks["mouthSeam"])
    return {
        "leftEyeSocketInsetMeters": left_socket,
        "rightEyeSocketInsetMeters": right_socket,
        "leftEyeGlobeInsetMeters": left_globe,
        "rightEyeGlobeInsetMeters": right_globe,
        "meanEyeSocketContrastMeters": (
            (left_socket - left_globe) + (right_socket - right_globe)
        )
        / 2,
        "noseProjectionInsetMeters": nose,
        "meanCheekInsetMeters": (cheek_left + cheek_right) / 2,
        "noseProjectionContrastMeters": (cheek_left + cheek_right) / 2 - nose,
        "nostrilInsetMeters": nostril,
        "noseWingInsetMeters": nose_wing,
        "nostrilContrastMeters": nostril - nose_wing,
        "lipInsetMeters": lip,
        "mouthSeamInsetMeters": mouth_seam,
        "mouthSeamContrastMeters": mouth_seam - lip,
        "featureReliefMinMeters": float(np.min(feature_relief)),
        "featureReliefMaxMeters": float(np.max(feature_relief)),
    }


def estimate_front_relief(
    rgb: np.ndarray,
    landmarks: np.ndarray,
    spec: FrontFaceReliefSpec | None = None,
    canonical_face_model: Path | None = None,
) -> FrontReliefScan:
    spec = spec or FrontFaceReliefSpec()
    landmarks = _validate_landmarks(landmarks)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be an RGB uint8 image")
    image_height, image_width = rgb.shape[:2]
    mapped, source_transform, mapping = _target_mapping(
        landmarks,
        (image_height, image_width),
        spec,
    )

    n_x, n_y = spec.front_cells_xy
    columns, rows = np.meshgrid(np.arange(n_x), np.arange(n_y), indexing="xy")
    target_x = columns.astype(np.float64) + 0.5
    target_y_image = rows.astype(np.float64) + 0.5
    query = np.column_stack((target_x.reshape(-1), target_y_image.reshape(-1)))

    source_min = source_transform[:, 0]
    source_span = source_transform[:, 1]
    target_bounds = np.asarray(mapping["targetFaceBoundsCells"], dtype=np.float64)
    target_min = target_bounds[:2]
    target_span = target_bounds[2:] - target_bounds[:2]
    source_uv = source_min + (query - target_min) / target_span * source_span
    source_uv = np.clip(source_uv, 0.0, 1.0).reshape(n_y, n_x, 2)

    oval_polygon = mapped[np.asarray(FACE_OVAL)].astype(np.float32)
    face_flat = np.asarray(
        [
            cv2.pointPolygonTest(oval_polygon, (float(point[0]), float(point[1])), False)
            >= 0
            for point in query
        ],
        dtype=bool,
    )
    face_mask_image = face_flat.reshape(n_y, n_x)
    rim_x = math.ceil(spec.border_rim_cells_xy[0])
    rim_y = math.ceil(spec.border_rim_cells_xy[1])
    face_mask_image[:rim_y, :] = False
    face_mask_image[-rim_y:, :] = False
    face_mask_image[:, :rim_x] = False
    face_mask_image[:, -rim_x:] = False

    mask_u8 = np.where(face_mask_image, 255, 0).astype(np.uint8)
    feature_class_image = _feature_classes(
        mapped,
        mask_u8.shape,
        spec.complex_region_radius_pixels,
    )
    feature_class_image[~face_mask_image] = 0

    map_x = (source_uv[:, :, 0] * (image_width - 1)).astype(np.float32)
    map_y = (source_uv[:, :, 1] * (image_height - 1)).astype(np.float32)
    grid_rgb = cv2.remap(
        rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    anchor_count = min(len(landmarks), 468)
    anchor_depth = _stabilize_feature_depth(-landmarks[:anchor_count, 2])
    relative_field, coarse_field = _rbf_field(
        mapped[:anchor_count],
        anchor_depth,
        mask_u8.shape,
        spec.coarse_depth_grid,
        feature_class_image > 0,
        mask_u8,
    )
    scan_scale = max(n_x, n_y) / 256
    smooth_sigma = max(0.6, 3.0 * scan_scale)
    detail_sigma = max(0.6, 2.5 * scan_scale)
    smooth_field = cv2.GaussianBlur(relative_field, (0, 0), smooth_sigma)
    detail_weight = cv2.GaussianBlur(
        (feature_class_image > 0).astype(np.float32),
        (0, 0),
        detail_sigma,
    )
    relative_field = smooth_field + (relative_field - smooth_field) * detail_weight * 0.20

    canonical_metrics: dict[str, Any] = {
        "used": False,
        "reason": "canonical-model-not-requested",
    }
    if canonical_face_model is not None:
        relative_field, _, canonical_metrics = _canonical_depth_field(
            mapped[:anchor_count],
            anchor_depth,
            relative_field,
            mask_u8,
            Path(canonical_face_model),
        )

    semantic_relief = _semantic_relief(
        mapped,
        mask_u8.shape,
        spec.depth_scale_face_width,
    )
    photometric_relief = _photometric_relief(
        grid_rgb,
        feature_class_image,
        spec.depth_scale_face_width,
    )
    semantic_relief[~face_mask_image] = 0
    photometric_relief[~face_mask_image] = 0
    legacy_relief = semantic_relief + photometric_relief
    relative_field += legacy_relief * 0.60

    field_values = relative_field[face_mask_image]
    field_low, field_high = np.quantile(field_values, (0.02, 0.98))
    field_span = max(float(field_high - field_low), 1e-8)
    landmark_inset = 1.0 - np.clip((relative_field - field_low) / field_span, 0.0, 1.0)

    distance = cv2.distanceTransform(face_mask_image.astype(np.uint8), cv2.DIST_L2, 5)
    distance_scale = max(float(np.quantile(distance[face_mask_image], 0.99)), 1.0)
    inward_distance = np.clip(distance / distance_scale, 0.0, 1.0)
    radial = 1.0 - inward_distance
    ellipsoid_inset = 1.0 - np.sqrt(np.clip(1.0 - radial**2, 0.0, 1.0))

    combined = np.clip(0.42 * ellipsoid_inset + 0.58 * landmark_inset, 0.0, 1.0)
    base_inset = spec.max_inset_m * (0.030 + 0.78 * combined)
    legacy_feature_offset = (
        -legacy_relief
        / spec.depth_scale_face_width
        * spec.max_inset_m
        * 1.35
    )
    explicit_feature_relief, feature_masks = _strong_feature_relief(
        mapped,
        grid_rgb,
        face_mask_image,
    )
    inset_image = base_inset + legacy_feature_offset + explicit_feature_relief
    eroded = cv2.erode(face_mask_image.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    contour = face_mask_image & ~eroded
    inset_image[contour] = np.maximum(inset_image[contour], spec.max_inset_m * 0.92)
    inset_image = np.clip(inset_image, spec.max_inset_m * 0.020, spec.max_inset_m)
    inset_image[~face_mask_image] = 0.0

    feature_metrics = _feature_metrics(
        inset_image,
        explicit_feature_relief,
        feature_masks,
    )

    inset = np.flipud(inset_image)
    feature_relief = np.flipud(explicit_feature_relief)
    face_mask = np.flipud(face_mask_image)
    feature_class = np.flipud(feature_class_image)
    curvature_inset = np.flipud(ellipsoid_inset)
    source_uv = np.flipud(source_uv)
    affected = inset[face_mask]

    def landmark_inset_m(index: int) -> float:
        x = int(np.clip(round(mapped[index, 0] - 0.5), 0, n_x - 1))
        y_image = int(np.clip(round(mapped[index, 1] - 0.5), 0, n_y - 1))
        return float(inset_image[y_image, x])

    depth_statistics = {
        "scanGridShapeYX": [n_y, n_x],
        "coarseDepthGrid": spec.coarse_depth_grid,
        "complexRegionRadiusPixels": spec.complex_region_radius_pixels,
        "depthScaleFaceWidth": spec.depth_scale_face_width,
        "canonicalDepthUsed": bool(canonical_metrics.get("used", False)),
        "canonicalDepthCorrelation": float(canonical_metrics.get("correlation", 0.0)),
        "coarseFieldRange": float(np.ptp(coarse_field)),
        "noseTipInsetMeters": landmark_inset_m(1),
        "leftCheekInsetMeters": landmark_inset_m(205),
        "rightCheekInsetMeters": landmark_inset_m(425),
        "leftEyeInsetMeters": landmark_inset_m(33),
        "rightEyeInsetMeters": landmark_inset_m(263),
        "mouthInsetMeters": landmark_inset_m(13),
        "requestedMaxInsetMeters": spec.max_inset_m,
        "actualMinInsetMeters": float(np.min(affected)) if len(affected) else 0.0,
        "actualMaxInsetMeters": float(np.max(affected)) if len(affected) else 0.0,
        "affectedFrontCellCount": int(np.count_nonzero(face_mask)),
    }
    return FrontReliefScan(
        indentation_m=inset.astype(np.float64),
        feature_relief_m=feature_relief.astype(np.float32),
        face_mask=face_mask,
        feature_class=feature_class.astype(np.uint8),
        curvature_inset_fraction=curvature_inset.astype(np.float32),
        source_uv_normalized=source_uv.astype(np.float64),
        landmarks_normalized=landmarks,
        mapping=mapping,
        frontal_metrics=_frontal_metrics(landmarks),
        feature_metrics=feature_metrics,
        depth_statistics=depth_statistics,
    )


def apply_front_relief(
    scan: FrontReliefScan,
    spec: FrontFaceReliefSpec,
) -> FrontReliefGeometry:
    coarse_shape = spec.coarse_cells_xyz
    coarse_x, coarse_y, coarse_z = coarse_shape
    front_x, front_y = spec.front_cells_xy
    subdivision_x, subdivision_y = spec.front_subdivision_xy
    coarse_indices = surface_cell_indices_xyz(coarse_shape)
    coarse_positions = cell_centers_xyz(
        coarse_indices,
        coarse_shape,
        spec.coarse_cell_pitch_m,
    )
    coarse_face_masks = cell_face_masks_xyz(coarse_indices, coarse_shape)

    if (subdivision_x, subdivision_y) == (1, 1):
        grid_xyz = coarse_indices
        base_positions = coarse_positions
        face_masks = coarse_face_masks
        front_scan_layer = coarse_indices[:, 2] == coarse_z - 1
        grid_shape_xyz = np.tile(
            np.asarray(coarse_shape, dtype=np.uint16),
            (len(grid_xyz), 1),
        )
        cell_pitch = np.full(len(grid_xyz), spec.coarse_cell_pitch_m, dtype=np.float32)
    else:
        coarse_front_interior = (
            (coarse_indices[:, 2] == coarse_z - 1)
            & (coarse_indices[:, 0] > 0)
            & (coarse_indices[:, 0] < coarse_x - 1)
            & (coarse_indices[:, 1] > 0)
            & (coarse_indices[:, 1] < coarse_y - 1)
        )
        keep = ~coarse_front_interior
        coarse_indices = coarse_indices[keep]
        coarse_positions = coarse_positions[keep]
        coarse_face_masks = coarse_face_masks[keep]

        fine_axis_x = np.arange(
            subdivision_x,
            front_x - subdivision_x,
            dtype=np.int32,
        )
        fine_axis_y = np.arange(
            subdivision_y,
            front_y - subdivision_y,
            dtype=np.int32,
        )
        fine_x, fine_y = np.meshgrid(fine_axis_x, fine_axis_y, indexing="ij")
        fine_indices = np.column_stack(
            (
                fine_x.reshape(-1),
                fine_y.reshape(-1),
                np.zeros(fine_x.size, dtype=np.int32),
            )
        )
        width, height, depth = spec.dimensions_m
        fine_positions = np.column_stack(
            (
                (fine_indices[:, 0] + 0.5) * spec.front_cell_pitch_m - width / 2,
                (fine_indices[:, 1] + 0.5) * spec.front_cell_pitch_m - height / 2,
                np.full(
                    len(fine_indices),
                    depth / 2 - spec.front_cell_pitch_m / 2,
                    dtype=np.float64,
                ),
            )
        )

        grid_xyz = np.vstack((coarse_indices, fine_indices))
        base_positions = np.vstack((coarse_positions, fine_positions))
        face_masks = np.concatenate(
            (
                coarse_face_masks,
                np.full(len(fine_indices), FACE_BITS["zMax"], dtype=np.uint8),
            )
        )
        front_scan_layer = np.concatenate(
            (np.zeros(len(coarse_indices), dtype=bool), np.ones(len(fine_indices), dtype=bool))
        )
        grid_shape_xyz = np.vstack(
            (
                np.tile(
                    np.asarray(coarse_shape, dtype=np.uint16),
                    (len(coarse_indices), 1),
                ),
                np.tile(
                    np.asarray((front_x, front_y, 1), dtype=np.uint16),
                    (len(fine_indices), 1),
                ),
            )
        )
        cell_pitch = np.concatenate(
            (
                np.full(len(coarse_indices), spec.coarse_cell_pitch_m, dtype=np.float32),
                np.full(len(fine_indices), spec.front_cell_pitch_m, dtype=np.float32),
            )
        )

    positions = base_positions.copy()
    scan_indices = grid_xyz[front_scan_layer]
    front_insets = scan.indentation_m[scan_indices[:, 1], scan_indices[:, 0]]
    positions[front_scan_layer, 2] -= front_insets

    affected = np.zeros(len(grid_xyz), dtype=bool)
    affected[front_scan_layer] = front_insets > 0
    confidence = np.full(
        len(grid_xyz),
        spec.cube.inferred_confidence,
        dtype=np.float32,
    )
    confidence[affected] = spec.observed_confidence
    source_bits = np.full(
        len(grid_xyz),
        TEMPLATE_INFERRED_SOURCE_BIT,
        dtype=np.uint8,
    )
    source_bits[affected] = FRONT_RELIEF_SOURCE_BITS
    geometry = FrontReliefGeometry(
        grid_xyz=grid_xyz,
        grid_shape_xyz=grid_shape_xyz,
        cell_pitch_m=cell_pitch,
        face_masks=face_masks,
        base_positions=base_positions,
        positions=positions,
        confidence=confidence,
        source_bits=source_bits,
        front_scan_layer=front_scan_layer,
    )
    if len(grid_xyz) != spec.hybrid_surface_cell_count:
        raise RuntimeError("hybrid surface-cell count does not match the configured budget")
    return geometry


def _curvature_gate(scan: FrontReliefScan) -> dict[str, float | bool]:
    depth = scan.depth_statistics
    features = scan.feature_metrics
    nose = float(depth["noseTipInsetMeters"])
    cheek = (
        float(depth["leftCheekInsetMeters"])
        + float(depth["rightCheekInsetMeters"])
    ) / 2
    span = float(depth["actualMaxInsetMeters"]) - float(depth["actualMinInsetMeters"])
    eye_contrast = float(features["meanEyeSocketContrastMeters"])
    nose_contrast = float(features["noseProjectionContrastMeters"])
    nostril_contrast = float(features["nostrilContrastMeters"])
    mouth_contrast = float(features["mouthSeamContrastMeters"])
    feature_min = float(features["featureReliefMinMeters"])
    feature_max = float(features["featureReliefMaxMeters"])
    passed = (
        nose + 0.0010 < cheek
        and span >= 0.012
        and eye_contrast >= 0.0012
        and nose_contrast >= 0.0018
        and nostril_contrast >= 0.0010
        and mouth_contrast >= 0.0012
        and feature_min <= -0.0020
        and feature_max >= 0.0020
    )
    return {
        "passed": passed,
        "noseTipInsetMeters": nose,
        "meanCheekInsetMeters": cheek,
        "faceInsetSpanMeters": span,
        "eyeSocketContrastMeters": eye_contrast,
        "noseProjectionContrastMeters": nose_contrast,
        "nostrilContrastMeters": nostril_contrast,
        "mouthSeamContrastMeters": mouth_contrast,
        "featureReliefMinMeters": feature_min,
        "featureReliefMaxMeters": feature_max,
    }


def _traceability_bytes(
    geometry: FrontReliefGeometry,
    scan: FrontReliefScan,
    image_shape: tuple[int, int],
) -> bytes:
    image_height, image_width = image_shape
    records: list[bytes] = []
    for cell_id, (
        grid,
        grid_shape,
        cell_pitch,
        base,
        position,
        face_mask,
        cell_confidence,
        bits,
        scan_layer,
    ) in enumerate(
        zip(
            geometry.grid_xyz,
            geometry.grid_shape_xyz,
            geometry.cell_pitch_m,
            geometry.base_positions,
            geometry.positions,
            geometry.face_masks,
            geometry.confidence,
            geometry.source_bits,
            geometry.front_scan_layer,
            strict=True,
        )
    ):
        indentation = float(base[2] - position[2])
        observed = bits == FRONT_RELIEF_SOURCE_BITS
        source_uv_normalized: list[float] | None = None
        source_uv_pixels: list[float] | None = None
        feature_class: int | None = None
        curvature_fraction: float | None = None
        feature_relief_m: float | None = None
        if observed:
            uv = scan.source_uv_normalized[int(grid[1]), int(grid[0])]
            source_uv_normalized = [round(float(uv[0]), 9), round(float(uv[1]), 9)]
            source_uv_pixels = [
                round(float(uv[0] * (image_width - 1)), 3),
                round(float(uv[1] * (image_height - 1)), 3),
            ]
            feature_class = int(scan.feature_class[int(grid[1]), int(grid[0])])
            curvature_fraction = round(
                float(scan.curvature_inset_fraction[int(grid[1]), int(grid[0])]),
                6,
            )
            feature_relief_m = round(
                float(scan.feature_relief_m[int(grid[1]), int(grid[0])]),
                9,
            )
        records.append(
            canonical_json_bytes(
                {
                    "basePositionMeters": [round(float(value), 9) for value in base],
                    "cellPitchMeters": round(float(cell_pitch), 9),
                    "cellId": cell_id,
                    "confidence": round(float(cell_confidence), 6),
                    "curvatureInsetFraction": curvature_fraction,
                    "faceMask": int(face_mask),
                    "featureClass": feature_class,
                    "featureReliefMeters": feature_relief_m,
                    "gridShapeXYZ": [int(value) for value in grid_shape],
                    "gridXYZ": [int(value) for value in grid],
                    "indentationMeters": round(indentation, 9),
                    "layer": "frontScan" if scan_layer else "cuboidShell",
                    "observedFrom2D": bool(observed),
                    "positionMeters": [round(float(value), 9) for value in position],
                    "sourceBits": int(bits),
                    "sourceUVNormalized": source_uv_normalized,
                    "sourceUVPixels": source_uv_pixels,
                }
            )
        )
    return b"\n".join(records) + b"\n"


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _mix_color(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    value: float,
) -> tuple[int, int, int]:
    value = float(np.clip(value, 0.0, 1.0))
    return tuple(
        round(left + (right - left) * value)
        for left, right in zip(start, end, strict=True)
    )


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    ):
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _relief_shading(
    scan: FrontReliefScan,
    spec: FrontFaceReliefSpec,
) -> np.ndarray:
    surface = -cv2.GaussianBlur(
        scan.indentation_m.astype(np.float32),
        (0, 0),
        1.1,
    )
    gradient_y, gradient_x = np.gradient(surface, spec.front_cell_pitch_m)
    normal = np.dstack((-gradient_x, -gradient_y, np.ones_like(surface)))
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-8)
    light = np.asarray([-0.42, 0.58, 0.70], dtype=np.float32)
    light /= np.linalg.norm(light)
    diffuse = np.clip(normal @ light, 0.0, 1.0)
    return (0.28 + 0.72 * diffuse).astype(np.float32)


def _relief_rgb(scan: FrontReliefScan, spec: FrontFaceReliefSpec) -> np.ndarray:
    depth = np.flipud(scan.indentation_m).astype(np.float32)
    shading = np.flipud(_relief_shading(scan, spec))
    ratio = np.clip(depth / spec.max_inset_m, 0.0, 1.0)[:, :, None]
    shallow = np.asarray([244, 201, 164], dtype=np.float32)
    deep = np.asarray([101, 40, 68], dtype=np.float32)
    color = shallow + (deep - shallow) * ratio
    color *= (0.70 + 0.42 * shading)[:, :, None]
    color[depth <= 0] = np.asarray([215, 222, 229], dtype=np.float32)
    return np.clip(color, 0, 255).astype(np.uint8)


def _perspective_polygons(
    geometry: FrontReliefGeometry,
    scan: FrontReliefScan,
    spec: FrontFaceReliefSpec,
) -> list[tuple[float, list[tuple[float, float]], tuple[int, int, int]]]:
    camera = np.asarray([0.33, 0.235, 0.47], dtype=np.float64)
    target = np.asarray([0.0, -0.002, 0.0], dtype=np.float64)
    forward = target - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    light = np.asarray([-0.45, 0.72, 0.80], dtype=np.float64)
    light /= np.linalg.norm(light)
    focal = 1360.0
    center = np.asarray([1175.0, 490.0])
    face_definitions = (
        (np.asarray([1.0, 0.0, 0.0]), ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1))),
        (np.asarray([-1.0, 0.0, 0.0]), ((-1, -1, 1), (-1, 1, 1), (-1, 1, -1), (-1, -1, -1))),
        (np.asarray([0.0, 1.0, 0.0]), ((-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1))),
        (np.asarray([0.0, -1.0, 0.0]), ((-1, -1, 1), (-1, -1, -1), (1, -1, -1), (1, -1, 1))),
        (np.asarray([0.0, 0.0, 1.0]), ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))),
        (np.asarray([0.0, 0.0, -1.0]), ((1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1))),
    )
    polygons: list[tuple[float, list[tuple[float, float]], tuple[int, int, int]]] = []
    relief_shading = _relief_shading(scan, spec)
    for grid, grid_shape, pitch, position, scan_layer in zip(
        geometry.grid_xyz,
        geometry.grid_shape_xyz,
        geometry.cell_pitch_m,
        geometry.positions,
        geometry.front_scan_layer,
        strict=True,
    ):
        grid_shape = np.asarray(grid_shape, dtype=np.int32)
        half = float(pitch) * spec.cube.cell_fill_ratio / 2
        indentation = 0.0
        affected = bool(scan_layer and scan.face_mask[grid[1], grid[0]])
        if affected:
            indentation = float(scan.indentation_m[grid[1], grid[0]])
            base_color = _mix_color(
                (238, 183, 139),
                (109, 45, 70),
                indentation / spec.max_inset_m,
            )
            local_light = float(relief_shading[grid[1], grid[0]])
            base_color = tuple(
                int(np.clip(channel * (0.72 + 0.38 * local_light), 0, 255))
                for channel in base_color
            )
        else:
            base_color = (194, 204, 215)
        outward_normals: set[tuple[int, int, int]] = set()
        if grid[0] == 0:
            outward_normals.add((-1, 0, 0))
        if grid[0] == grid_shape[0] - 1:
            outward_normals.add((1, 0, 0))
        if grid[1] == 0:
            outward_normals.add((0, -1, 0))
        if grid[1] == grid_shape[1] - 1:
            outward_normals.add((0, 1, 0))
        if grid[2] == 0:
            outward_normals.add((0, 0, -1))
        if grid[2] == grid_shape[2] - 1:
            outward_normals.add((0, 0, 1))
        for normal, signs in face_definitions:
            normal_key = tuple(int(value) for value in normal)
            if normal_key not in outward_normals:
                continue
            face_center = position + normal * half
            if float(np.dot(normal, camera - face_center)) <= 0:
                continue
            vertices = position + np.asarray(signs, dtype=np.float64) * half
            relative = vertices - camera
            depth = relative @ forward
            if np.any(depth <= 0):
                continue
            projected = np.column_stack(
                (
                    center[0] + focal * (relative @ right) / depth,
                    center[1] - focal * (relative @ up) / depth,
                )
            )
            brightness = 0.64 + 0.36 * max(float(np.dot(normal, light)), 0.0)
            if affected:
                brightness *= 0.96
            color = tuple(int(np.clip(channel * brightness, 0, 255)) for channel in base_color)
            polygons.append(
                (
                    float(np.mean(depth)),
                    [(float(point[0]), float(point[1])) for point in projected],
                    color,
                )
            )
    return sorted(polygons, key=lambda item: item[0], reverse=True)


def render_front_relief_preview(
    destination: Path,
    geometry: FrontReliefGeometry,
    scan: FrontReliefScan,
    spec: FrontFaceReliefSpec,
) -> None:
    canvas = Image.new("RGB", (1600, 900), (235, 239, 243))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(38)
    section_font = _font(24)
    body_font = _font(18)
    small_font = _font(15)

    draw.text((54, 36), "Z+ 正面 · 单图内凹 3D Pixel 长方体", fill=(30, 37, 46), font=title_font)
    draw.text(
        (56, 88),
        "仅正面表层 Pixel 沿 -Z 移动；边框、侧面、背面不变",
        fill=(87, 97, 108),
        font=body_font,
    )
    draw.rounded_rectangle((42, 133, 710, 848), radius=18, fill=(249, 250, 252))
    draw.rounded_rectangle((730, 133, 1558, 848), radius=18, fill=(249, 250, 252))
    draw.text((70, 157), "正面深度图", fill=(38, 46, 55), font=section_font)
    draw.text((758, 157), "实际 Pixel 位移预览", fill=(38, 46, 55), font=section_font)

    front_x, front_y = spec.front_cells_xy
    preview_extent = 570
    display_scale = min(preview_extent / front_x, preview_extent / front_y)
    grid_width = round(front_x * display_scale)
    grid_height = round(front_y * display_scale)
    grid_left = 85 + (preview_extent - grid_width) // 2
    grid_top = 215
    relief_rgb = _relief_rgb(scan, spec)
    depth_image = Image.fromarray(relief_rgb).resize(
        (grid_width, grid_height),
        Image.Resampling.NEAREST,
    )
    canvas.paste(depth_image, (grid_left, grid_top))
    if max(front_x, front_y) > 160:
        grid_step_x = max(16, front_x // 8)
        grid_step_y = max(16, front_y // 8)
        cell_size_x = grid_width / front_x
        cell_size_y = grid_height / front_y
        for index in range(0, front_x + 1, grid_step_x):
            offset = grid_left + index * cell_size_x
            draw.line(
                (round(offset), grid_top, round(offset), grid_top + grid_height),
                fill=(229, 232, 235),
                width=1,
            )
        for index in range(0, front_y + 1, grid_step_y):
            offset = grid_top + index * cell_size_y
            draw.line(
                (grid_left, round(offset), grid_left + grid_width, round(offset)),
                fill=(229, 232, 235),
                width=1,
            )
    draw.rectangle(
        (grid_left, grid_top, grid_left + grid_width, grid_top + grid_height),
        outline=(82, 91, 102),
        width=2,
    )
    draw.text((85, 801), "0 cm  平面", fill=(92, 101, 112), font=small_font)
    for index in range(101):
        value = index / 100
        color = _mix_color((244, 201, 164), (101, 40, 68), value)
        draw.line((192 + index * 2.5, 808, 192 + index * 2.5, 824), fill=color, width=3)
    draw.text(
        (448, 801),
        f"{spec.max_inset_m * 100:.1f} cm  最深",
        fill=(92, 101, 112),
        font=small_font,
    )

    front_quad = np.asarray(
        [[875, 235], [1270, 220], [1290, 765], [895, 785]],
        dtype=np.float32,
    )
    draw.polygon(
        [(875, 235), (1010, 175), (1405, 165), (1270, 220)],
        fill=(201, 209, 218),
        outline=(104, 113, 123),
    )
    draw.polygon(
        [(1270, 220), (1405, 165), (1425, 705), (1290, 765)],
        fill=(169, 180, 192),
        outline=(104, 113, 123),
    )
    source_quad = np.asarray(
        [
            [0, 0],
            [front_x - 1, 0],
            [front_x - 1, front_y - 1],
            [0, front_y - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source_quad, front_quad)
    warped = cv2.warpPerspective(
        relief_rgb,
        transform,
        canvas.size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    warp_mask = cv2.warpPerspective(
        np.full((front_y, front_x), 255, dtype=np.uint8),
        transform,
        canvas.size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    canvas.paste(Image.fromarray(warped), (0, 0), Image.fromarray(warp_mask))
    draw = ImageDraw.Draw(canvas)
    draw.line(
        [tuple(point) for point in front_quad.astype(int)] + [tuple(front_quad[0].astype(int))],
        fill=(82, 91, 102),
        width=2,
        joint="curve",
    )

    affected = int(scan.depth_statistics["affectedFrontCellCount"])
    actual_max_cm = float(scan.depth_statistics["actualMaxInsetMeters"]) * 100
    width_cm, height_cm, depth_cm = (value * 100 for value in spec.dimensions_m)
    draw.rounded_rectangle((820, 790, 1525, 835), radius=12, fill=(236, 239, 243))
    draw.text(
        (838, 802),
        (
            f"{width_cm:.1f}×{height_cm:.1f}×{depth_cm:.1f} cm"
            f" · 正面 {front_x}×{front_y} · {affected} Pixel · 内凹 {actual_max_cm:.2f} cm"
        ),
        fill=(58, 67, 77),
        font=body_font,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def _write_front_relief_asset(
    source_image: Path,
    face_landmarker_model: Path,
    destination: Path,
    rgb: np.ndarray,
    scan: FrontReliefScan,
    spec: FrontFaceReliefSpec,
    canonical_face_model: Path | None = None,
) -> dict[str, Any]:
    destination = Path(destination)
    geometry = apply_front_relief(scan, spec)
    affected = geometry.source_bits == FRONT_RELIEF_SOURCE_BITS

    model_path = destination / "models" / "voxels.glb"
    records_path = destination / "pixels" / "cells.jsonl"
    schema_path = destination / "pixels" / "schema.json"
    depth_grid_path = destination / "relief" / "front-depth-grid.json"
    scan_path = destination / "relief" / "scan.json"
    preview_path = destination / "preview" / "front-relief-16x9.png"
    manifest_path = destination / "manifest.json"

    export_instanced_voxels(
        geometry.positions,
        geometry.cell_pitch_m,
        geometry.confidence,
        geometry.source_bits,
        model_path,
        fill_ratio=spec.cube.cell_fill_ratio,
    )
    atomic_write_bytes(
        records_path,
        _traceability_bytes(
            geometry,
            scan,
            rgb.shape[:2],
        ),
    )
    schema = {
        "schemaVersion": "1.0.0",
        "format": "face3d-front-relief-3d-pixel-cuboid-jsonl",
        "recordCount": len(geometry.grid_xyz),
        "recordOrder": "coarse cuboid shell, then fine zMax scan layer; each layer x-major",
        "units": "meter",
        "fields": {
            "cellId": "uint32",
            "gridXYZ": "uint32[3]",
            "gridShapeXYZ": "uint16[3]; grid dimensions for this layer",
            "cellPitchMeters": "float64",
            "layer": "cuboidShell | frontScan",
            "basePositionMeters": "float64[3]",
            "positionMeters": "float64[3]",
            "indentationMeters": "float64; non-negative inward displacement on zMax only",
            "faceMask": "uint8",
            "confidence": "float32",
            "sourceBits": "uint8",
            "observedFrom2D": "bool",
            "sourceUVNormalized": "float64[2] | null",
            "sourceUVPixels": "float64[2] | null",
            "featureClass": "uint8 | null; 1 eye, 2 nose, 3 mouth, 4 ear/jaw",
            "featureReliefMeters": (
                "float64 | null; signed explicit local relief, positive is deeper"
            ),
            "curvatureInsetFraction": "float32 | null",
        },
        "faceBits": FACE_BITS,
        "sourceBits": {
            "frontObserved": FRONT_OBSERVED_SOURCE_BIT,
            "templateInferred": TEMPLATE_INFERRED_SOURCE_BIT,
            "frontRelativeDepth": FRONT_RELIEF_SOURCE_BITS,
        },
    }
    atomic_write_json(schema_path, schema)
    atomic_write_json(
        depth_grid_path,
        {
            "schemaVersion": "1.0.0",
            "gridShape": list(spec.front_shape_yx),
            "gridIndexing": "rows are cuboid Y low-to-high; columns are cuboid X low-to-high",
            "inwardAxis": "-Z",
            "indentationMeters": np.round(scan.indentation_m, 9).tolist(),
            "featureReliefMeters": np.round(scan.feature_relief_m, 9).tolist(),
            "faceMask": scan.face_mask.astype(np.uint8).tolist(),
            "featureClass": scan.feature_class.tolist(),
            "curvatureInsetFraction": np.round(
                scan.curvature_inset_fraction,
                7,
            ).tolist(),
            "sourceUVNormalized": np.round(scan.source_uv_normalized, 9).tolist(),
        },
    )
    atomic_write_json(
        scan_path,
        {
            "schemaVersion": "1.0.0",
            "sourceImage": {
                "path": _portable_path(source_image),
                "sha256": sha256_file(source_image),
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
            },
            "faceLandmarker": {
                "path": _portable_path(face_landmarker_model),
                "sha256": sha256_file(face_landmarker_model),
            },
            "canonicalFaceModel": (
                {
                    "path": _portable_path(canonical_face_model),
                    "sha256": sha256_file(canonical_face_model),
                }
                if canonical_face_model is not None
                else None
            ),
            "scanConfig": (
                {
                    "path": _portable_path(spec.scan_config_path),
                    "sha256": sha256_file(spec.scan_config_path),
                }
                if spec.scan_config_path is not None and spec.scan_config_path.is_file()
                else None
            ),
            "landmarksNormalized": np.round(scan.landmarks_normalized, 9).tolist(),
            "mapping": scan.mapping,
            "frontalMetrics": scan.frontal_metrics,
            "featureMetrics": scan.feature_metrics,
            "depthStatistics": scan.depth_statistics,
            "depthMethod": (
                f"face-v1 pixel-direct scan: coarse {spec.coarse_depth_grid}-grid RBF baseline, "
                "canonical 468-point "
                "depth regularization, protected semantic and photometric detail, explicit "
                "millimetre-scale eye/nose/mouth sculpting, plus an ellipsoidal curvature prior"
            ),
        },
    )
    render_front_relief_preview(preview_path, geometry, scan, spec)

    half_cells = geometry.cell_pitch_m[:, None] * spec.cube.cell_fill_ratio / 2
    actual_min = (geometry.positions - half_cells).min(axis=0)
    actual_max = (geometry.positions + half_cells).max(axis=0)
    procedure = {
        "cuboid": {
            "dimensionsMeters": list(spec.dimensions_m),
            "coarseCellsXYZ": list(spec.coarse_cells_xyz),
            "frontCellsXY": list(spec.front_cells_xy),
            "cellFillRatio": spec.cube.cell_fill_ratio,
        },
        "relief": {
            "modifiedFace": "zMax",
            "inwardAxis": "-Z",
            "requestedMaxInsetMeters": spec.max_inset_m,
            "coarseDepthGrid": spec.coarse_depth_grid,
            "complexRegionRadiusPixels": spec.complex_region_radius_pixels,
            "depthScaleFaceWidth": spec.depth_scale_face_width,
            "featureSculpting": "explicit-millimetre-eye-nose-mouth-relief",
            "mapping": scan.mapping,
        },
    }
    manifest = {
        "schemaVersion": "1.0.0",
        "assetType": "3d-pixel-cuboid-front-face-relief",
        "primaryAsset": True,
        "units": "meter",
        "geometry": {
            "centerMeters": [0.0, 0.0, 0.0],
            "nominalDimensionsMeters": list(spec.dimensions_m),
            "nominalDimensionsCentimeters": [value * 100 for value in spec.dimensions_m],
            "actualPixelBoundsMeters": {
                "min": np.round(actual_min, 9).tolist(),
                "max": np.round(actual_max, 9).tolist(),
            },
            "surfaceOnly": True,
        },
        "pixel": {
            "representation": (
                f"{spec.coarse_cells_xyz}-cuboid-shell-with-"
                f"{spec.front_cells_xy}-subdivided-inset-zMax-scan-layer"
            ),
            "surfaceOnly": True,
            "cuboidShellCellsXYZ": list(spec.coarse_cells_xyz),
            "cuboidShellCellPitchMeters": spec.coarse_cell_pitch_m,
            "frontScanCellsXY": list(spec.front_cells_xy),
            "frontScanCellPitchMeters": spec.front_cell_pitch_m,
            "cellFillRatio": spec.cube.cell_fill_ratio,
            "instanceCount": len(geometry.grid_xyz),
            "expectedInstanceCount": spec.hybrid_surface_cell_count,
            "maximumInstanceBudget": spec.maximum_cells,
            "frontReliefCellCount": int(np.count_nonzero(affected)),
        },
        "frontRelief": {
            "modifiedFace": "zMax",
            "inwardAxis": "-Z",
            "inwardOnly": True,
            "requestedMaxInsetMeters": spec.max_inset_m,
            "actualMaxInsetMeters": scan.depth_statistics["actualMaxInsetMeters"],
            "outerCellRimUnchanged": True,
            "otherFacesUnchanged": True,
            "exactMetricDepthMeasured": False,
            "depthEvidence": "single-front-image-relative-depth",
            "curvaturePrior": "ellipsoidal face dome blended with face-v1 relative depth",
            "scanConfiguration": {
                "gridShapeXY": list(spec.front_cells_xy),
                "coarseDepthGrid": spec.coarse_depth_grid,
                "complexRegionRadiusPixels": spec.complex_region_radius_pixels,
                "depthScaleFaceWidth": spec.depth_scale_face_width,
            },
            "naturalCurvatureGate": _curvature_gate(scan),
            "featureFidelityGate": _curvature_gate(scan),
            "featureMetrics": scan.feature_metrics,
            "mapping": scan.mapping,
        },
        "provenance": {
            "kind": "single-front-image-relative-depth-plus-template-inference",
            "observedFrom2D": True,
            "sourceImage": {
                "path": _portable_path(source_image),
                "sha256": sha256_file(source_image),
            },
            "faceLandmarkerSha256": sha256_file(face_landmarker_model),
            "canonicalFaceModelSha256": (
                sha256_file(canonical_face_model) if canonical_face_model is not None else None
            ),
            "scanConfigSha256": (
                sha256_file(spec.scan_config_path)
                if spec.scan_config_path is not None and spec.scan_config_path.is_file()
                else None
            ),
            "procedureSpecSha256": sha256_json(procedure),
            "limitations": [
                "A single image constrains screen-space location but not metric facial depth.",
                (
                    "All indentation depth is relative and template-inferred; "
                    "sourceBits therefore include 8."
                ),
                (
                    f"The {spec.front_cells_xy} front grid improves visible detail but does not "
                    "turn a single image "
                    "into metric or hidden-surface scan evidence."
                ),
            ],
        },
        "files": {
            "model": "models/voxels.glb",
            "modelSha256": sha256_file(model_path),
            "traceability": "pixels/cells.jsonl",
            "traceabilitySha256": sha256_file(records_path),
            "schema": "pixels/schema.json",
            "schemaSha256": sha256_file(schema_path),
            "depthGrid": "relief/front-depth-grid.json",
            "depthGridSha256": sha256_file(depth_grid_path),
            "scan": "relief/scan.json",
            "scanSha256": sha256_file(scan_path),
            "preview": "preview/front-relief-16x9.png",
            "previewSha256": sha256_file(preview_path),
        },
    }
    atomic_write_json(manifest_path, manifest)
    return {
        "ok": True,
        "output": str(destination),
        "model": str(model_path),
        "manifest": str(manifest_path),
        "preview": str(preview_path),
        "dimensionsCentimeters": [value * 100 for value in spec.dimensions_m],
        "maximumInsetCentimeters": scan.depth_statistics["actualMaxInsetMeters"] * 100,
        "cuboidShellPixelMillimeters": spec.coarse_cell_pitch_m * 1000,
        "frontScanPixelMillimeters": spec.front_cell_pitch_m * 1000,
        "frontReliefCellCount": int(np.count_nonzero(affected)),
        "instanceCount": len(geometry.grid_xyz),
        "featureFidelityGate": _curvature_gate(scan),
    }


def create_front_face_relief(
    source_image: Path,
    face_landmarker_model: Path,
    destination: Path,
    spec: FrontFaceReliefSpec | None = None,
    canonical_face_model: Path | None = None,
) -> dict[str, Any]:
    spec = spec or FrontFaceReliefSpec()
    source_image = Path(source_image)
    face_landmarker_model = Path(face_landmarker_model)
    canonical_face_model = (
        None if canonical_face_model is None else Path(canonical_face_model)
    )
    if not source_image.is_file():
        fail("source-image-missing", f"正面图片不存在: {source_image}", stage="input")
    if not face_landmarker_model.is_file():
        fail(
            "face-landmarker-missing",
            f"Face Landmarker 模型不存在: {face_landmarker_model}",
            stage="assets",
        )
    if canonical_face_model is not None and not canonical_face_model.is_file():
        fail(
            "canonical-face-model-missing",
            f"Canonical Face Model 不存在: {canonical_face_model}",
            stage="assets",
        )
    try:
        with Image.open(source_image) as opened:
            rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    except Exception as exc:
        fail("image-decode-failed", f"无法读取正面图片: {exc}", stage="input")
    with face_landmarker(face_landmarker_model) as detector:
        landmarks, _, _ = _detect(detector, rgb)
    scan = estimate_front_relief(
        rgb,
        landmarks,
        spec,
        canonical_face_model=canonical_face_model,
    )
    if (
        scan.frontal_metrics["noseHorizontalOffsetFaceWidths"] > 0.12
        or scan.frontal_metrics["eyeTiltFaceHeights"] > 0.08
    ):
        fail(
            "front-view-required",
            "正面内凹扫描要求近似正脸且双眼近似水平",
            stage="input",
            details=scan.frontal_metrics,
        )
    curvature_gate = _curvature_gate(scan)
    if not curvature_gate["passed"]:
        fail(
            "front-curvature-gate-failed",
            "正面扫描未形成可辨识的眼窝、鼻部、嘴唇或自然面部弧度",
            stage="pixel-cube-relief",
            details=curvature_gate,
        )
    return _write_front_relief_asset(
        source_image,
        face_landmarker_model,
        destination,
        rgb,
        scan,
        spec,
        canonical_face_model,
    )
