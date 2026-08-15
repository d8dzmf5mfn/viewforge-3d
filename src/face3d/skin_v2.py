from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab

from face3d.config import Face3DConfig
from face3d.io import sha256_file
from face3d.models import REQUIRED_VIEWS, CameraRecord, ViewRole
from face3d.skin import (
    _delight,
    _head_cylindrical_coordinates,
    _linear_to_srgb,
    _rasterize_atlas,
    _sample_image,
    _smoothstep,
    _vertex_visibility,
    _write_image,
)
from face3d.unified_head import UnifiedHeadAsset


@dataclass(slots=True)
class SurfaceProjection:
    color: np.ndarray
    confidence: np.ndarray
    source_role: np.ndarray
    source_uv: np.ndarray
    depth: np.ndarray
    source_bits: np.ndarray
    per_view_weights: np.ndarray
    seam_delta_e_median: float
    seam_delta_e_p95: float


@dataclass(slots=True)
class SkinV2Result:
    metrics: dict[str, Any]
    projection: SurfaceProjection


def _read_projection_inputs(
    run_dir: Path,
) -> tuple[dict[ViewRole, np.ndarray], dict[ViewRole, np.ndarray], np.ndarray]:
    linear: dict[ViewRole, np.ndarray] = {}
    masks: dict[ViewRole, np.ndarray] = {}
    medians: list[np.ndarray] = []
    for role in REQUIRED_VIEWS:
        bgr = cv2.imread(str(run_dir / "references" / f"{role.value}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(
            str(run_dir / "working" / "masks" / f"{role.value}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if bgr is None or mask is None:
            raise ValueError(f"Face v2 projection input is unreadable: {role.value}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        de_lit = _delight(image, mask)
        foreground = de_lit[mask > 127]
        if not len(foreground):
            raise ValueError(f"Face v2 mask is empty: {role.value}")
        median = np.median(foreground, axis=0)
        medians.append(median)
        linear[role] = de_lit
        masks[role] = mask
    common = np.median(np.stack(medians), axis=0)
    for role, median in zip(REQUIRED_VIEWS, medians, strict=True):
        gain = np.clip(common / np.maximum(median, 1e-4), 0.80, 1.25)
        linear[role] = np.clip(linear[role] * gain[None, None, :], 0.0, 1.0)
    return linear, masks, common


def project_multiview_skin(
    run_dir: Path,
    mesh: trimesh.Trimesh,
    cameras: list[CameraRecord],
) -> SurfaceProjection:
    images, masks, fallback = _read_projection_inputs(run_dir)
    camera_by_role = {camera.role: camera for camera in cameras}
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64).copy()
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    head_angle = np.degrees(_head_cylindrical_coordinates(vertices)[0])
    count = len(vertices)
    weights = np.zeros((count, len(REQUIRED_VIEWS)), dtype=np.float32)
    samples = np.zeros((count, len(REQUIRED_VIEWS), 3), dtype=np.float32)
    low_samples = np.zeros_like(samples)
    source_uv_by_view = np.zeros((count, len(REQUIRED_VIEWS), 2), dtype=np.float64)
    depth_by_view = np.zeros((count, len(REQUIRED_VIEWS)), dtype=np.float64)

    for view_index, role in enumerate(REQUIRED_VIEWS):
        image = images[role]
        mask = masks[role]
        camera = camera_by_role[role]
        rotation, _ = cv2.Rodrigues(np.asarray(camera.rotation_vector, dtype=np.float64))
        translation = np.asarray(camera.translation, dtype=np.float64)
        camera_vertices = vertices @ rotation.T + translation
        depth = camera_vertices[:, 2]
        positive = depth > 1e-6
        pixels = np.zeros((count, 2), dtype=np.float64)
        pixels[positive] = camera_vertices[positive, :2] / depth[positive, None]
        pixels[positive] *= camera.focal_length_px
        pixels[positive] += np.asarray(camera.principal_point_px)
        inside = (
            positive
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] <= camera.width - 1)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] <= camera.height - 1)
        )
        visible = _vertex_visibility(mesh, pixels, depth, positive, image.shape[:2])
        mask_support = _sample_image((mask / 255.0)[..., None], pixels)[:, 0]
        distance = cv2.distanceTransform((mask > 127).astype(np.uint8), cv2.DIST_L2, 5)
        edge = np.clip(
            _sample_image(distance[..., None], pixels)[:, 0]
            / max(min(image.shape[:2]) * 0.015, 4.0),
            0.0,
            1.0,
        )
        camera_center = -rotation.T @ translation
        view = camera_center[None, :] - vertices
        view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
        facing = np.clip(np.sum(normals * view, axis=1), 0.0, 1.0)
        sharpness_map = np.abs(cv2.Laplacian(image.mean(axis=2), cv2.CV_32F))
        sharpness = _sample_image(sharpness_map[..., None], pixels)[:, 0]
        sharpness_scale = (
            float(np.quantile(sharpness[inside], 0.90)) if np.any(inside) else 1.0
        )
        if sharpness_scale <= 1e-6:
            # A uniformly coloured synthetic/reference region is not evidence
            # of blur. Keep it observable; only downweight relative blur when
            # the image contains a measurable high-frequency signal.
            sharpness = np.ones_like(sharpness)
        else:
            sharpness /= sharpness_scale
            sharpness = np.clip(sharpness, 0.25, 1.0)
        if role == ViewRole.FRONT:
            route = 1.0 - _smoothstep(45.0, 82.0, np.abs(head_angle))
            route *= 1.20
        elif role == ViewRole.LEFT45:
            route = _smoothstep(12.0, 48.0, head_angle)
            route *= 1.0 - _smoothstep(138.0, 174.0, head_angle)
        else:
            route = _smoothstep(12.0, 48.0, -head_angle)
            route *= 1.0 - _smoothstep(138.0, 174.0, -head_angle)
        weight = (
            inside.astype(np.float64)
            * visible
            * (mask_support > 0.55)
            * edge
            * facing**3.0
            * sharpness
            * route
        )
        low = cv2.GaussianBlur(image, (0, 0), max(min(image.shape[:2]) / 70.0, 8.0))
        samples[:, view_index] = _sample_image(image, pixels)
        low_samples[:, view_index] = _sample_image(low, pixels)
        weights[:, view_index] = weight.astype(np.float32)
        source_uv_by_view[:, view_index] = pixels
        depth_by_view[:, view_index] = depth

    sum_weight = weights.sum(axis=1)
    observed = sum_weight > 1e-6
    color = np.broadcast_to(fallback, (count, 3)).astype(np.float32).copy()
    low_color = color.copy()
    color[observed] = np.einsum(
        "nvc,nv->nc", samples[observed], weights[observed], optimize=True
    ) / sum_weight[observed, None]
    low_color[observed] = np.einsum(
        "nvc,nv->nc", low_samples[observed], weights[observed], optimize=True
    ) / sum_weight[observed, None]
    high_weight = weights**1.5
    high_sum = high_weight.sum(axis=1)
    high = np.zeros_like(color)
    high[observed] = np.einsum(
        "nvc,nv->nc",
        samples[observed] - low_samples[observed],
        high_weight[observed],
        optimize=True,
    ) / np.maximum(high_sum[observed, None], 1e-8)
    color[observed] = np.clip(low_color[observed] + high[observed], 0.0, 1.0)

    source_role = np.argmax(weights, axis=1).astype(np.uint8)
    source_role[~observed] = 255
    rows = np.arange(count)
    selected = np.minimum(source_role.astype(np.int64), len(REQUIRED_VIEWS) - 1)
    source_uv = np.rint(source_uv_by_view[rows, selected]).clip(0, 65535).astype(np.uint16)
    depth = depth_by_view[rows, selected].astype(np.float32)
    source_bits = np.asarray([1, 2, 4], dtype=np.uint8)[selected]
    source_bits[~observed] = 8
    depth[~observed] = 0.0
    confidence = np.clip(sum_weight / 1.20, 0.0, 1.0).astype(np.float32)

    seam_values: list[np.ndarray] = []
    for first in range(3):
        for second in range(first + 1, 3):
            overlap = (weights[:, first] > 0.05) & (weights[:, second] > 0.05)
            if not np.any(overlap):
                continue
            first_lab = rgb2lab(samples[overlap, first].reshape(-1, 1, 3)).reshape(-1, 3)
            second_lab = rgb2lab(samples[overlap, second].reshape(-1, 1, 3)).reshape(-1, 3)
            seam_values.append(deltaE_ciede2000(first_lab, second_lab))
    delta_e = np.concatenate(seam_values) if seam_values else np.asarray([0.0])
    return SurfaceProjection(
        color=color,
        confidence=confidence,
        source_role=source_role,
        source_uv=source_uv,
        depth=depth,
        source_bits=source_bits,
        per_view_weights=weights,
        seam_delta_e_median=float(np.median(delta_e)),
        seam_delta_e_p95=float(np.quantile(delta_e, 0.95)),
    )


def build_skin_v2(
    run_dir: Path,
    head: UnifiedHeadAsset,
    cameras: list[CameraRecord],
    config: Face3DConfig,
    *,
    observed_confidence_threshold: float = 0.12,
) -> SkinV2Result:
    projection = project_multiview_skin(run_dir, head.skin_mesh, cameras)
    micro_path = config.resolve_asset(config.skin.micro_albedo_source)
    if not micro_path.is_file():
        raise FileNotFoundError(f"skin micro-albedo source is missing: {micro_path}")
    micro = np.asarray(Image.open(micro_path).convert("RGB"), dtype=np.uint8)
    render_mesh = head.render_mesh
    render_projection = projection.color[head.render_to_skin]
    render_confidence = projection.confidence[head.render_to_skin]
    supported = projection.confidence >= observed_confidence_threshold
    fallback = (
        np.median(projection.color[supported], axis=0)
        if np.any(supported)
        else np.median(projection.color, axis=0)
    )
    atlas, atlas_confidence = _rasterize_atlas(
        render_mesh,
        head.uv,
        render_projection,
        render_confidence,
        fallback,
        micro,
        config.skin.atlas_resolution,
    )
    # Projection and blending intentionally run in linear light. A glTF base
    # colour texture is decoded as sRGB by the renderer, so encode the baked
    # atlas back to sRGB before writing it; otherwise it is decoded twice and
    # the skin renders several stops too dark.
    atlas = np.rint(
        _linear_to_srgb(atlas.astype(np.float32) / 255.0) * 255.0
    ).astype(np.uint8)
    textures = run_dir / "textures"
    atlas_image = _write_image(
        textures / "head-albedo.jpg",
        atlas,
        image_format="JPEG",
        quality=config.skin.jpeg_quality,
    )
    confidence_rgb = np.repeat(
        np.rint(np.clip(atlas_confidence, 0.0, 1.0) * 255).astype(np.uint8)[..., None],
        3,
        axis=2,
    )
    _write_image(
        textures / "head-confidence.png", confidence_rgb, image_format="PNG"
    )
    source_palette = np.asarray(
        [[32, 123, 255], [44, 196, 124], [246, 158, 38], [96, 100, 108]],
        dtype=np.float32,
    ) / 255.0
    source_index = projection.source_role.copy()
    source_index[source_index == 255] = 3
    source_color = source_palette[source_index][head.render_to_skin]
    neutral_micro = np.full((32, 32, 3), 128, dtype=np.uint8)
    source_map, _ = _rasterize_atlas(
        render_mesh,
        head.uv,
        source_color,
        np.ones(len(source_color), dtype=np.float32),
        source_palette[3],
        neutral_micro,
        config.skin.atlas_resolution,
    )
    _write_image(textures / "head-source.png", source_map, image_format="PNG")
    head.export_head_glb(run_dir / "models" / "head.glb", atlas_image)
    observed = projection.confidence >= observed_confidence_threshold
    metrics = {
        "representation": "canonical-uv-zbuffer-multifrequency-projection",
        "model": "models/head.glb",
        "atlas": "textures/head-albedo.jpg",
        "confidenceMap": "textures/head-confidence.png",
        "sourceMap": "textures/head-source.png",
        "atlasResolution": [config.skin.atlas_resolution, config.skin.atlas_resolution],
        "observedVertexFraction": float(np.mean(observed)),
        "atlasObservedFraction": float(
            np.mean(atlas_confidence >= observed_confidence_threshold)
        ),
        "observedConfidenceThreshold": observed_confidence_threshold,
        "meanProjectionConfidence": float(np.mean(projection.confidence)),
        "projectionMethod": "per-camera-zbuffer-angle-sharpness",
        "blendMethod": "low-frequency-weighted-plus-high-frequency-best-supported",
        "localWarpFraction": 0.0,
        "seamDeltaE00Median": projection.seam_delta_e_median,
        "seamDeltaE00P95": projection.seam_delta_e_p95,
        "geometryHash": head.geometry_sha256,
        "neutralGeometryHash": head.geometry_sha256,
        "skinGeometryHash": head.geometry_sha256,
        "maximumVertexDifference": 0.0,
        "inferredRegions": ["topCranium", "rearCranium", "shortNeck"],
        "modelSha256": sha256_file(run_dir / "models" / "head.glb"),
        "atlasSha256": sha256_file(textures / "head-albedo.jpg"),
        "confidenceSha256": sha256_file(textures / "head-confidence.png"),
        "sourceSha256": sha256_file(textures / "head-source.png"),
        "passed": bool(
            np.all(np.isfinite(projection.color))
            and np.all(np.isfinite(projection.confidence))
            and np.mean(observed) >= config.skin.minimum_observed_vertex_fraction
            and projection.seam_delta_e_median <= 3.0
            and projection.seam_delta_e_p95 <= 8.0
        ),
    }
    return SkinV2Result(metrics=metrics, projection=projection)
