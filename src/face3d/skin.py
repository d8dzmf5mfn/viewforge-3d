from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import gaussian_filter

from face3d.io import atomic_write_bytes, sha256_file
from face3d.models import CameraRecord, ViewRole


@dataclass(slots=True)
class SkinComponentResult:
    metrics: dict[str, Any]


def _srgb_to_linear(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    return np.where(value <= 0.0031308, value * 12.92, 1.055 * value ** (1 / 2.4) - 0.055)


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    normalized = np.clip((np.asarray(value, dtype=np.float64) - edge0) / (edge1 - edge0), 0, 1)
    return normalized * normalized * (3.0 - 2.0 * normalized)


def _delight(image: np.ndarray, mask: np.ndarray, *, strength: float = 0.46) -> np.ndarray:
    """Remove broad illumination while retaining eyes, lips, follicles, and pores."""
    linear = _srgb_to_linear(np.asarray(image, dtype=np.float32) / 255.0)
    luminance = linear[..., 0] * 0.2126 + linear[..., 1] * 0.7152 + linear[..., 2] * 0.0722
    sigma = max(min(image.shape[:2]) / 34.0, 12.0)
    illumination = cv2.GaussianBlur(luminance, (0, 0), sigma)
    foreground = mask > 127
    target = float(np.median(illumination[foreground])) if np.any(foreground) else 0.42
    gain = np.clip((target / np.maximum(illumination, 0.025)) ** strength, 0.72, 1.38)
    return np.clip(linear * gain[..., None], 0.0, 1.0)


def _head_cylindrical_coordinates(
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return head-centred UVs with a local, vertically feathered ear warp."""
    vertices = np.asarray(vertices, dtype=np.float64)
    lower = np.quantile(vertices, 0.002, axis=0)
    upper = np.quantile(vertices, 0.998, axis=0)
    center_x = float((lower[0] + upper[0]) * 0.5)
    center_z = float((lower[2] + upper[2]) * 0.5)
    angle = np.arctan2(vertices[:, 0] - center_x, vertices[:, 2] - center_z)
    y = vertices[:, 1]
    v = 0.03 + (y - lower[1]) / max(float(upper[1] - lower[1]), 1e-8) * 0.94
    v = np.clip(v, 0.02, 0.98)
    linear_u = 0.5 + angle / (2.0 * np.pi)
    normalized = np.clip(np.abs(angle) / np.pi, 0.0, 1.0)
    front_arc = np.minimum(normalized, 0.58) / 0.58 * 0.38
    rear_arc = 0.38 + np.maximum(normalized - 0.58, 0.0) / 0.42 * 0.12
    ear_u = 0.5 + np.sign(angle) * np.where(normalized <= 0.58, front_arc, rear_arc)
    ear_height_weight = np.exp(-(((v - 0.60) / 0.145) ** 4))
    u = linear_u * (1.0 - ear_height_weight) + ear_u * ear_height_weight
    return angle, u, v


def _sample_image(image: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    source = np.asarray(image, dtype=np.float32)
    if source.ndim == 2:
        source = source[..., None]
    height, width = source.shape[:2]
    x = np.clip(np.asarray(pixels[:, 0], dtype=np.float32), 0, width - 1)
    y = np.clip(np.asarray(pixels[:, 1], dtype=np.float32), 0, height - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fraction_x = (x - x0)[:, None]
    fraction_y = (y - y0)[:, None]
    top = source[y0, x0] * (1.0 - fraction_x) + source[y0, x1] * fraction_x
    bottom = source[y1, x0] * (1.0 - fraction_x) + source[y1, x1] * fraction_x
    return top * (1.0 - fraction_y) + bottom * fraction_y


def _vertex_visibility(
    mesh: trimesh.Trimesh,
    pixels: np.ndarray,
    depth: np.ndarray,
    positive_depth: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    source_height, source_width = image_shape
    width = 512
    height = max(int(round(width * source_height / source_width)), 1)
    scale = np.asarray([width / source_width, height / source_height], dtype=np.float64)
    scaled_pixels = pixels * scale[None, :]
    faces = np.asarray(mesh.faces, dtype=np.int64)
    valid_faces = np.all(positive_depth[faces], axis=1)
    triangles = np.rint(scaled_pixels[faces[valid_faces]]).astype(np.int32)
    face_depth = np.mean(depth[faces[valid_faces]], axis=1)
    order = np.argsort(face_depth)[::-1]
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    for face_index in order:
        triangle = triangles[face_index]
        if (
            triangle[:, 0].max() < 0
            or triangle[:, 1].max() < 0
            or triangle[:, 0].min() >= width
            or triangle[:, 1].min() >= height
        ):
            continue
        cv2.fillConvexPoly(depth_buffer, triangle, float(face_depth[face_index]))
    sample_x = np.clip(np.rint(scaled_pixels[:, 0]).astype(np.int32), 0, width - 1)
    sample_y = np.clip(np.rint(scaled_pixels[:, 1]).astype(np.int32), 0, height - 1)
    nearest_depth = depth_buffer[sample_y, sample_x]
    tolerance = max(float(np.ptp(np.asarray(mesh.vertices), axis=0).max()) / 90.0, 0.009)
    return positive_depth & np.isfinite(nearest_depth) & (depth <= nearest_depth + tolerance)


def _project_vertex_albedo(
    mesh: trimesh.Trimesh,
    cameras: dict[ViewRole, CameraRecord],
    references: dict[ViewRole, Path],
    masks: dict[ViewRole, Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    normal_length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_length, 1e-12)
    head_angle, _, _ = _head_cylindrical_coordinates(vertices)
    azimuth_degrees = np.degrees(head_angle)
    accumulated = np.zeros((len(vertices), 3), dtype=np.float64)
    weight_sum = np.zeros(len(vertices), dtype=np.float64)
    skin_samples: list[np.ndarray] = []

    for role in (ViewRole.FRONT, ViewRole.LEFT45, ViewRole.RIGHT45):
        image_bgr = cv2.imread(str(references[role]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(masks[role]), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or mask is None:
            raise ValueError(f"skin projection input is unreadable: {role.value}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        de_lit = _delight(image, mask)
        foreground = mask > 127
        pixels_for_tone = de_lit[foreground]
        if len(pixels_for_tone):
            luminance = pixels_for_tone @ np.asarray([0.2126, 0.7152, 0.0722])
            lower, upper = np.quantile(luminance, (0.18, 0.82))
            stable = pixels_for_tone[(luminance >= lower) & (luminance <= upper)]
            if len(stable):
                skin_samples.append(stable[:: max(len(stable) // 40_000, 1)])

        camera = cameras[role]
        rotation, _ = cv2.Rodrigues(np.asarray(camera.rotation_vector, dtype=np.float64))
        translation = np.asarray(camera.translation, dtype=np.float64)
        camera_vertices = vertices @ rotation.T + translation
        depth = camera_vertices[:, 2]
        pixels = np.zeros((len(vertices), 2), dtype=np.float64)
        positive_depth = depth > 1e-6
        pixels[positive_depth, 0] = (
            camera_vertices[positive_depth, 0] / depth[positive_depth] * camera.focal_length_px
            + camera.principal_point_px[0]
        )
        pixels[positive_depth, 1] = (
            camera_vertices[positive_depth, 1] / depth[positive_depth] * camera.focal_length_px
            + camera.principal_point_px[1]
        )
        inside = (
            positive_depth
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] <= image.shape[1] - 1)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] <= image.shape[0] - 1)
        )
        visible = _vertex_visibility(mesh, pixels, depth, positive_depth, image.shape[:2])
        distance = cv2.distanceTransform((mask > 127).astype(np.uint8), cv2.DIST_L2, 5)
        mask_support = _sample_image((mask.astype(np.float32) / 255.0)[..., None], pixels)[:, 0]
        edge_support = np.clip(
            _sample_image(distance[..., None], pixels)[:, 0] / max(min(image.shape[:2]) * 0.018, 4),
            0.0,
            1.0,
        )
        camera_center = -rotation.T @ translation
        view = camera_center[None, :] - vertices
        view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
        facing = np.clip(np.sum(normals * view, axis=1), 0.0, 1.0)
        role_boost = 1.18 if role == ViewRole.FRONT else 1.0
        if role == ViewRole.FRONT:
            angular_route = 1.0 - _smoothstep(50.0, 82.0, np.abs(azimuth_degrees))
        elif role == ViewRole.LEFT45:
            angular_route = _smoothstep(16.0, 52.0, azimuth_degrees)
            angular_route *= 1.0 - _smoothstep(138.0, 174.0, azimuth_degrees)
        else:
            angular_route = _smoothstep(16.0, 52.0, -azimuth_degrees)
            angular_route *= 1.0 - _smoothstep(138.0, 174.0, -azimuth_degrees)
        weight = (
            inside.astype(np.float64)
            * visible
            * (mask_support > 0.55)
            * edge_support
            * facing**3.2
            * role_boost
            * angular_route
        )
        sampled = _sample_image(de_lit, pixels)
        accumulated += sampled * weight[:, None]
        weight_sum += weight

    if not skin_samples:
        raise ValueError("skin projection found no foreground colour samples")
    stable_skin = np.concatenate(skin_samples, axis=0)
    fallback = np.median(stable_skin, axis=0).astype(np.float32)
    projected = np.broadcast_to(fallback, (len(vertices), 3)).copy()
    observed = weight_sum > 1e-5
    projected[observed] = (accumulated[observed] / weight_sum[observed, None]).astype(np.float32)
    confidence = np.clip(weight_sum / 1.18, 0.0, 1.0).astype(np.float32)
    lateral_blend = _smoothstep(72.0, 118.0, np.abs(azimuth_degrees)).astype(np.float32)
    projected = projected * (1.0 - lateral_blend[:, None] * 0.72) + fallback * (
        lateral_blend[:, None] * 0.72
    )
    confidence *= 1.0 - lateral_blend * 0.45
    return _linear_to_srgb(projected), confidence, _linear_to_srgb(fallback)


def _estimate_front_alignment(
    front_reference_path: Path,
    uv_albedo_source_path: Path,
) -> tuple[np.ndarray | None, int]:
    front = cv2.imread(str(front_reference_path), cv2.IMREAD_GRAYSCALE)
    atlas = cv2.imread(str(uv_albedo_source_path), cv2.IMREAD_GRAYSCALE)
    if front is None or atlas is None:
        return None, 0
    sift = cv2.SIFT_create(nfeatures=4_000)
    front_keypoints, front_descriptors = sift.detectAndCompute(front, None)
    atlas_keypoints, atlas_descriptors = sift.detectAndCompute(atlas, None)
    if front_descriptors is None or atlas_descriptors is None:
        return None, 0
    matches = cv2.BFMatcher().knnMatch(front_descriptors, atlas_descriptors, k=2)
    reliable = [first for first, second in matches if first.distance < 0.72 * second.distance]
    if len(reliable) < 24:
        return None, 0
    source = np.float32([front_keypoints[match.queryIdx].pt for match in reliable])
    destination = np.float32([atlas_keypoints[match.trainIdx].pt for match in reliable])
    homography, inliers = cv2.findHomography(source, destination, cv2.RANSAC, 4.0)
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    if homography is None or inlier_count < 18 or not np.all(np.isfinite(homography)):
        return None, inlier_count
    return np.asarray(homography, dtype=np.float64), inlier_count


def _front_centered_uv(
    vertices: np.ndarray,
    front_camera: CameraRecord,
    _front_mask: np.ndarray,
    front_alignment: np.ndarray | None,
    atlas_shape: tuple[int, int],
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    angle, cylindrical_u, cylindrical_v = _head_cylindrical_coordinates(vertices)
    if front_alignment is None:
        return np.column_stack((cylindrical_u, cylindrical_v)).astype(np.float32)

    rotation, _ = cv2.Rodrigues(np.asarray(front_camera.rotation_vector, dtype=np.float64))
    camera_vertices = vertices @ rotation.T + np.asarray(front_camera.translation, dtype=np.float64)
    depth = np.maximum(camera_vertices[:, 2], 1e-6)
    front_pixels = np.column_stack(
        (
            camera_vertices[:, 0] / depth * front_camera.focal_length_px
            + front_camera.principal_point_px[0],
            camera_vertices[:, 1] / depth * front_camera.focal_length_px
            + front_camera.principal_point_px[1],
        )
    )
    atlas_pixels = cv2.perspectiveTransform(
        front_pixels.astype(np.float32).reshape(-1, 1, 2),
        front_alignment,
    ).reshape(-1, 2)
    atlas_height, atlas_width = atlas_shape
    projected_u = atlas_pixels[:, 0] / max(atlas_width - 1, 1)
    projected_v = 1.0 - atlas_pixels[:, 1] / max(atlas_height - 1, 1)
    side_blend = _smoothstep(np.radians(28.0), np.radians(50.0), np.abs(angle))
    u = projected_u * (1.0 - side_blend) + cylindrical_u * side_blend
    v = projected_v * (1.0 - side_blend) + cylindrical_v * side_blend

    # Calibrate the atlas against the reconstructed face and standing pinnae
    # independently. Feathered angular and height weights avoid a new UV seam.
    front_alignment_weight = 1.0 - _smoothstep(
        np.radians(28.0),
        np.radians(52.0),
        np.abs(angle),
    )
    front_height_weight = np.exp(-(((cylindrical_v - 0.61) / 0.30) ** 4))
    ear_angle_weight = _smoothstep(
        np.radians(48.0),
        np.radians(68.0),
        np.abs(angle),
    )
    ear_angle_weight *= 1.0 - _smoothstep(
        np.radians(112.0),
        np.radians(140.0),
        np.abs(angle),
    )
    ear_height_weight = np.exp(-(((cylindrical_v - 0.60) / 0.16) ** 4))
    v -= front_alignment_weight * front_height_weight * 0.018
    v += ear_angle_weight * ear_height_weight * 0.026
    u = np.where(np.abs(angle) < np.radians(52.0), np.clip(u, 0.02, 0.98), u)
    v = np.clip(v, 0.02, 0.98)
    return np.column_stack((u, v)).astype(np.float32)


def _duplicate_wrap_seam(
    mesh: trimesh.Trimesh,
    uv: np.ndarray,
) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32).tolist()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32).tolist()
    texture_uv = np.asarray(uv, dtype=np.float32).tolist()
    source_indices = list(range(len(vertices)))
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    duplicate: dict[int, int] = {}
    for face_index, face in enumerate(faces):
        face_u = uv[face, 0]
        if float(np.ptp(face_u)) <= 0.5:
            continue
        for corner, source_index in enumerate(face):
            source_index = int(source_index)
            if uv[source_index, 0] >= 0.5:
                continue
            target_index = duplicate.get(source_index)
            if target_index is None:
                target_index = len(vertices)
                duplicate[source_index] = target_index
                vertices.append(vertices[source_index])
                normals.append(normals[source_index])
                wrapped = list(texture_uv[source_index])
                wrapped[0] += 1.0
                texture_uv.append(wrapped)
                source_indices.append(source_index)
            faces[face_index, corner] = target_index
    wrapped_mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=faces,
        vertex_normals=np.asarray(normals, dtype=np.float32),
        process=False,
        validate=False,
    )
    return (
        wrapped_mesh,
        np.asarray(texture_uv, dtype=np.float32),
        np.asarray(source_indices, dtype=np.int64),
    )


def _mirrored_micro_pattern(source: np.ndarray, resolution: int) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(source, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray /= 255.0
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    low = cv2.GaussianBlur(gray, (0, 0), 13.0)
    detail = np.clip((gray - low) / 0.16, -1.0, 1.0)
    mirrored = np.concatenate((detail, np.fliplr(detail)), axis=1)
    mirrored = np.concatenate((mirrored, np.flipud(mirrored)), axis=0)
    repeats = int(np.ceil(resolution / mirrored.shape[0]))
    return np.tile(mirrored, (repeats, repeats))[:resolution, :resolution]


def _rasterize_atlas(
    mesh: trimesh.Trimesh,
    uv: np.ndarray,
    vertex_color: np.ndarray,
    vertex_confidence: np.ndarray,
    fallback_color: np.ndarray,
    micro_source: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    micro = _mirrored_micro_pattern(micro_source, resolution)
    fallback = np.clip(
        fallback_color[None, None, :] * (1.0 + micro[..., None] * 0.045), 0.0, 1.0
    ).astype(np.float32)
    atlas = fallback.copy()
    confidence = np.zeros((resolution, resolution), dtype=np.float32)
    radius_buffer = np.full((resolution, resolution), -np.inf, dtype=np.float32)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    radial_center = np.median(vertices[:, (0, 2)], axis=0)
    radii = np.linalg.norm(vertices[:, (0, 2)] - radial_center[None, :], axis=1)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    scale = float(resolution - 1)

    for face in faces:
        triangle_uv = uv[face]
        minimum_u = float(np.min(triangle_uv[:, 0]))
        maximum_u = float(np.max(triangle_uv[:, 0]))
        first_offset = int(np.ceil(-maximum_u))
        last_offset = int(np.floor(1.0 - minimum_u))
        for offset in range(first_offset, last_offset + 1):
            x = (triangle_uv[:, 0] + offset) * scale
            y = (1.0 - triangle_uv[:, 1]) * scale
            x0 = max(int(np.floor(np.min(x))), 0)
            x1 = min(int(np.ceil(np.max(x))), resolution - 1)
            y0 = max(int(np.floor(np.min(y))), 0)
            y1 = min(int(np.ceil(np.max(y))), resolution - 1)
            if x1 < x0 or y1 < y0:
                continue
            denominator = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
            if abs(float(denominator)) < 1e-7:
                continue
            xx, yy = np.meshgrid(
                np.arange(x0, x1 + 1, dtype=np.float32) + 0.5,
                np.arange(y0, y1 + 1, dtype=np.float32) + 0.5,
            )
            weight0 = ((y[1] - y[2]) * (xx - x[2]) + (x[2] - x[1]) * (yy - y[2])) / denominator
            weight1 = ((y[2] - y[0]) * (xx - x[2]) + (x[0] - x[2]) * (yy - y[2])) / denominator
            weight2 = 1.0 - weight0 - weight1
            inside = (weight0 >= -1e-4) & (weight1 >= -1e-4) & (weight2 >= -1e-4)
            if not np.any(inside):
                continue
            radial = weight0 * radii[face[0]] + weight1 * radii[face[1]] + weight2 * radii[face[2]]
            current_radius = radius_buffer[y0 : y1 + 1, x0 : x1 + 1]
            update = inside & (radial > current_radius)
            if not np.any(update):
                continue
            color = (
                weight0[..., None] * vertex_color[face[0]]
                + weight1[..., None] * vertex_color[face[1]]
                + weight2[..., None] * vertex_color[face[2]]
            )
            projected_confidence = (
                weight0 * vertex_confidence[face[0]]
                + weight1 * vertex_confidence[face[1]]
                + weight2 * vertex_confidence[face[2]]
            )
            atlas_region = atlas[y0 : y1 + 1, x0 : x1 + 1]
            confidence_region = confidence[y0 : y1 + 1, x0 : x1 + 1]
            atlas_region[update] = color[update]
            confidence_region[update] = projected_confidence[update]
            current_radius[update] = radial[update]

    blend = gaussian_filter(np.clip(confidence / 0.20, 0.0, 1.0), sigma=3.0)
    atlas = atlas * blend[..., None] + fallback * (1.0 - blend[..., None])
    atlas = np.clip(atlas * (1.0 + micro[..., None] * 0.018), 0.0, 1.0)
    return np.rint(atlas * 255.0).astype(np.uint8), confidence


def _detail_maps(
    micro_source: np.ndarray, resolution: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    height = _mirrored_micro_pattern(micro_source, resolution)
    height = gaussian_filter(height, sigma=0.55) - gaussian_filter(height, sigma=2.8)
    scale = max(float(np.quantile(np.abs(height), 0.985)), 1e-6)
    height = np.clip(height / scale, -1.0, 1.0)
    derivative_x = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    derivative_y = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    normal = np.dstack((-derivative_x * 0.72, derivative_y * 0.72, np.ones_like(height)))
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-8)
    normal_rgb = np.rint((normal * 0.5 + 0.5) * 255.0).astype(np.uint8)

    rng = np.random.default_rng(seed)
    noise = gaussian_filter(rng.normal(size=(resolution, resolution)).astype(np.float32), sigma=9.0)
    noise /= max(float(np.std(noise)), 1e-6)
    roughness = np.clip(0.53 + noise * 0.035 + np.abs(height) * 0.055, 0.40, 0.72)
    metallic_roughness = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    metallic_roughness[..., 0] = 255
    metallic_roughness[..., 1] = np.rint(roughness * 255.0).astype(np.uint8)
    return normal_rgb, metallic_roughness


def _write_image(
    path: Path, image: np.ndarray, *, image_format: str, quality: int = 92
) -> Image.Image:
    output = io.BytesIO()
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    save_options: dict[str, Any] = {}
    if image_format == "JPEG":
        save_options.update(quality=quality, subsampling=0, optimize=True)
    else:
        save_options.update(optimize=True, compress_level=9)
    pil_image.save(output, format=image_format, **save_options)
    encoded = output.getvalue()
    atomic_write_bytes(path, encoded)
    embedded = Image.open(io.BytesIO(encoded))
    embedded.load()
    return embedded


def build_skin_component(
    run_dir: Path,
    mesh: trimesh.Trimesh,
    cameras: list[CameraRecord],
    *,
    uv_albedo_source_path: Path,
    micro_source_path: Path,
    atlas_resolution: int,
    detail_resolution: int,
    minimum_observed_vertex_fraction: float,
    jpeg_quality: int,
    seed: int,
) -> SkinComponentResult:
    if not uv_albedo_source_path.is_file():
        raise FileNotFoundError(f"skin UV albedo source is missing: {uv_albedo_source_path}")
    if not micro_source_path.is_file():
        raise FileNotFoundError(f"skin micro-albedo source is missing: {micro_source_path}")
    camera_by_role = {camera.role: camera for camera in cameras}
    references = {
        role: run_dir / "references" / f"{role.value}.png"
        for role in (ViewRole.FRONT, ViewRole.LEFT45, ViewRole.RIGHT45)
    }
    masks = {
        role: run_dir / "working" / "masks" / f"{role.value}.png"
        for role in (ViewRole.FRONT, ViewRole.LEFT45, ViewRole.RIGHT45)
    }
    vertex_color, vertex_confidence, fallback_color = _project_vertex_albedo(
        mesh, camera_by_role, references, masks
    )
    front_mask = cv2.imread(str(masks[ViewRole.FRONT]), cv2.IMREAD_GRAYSCALE)
    if front_mask is None:
        raise ValueError("front mask is unreadable for UV generation")
    front_alignment, front_alignment_inliers = _estimate_front_alignment(
        references[ViewRole.FRONT], uv_albedo_source_path
    )
    uv_albedo_opened = Image.open(uv_albedo_source_path).convert("RGB")
    uv = _front_centered_uv(
        np.asarray(mesh.vertices),
        camera_by_role[ViewRole.FRONT],
        front_mask,
        front_alignment,
        (uv_albedo_opened.height, uv_albedo_opened.width),
    )
    wrapped_mesh, wrapped_uv, source_indices = _duplicate_wrap_seam(mesh, uv)
    wrapped_color = vertex_color[source_indices]
    wrapped_confidence = vertex_confidence[source_indices]
    micro_source = np.asarray(Image.open(micro_source_path).convert("RGB"), dtype=np.uint8)
    projected_atlas, atlas_confidence = _rasterize_atlas(
        wrapped_mesh,
        wrapped_uv,
        wrapped_color,
        wrapped_confidence,
        fallback_color,
        micro_source,
        atlas_resolution,
    )
    uv_albedo_source = np.asarray(Image.open(uv_albedo_source_path).convert("RGB"), dtype=np.uint8)
    uv_albedo_source = cv2.resize(
        uv_albedo_source,
        (atlas_resolution, atlas_resolution),
        interpolation=cv2.INTER_LANCZOS4,
    )
    neutral_albedo = _delight(
        uv_albedo_source,
        np.full(uv_albedo_source.shape[:2], 255, dtype=np.uint8),
        strength=0.42,
    )
    atlas_y, atlas_x = np.mgrid[:atlas_resolution, :atlas_resolution]
    normalized_x = atlas_x / max(atlas_resolution - 1, 1)
    normalized_y = atlas_y / max(atlas_resolution - 1, 1)
    face_radius = np.sqrt(((normalized_x - 0.5) / 0.25) ** 2 + ((normalized_y - 0.47) / 0.43) ** 2)
    face_keep = 1.0 - _smoothstep(0.72, 1.24, face_radius)
    ear_keep = np.zeros_like(face_keep)
    for ear_x in (0.18, 0.82):
        ear_radius = np.sqrt(
            ((normalized_x - ear_x) / 0.095) ** 2 + ((normalized_y - 0.39) / 0.145) ** 2
        )
        ear_keep = np.maximum(ear_keep, 1.0 - _smoothstep(0.80, 1.20, ear_radius))
    identity_keep = np.maximum(face_keep, ear_keep)
    broad_skin = cv2.GaussianBlur(
        neutral_albedo,
        (0, 0),
        max(atlas_resolution / 15.0, 24.0),
    )
    neutral_albedo = neutral_albedo * identity_keep[..., None] + broad_skin * (
        1.0 - identity_keep[..., None]
    )
    central = neutral_albedo[
        round(atlas_resolution * 0.18) : round(atlas_resolution * 0.82),
        round(atlas_resolution * 0.22) : round(atlas_resolution * 0.78),
    ]
    central_median = np.median(central.reshape(-1, 3), axis=0)
    target_linear = _srgb_to_linear(fallback_color)
    tone_gain = np.clip(target_linear / np.maximum(central_median, 1e-4), 0.82, 1.18)
    neutral_albedo = np.clip(neutral_albedo * tone_gain[None, None, :], 0.0, 1.0)
    # Match only broad observed-view colour. Keeping the generated atlas's
    # high-frequency identity detail while feathering a low-frequency tone
    # correction avoids another visible face/scalp patch boundary.
    projected_linear = _srgb_to_linear(projected_atlas.astype(np.float32) / 255.0)
    low_sigma = max(atlas_resolution / 72.0, 8.0)
    base_low = cv2.GaussianBlur(neutral_albedo, (0, 0), low_sigma)
    projected_low = cv2.GaussianBlur(projected_linear, (0, 0), low_sigma)
    projection_weight = gaussian_filter(
        np.clip(atlas_confidence / 0.18, 0.0, 1.0),
        sigma=max(atlas_resolution / 180.0, 3.0),
    )
    low_frequency_gain = np.clip(
        (projected_low + 0.015) / np.maximum(base_low + 0.015, 1e-4),
        0.84,
        1.16,
    )
    low_frequency_gain = 1.0 + (low_frequency_gain - 1.0) * (projection_weight[..., None] * 0.42)
    neutral_albedo = np.clip(neutral_albedo * low_frequency_gain, 0.0, 1.0)
    micro = _mirrored_micro_pattern(micro_source, atlas_resolution)
    atlas = (
        np.rint(_linear_to_srgb(neutral_albedo) * (1.0 + micro[..., None] * 0.012) * 255.0)
        .clip(0, 255)
        .astype(np.uint8)
    )
    normal, metallic_roughness = _detail_maps(micro_source, detail_resolution, seed)

    textures = run_dir / "textures"
    atlas_path = textures / "skin-atlas.jpg"
    confidence_path = textures / "skin-confidence.png"
    atlas_image = _write_image(atlas_path, atlas, image_format="JPEG", quality=jpeg_quality)
    normal_image = _write_image(textures / "skin-normal.png", normal, image_format="PNG")
    metallic_roughness_image = _write_image(
        textures / "skin-metallic-roughness.png",
        metallic_roughness,
        image_format="PNG",
    )
    confidence_rgb = np.repeat(
        np.rint(np.clip(atlas_confidence, 0.0, 1.0) * 255.0).astype(np.uint8)[..., None],
        3,
        axis=2,
    )
    _write_image(confidence_path, confidence_rgb, image_format="PNG")

    material = trimesh.visual.material.PBRMaterial(
        name="face-skin-pbr-v1",
        baseColorFactor=(1.0, 1.0, 1.0, 1.0),
        baseColorTexture=atlas_image,
        metallicFactor=0.0,
        roughnessFactor=1.0,
        metallicRoughnessTexture=metallic_roughness_image,
        normalTexture=normal_image,
        doubleSided=False,
    )
    wrapped_mesh.visual = trimesh.visual.TextureVisuals(uv=wrapped_uv, material=material)
    wrapped_mesh.metadata["name"] = "projected-face-skin"
    model_path = run_dir / "models" / "skin.glb"
    atomic_write_bytes(
        model_path,
        trimesh.exchange.gltf.export_glb(wrapped_mesh, include_normals=True),
    )

    observed_vertex_fraction = float(np.mean(vertex_confidence >= 0.12))
    metrics = {
        "representation": "single-cylindrical-uv-atlas-with-centered-front-face",
        "transitionStrategy": "head-centred-ear-aware-uv-with-feathered-multiview-tone",
        "model": "models/skin.glb",
        "atlas": "textures/skin-atlas.jpg",
        "confidenceMap": "textures/skin-confidence.png",
        "atlasResolution": [atlas_resolution, atlas_resolution],
        "detailResolution": [detail_resolution, detail_resolution],
        "frontFaceUCenter": 0.5,
        "frontAlignmentMethod": "sift-homography"
        if front_alignment is not None
        else "cylindrical-fallback",
        "frontAlignmentInliers": front_alignment_inliers,
        "observedVertexFraction": observed_vertex_fraction,
        "meanProjectionConfidence": float(np.mean(vertex_confidence)),
        "atlasObservedFraction": float(np.mean(atlas_confidence >= 0.12)),
        "seamDuplicateVertices": int(len(wrapped_mesh.vertices) - len(mesh.vertices)),
        "inferredRegions": ["rearCranium", "shortNeck"],
        "sourceViews": [role.value for role in camera_by_role],
        "microAlbedoSourceSha256": sha256_file(micro_source_path),
        "uvAlbedoSourceSha256": sha256_file(uv_albedo_source_path),
        "modelSha256": sha256_file(model_path),
        "atlasSha256": sha256_file(atlas_path),
        "confidenceSha256": sha256_file(confidence_path),
        "independentPbrChannels": ["albedo", "normal", "roughness", "metalness"],
        "passed": bool(
            observed_vertex_fraction >= minimum_observed_vertex_fraction
            and np.all(np.isfinite(vertex_confidence))
            and np.all(np.isfinite(wrapped_uv))
        ),
    }
    return SkinComponentResult(metrics=metrics)
