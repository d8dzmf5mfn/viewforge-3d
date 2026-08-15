from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

from face3d.io import atomic_write_bytes, atomic_write_json, sha256_file
from face3d.models import CameraRecord
from face3d.render import render_flat_mesh
from face3d.report import _canonical_side_camera, _skin_preview_mesh
from face3d.unified_head import UnifiedHeadAsset, geometry_hash

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = ROOT / "assets/template-head-v0/anatomy/template-head-v0.unified.npz"
DEFAULT_SOURCE_MODEL = ROOT / ".local/demo-assets/LeePerrySmith.glb"
DEFAULT_SOURCE_ALBEDO = ROOT / ".local/demo-assets/Map-COL.jpg"
DEFAULT_SOURCE_LICENSE = ROOT / ".local/demo-assets/LeePerrySmith_License.txt"
DEFAULT_OUTPUT = ROOT / "assets/template-head-v0/preview"


def _single_mesh(path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(path, force="scene")
    meshes = [
        geometry
        for geometry in scene.geometry.values()
        if isinstance(geometry, trimesh.Trimesh)
    ]
    if len(meshes) != 1:
        raise ValueError(f"source GLB must contain one mesh, found {len(meshes)}: {path}")
    mesh = meshes[0]
    if getattr(mesh.visual, "uv", None) is None:
        raise ValueError(f"source GLB has no texture coordinates: {path}")
    return mesh


def _source_surface_metrics(
    head: UnifiedHeadAsset,
    source: trimesh.Trimesh,
) -> dict[str, float]:
    source_vertices = np.asarray(source.vertices, dtype=np.float32)
    source_faces = np.asarray(source.faces, dtype=np.int64)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(source_vertices),
        o3d.core.Tensor(source_faces.astype(np.uint32)),
    )
    query = scene.compute_closest_points(
        o3d.core.Tensor(np.asarray(head.render_vertices, dtype=np.float32))
    )
    closest = query["points"].numpy().astype(np.float64)
    distance = np.linalg.norm(closest - head.render_vertices, axis=1)
    diagonal = max(float(np.linalg.norm(np.ptp(head.skin_vertices, axis=0))), 1e-12)
    return {
        "meanSurfaceDistanceHeadDiagonal": float(np.mean(distance) / diagonal),
        "p99SurfaceDistanceHeadDiagonal": float(np.quantile(distance, 0.99) / diagonal),
        "maximumSurfaceDistanceHeadDiagonal": float(np.max(distance) / diagonal),
    }


def _sample_bilinear(
    image: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.mod(u, 1.0) * (width - 1)
    y = (1.0 - np.clip(v, 0.0, 1.0)) * (height - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    x_weight = x - x0
    y_weight = y - y0
    return (
        image[y0, x0] * (1.0 - x_weight)[:, None] * (1.0 - y_weight)[:, None]
        + image[y0, x1] * x_weight[:, None] * (1.0 - y_weight)[:, None]
        + image[y1, x0] * (1.0 - x_weight)[:, None] * y_weight[:, None]
        + image[y1, x1] * x_weight[:, None] * y_weight[:, None]
    )


def _bake_canonical_atlas(
    head: UnifiedHeadAsset,
    source: trimesh.Trimesh,
    source_albedo: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, dict[str, float]]:
    fallback = np.median(source_albedo.reshape(-1, 3), axis=0)
    atlas = np.broadcast_to(fallback, (resolution, resolution, 3)).copy()
    coverage = np.zeros((resolution, resolution), dtype=np.uint8)
    scale = resolution - 1
    source_vertices = np.asarray(source.vertices, dtype=np.float32)
    source_faces = np.asarray(source.faces, dtype=np.int64)
    source_uv = np.asarray(source.visual.uv, dtype=np.float64)
    source_scene = o3d.t.geometry.RaycastingScene()
    source_scene.add_triangles(
        o3d.core.Tensor(source_vertices),
        o3d.core.Tensor(source_faces.astype(np.uint32)),
    )
    atlas_flat = atlas.reshape(-1, 3)
    coverage_flat = coverage.reshape(-1)
    pending_pixels: list[np.ndarray] = []
    pending_points: list[np.ndarray] = []
    pending_count = 0

    def flush() -> None:
        nonlocal pending_count
        if not pending_count:
            return
        pixel_indices = np.concatenate(pending_pixels)
        points = np.concatenate(pending_points).astype(np.float32, copy=False)
        query = source_scene.compute_closest_points(o3d.core.Tensor(points))
        primitive = query["primitive_ids"].numpy().astype(np.int64)
        barycentric_12 = query["primitive_uvs"].numpy().astype(np.float64)
        barycentric = np.column_stack(
            (1.0 - barycentric_12.sum(axis=1), barycentric_12)
        )
        mapped_uv = np.einsum(
            "ni,nij->nj",
            barycentric,
            source_uv[source_faces[primitive]],
            optimize=True,
        )
        atlas_flat[pixel_indices] = _sample_bilinear(
            source_albedo,
            mapped_uv[:, 0],
            mapped_uv[:, 1],
        )
        coverage_flat[pixel_indices] = 255
        pending_pixels.clear()
        pending_points.clear()
        pending_count = 0

    for face in np.asarray(head.render_faces, dtype=np.int64):
        target_uv = np.asarray(head.uv[face], dtype=np.float64)
        x = target_uv[:, 0] * scale
        y = (1.0 - target_uv[:, 1]) * scale
        x0 = max(int(np.floor(x.min())), 0)
        x1 = min(int(np.ceil(x.max())), scale)
        y0 = max(int(np.floor(y.min())), 0)
        y1 = min(int(np.ceil(y.max())), scale)
        if x1 < x0 or y1 < y0:
            continue
        denominator = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (
            y[0] - y[2]
        )
        if abs(float(denominator)) < 1e-8:
            continue
        xx, yy = np.meshgrid(
            np.arange(x0, x1 + 1, dtype=np.float64) + 0.5,
            np.arange(y0, y1 + 1, dtype=np.float64) + 0.5,
        )
        weight0 = (
            (y[1] - y[2]) * (xx - x[2]) + (x[2] - x[1]) * (yy - y[2])
        ) / denominator
        weight1 = (
            (y[2] - y[0]) * (xx - x[2]) + (x[0] - x[2]) * (yy - y[2])
        ) / denominator
        weight2 = 1.0 - weight0 - weight1
        inside = (
            (weight0 >= -1e-5)
            & (weight1 >= -1e-5)
            & (weight2 >= -1e-5)
        )
        if not np.any(inside):
            continue

        rows, columns = np.nonzero(inside)
        pixel_indices = (rows + y0) * resolution + columns + x0
        triangle = np.asarray(head.render_vertices[face], dtype=np.float64)
        points = (
            weight0[..., None] * triangle[0]
            + weight1[..., None] * triangle[1]
            + weight2[..., None] * triangle[2]
        )[inside]
        pending_pixels.append(pixel_indices.astype(np.int64, copy=False))
        pending_points.append(points.astype(np.float32, copy=False))
        pending_count += len(pixel_indices)
        if pending_count >= 250_000:
            flush()

    flush()

    used_fraction = float(np.mean(coverage > 0))
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(8):
        expanded = cv2.dilate(coverage, kernel, iterations=1)
        gutter = (coverage == 0) & (expanded > 0)
        for channel in range(3):
            dilated = cv2.dilate(
                atlas[..., channel].astype(np.float32), kernel, iterations=1
            )
            atlas[..., channel][gutter] = dilated[gutter]
        coverage = expanded

    return np.clip(np.rint(atlas), 0, 255).astype(np.uint8), {
        "canonicalUvUsedFraction": used_fraction,
        "canonicalUvWithGutterFraction": float(np.mean(coverage > 0)),
        "gutterPixels": 8,
    }


def _jpeg_bytes(image: np.ndarray, quality: int = 95) -> bytes:
    output = io.BytesIO()
    Image.fromarray(image).save(
        output,
        format="JPEG",
        quality=quality,
        subsampling=0,
        optimize=True,
    )
    return output.getvalue()


def _render_flat_views(
    output: Path,
    head: UnifiedHeadAsset,
    atlas_path: Path,
) -> list[str]:
    manifest = json.loads(
        (ROOT / "assets/template-head-v0/manifest.json").read_text(encoding="utf-8")
    )
    mesh = _skin_preview_mesh(head, atlas_path)
    rendered: list[str] = []
    for role in ("front", "left45", "right45"):
        destination = output / "qa" / f"flat-{role}.png"
        render_flat_mesh(
            mesh,
            CameraRecord.model_validate(manifest["cameras"][role]),
            destination,
            use_mesh_face_colors=True,
        )
        rendered.append(destination.relative_to(output).as_posix())
    destination = output / "qa" / "flat-side.png"
    render_flat_mesh(
        mesh,
        _canonical_side_camera(mesh),
        destination,
        use_mesh_face_colors=True,
    )
    rendered.append(destination.relative_to(output).as_posix())
    return rendered


def create(
    output: Path,
    *,
    template: Path = DEFAULT_TEMPLATE,
    source_model: Path = DEFAULT_SOURCE_MODEL,
    source_albedo: Path = DEFAULT_SOURCE_ALBEDO,
    source_license: Path = DEFAULT_SOURCE_LICENSE,
    resolution: int = 2048,
) -> dict[str, Any]:
    required = (template, source_model, source_albedo, source_license)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing local preview inputs: {missing}")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    head = UnifiedHeadAsset.load(template)
    geometry_before = geometry_hash(head.render_vertices, head.render_faces)
    canonical_uv_before = np.asarray(head.uv, dtype=np.float32).copy()
    source = _single_mesh(source_model)
    transfer_metrics = _source_surface_metrics(head, source)
    source_image = np.asarray(Image.open(source_albedo).convert("RGB"), dtype=np.float64)
    atlas, atlas_metrics = _bake_canonical_atlas(
        head,
        source,
        source_image,
        resolution,
    )

    atlas_path = output / "head-albedo.jpg"
    model_path = output / "head-skinned.glb"
    atomic_write_bytes(atlas_path, _jpeg_bytes(atlas))
    head.export_head_glb(model_path, Image.open(atlas_path).convert("RGB"))
    flat_views = _render_flat_views(output, head, atlas_path)

    geometry_after = geometry_hash(head.render_vertices, head.render_faces)
    uv_maximum_difference = float(
        np.max(np.abs(np.asarray(head.uv, dtype=np.float32) - canonical_uv_before))
    )
    if geometry_after != geometry_before or uv_maximum_difference != 0.0:
        raise RuntimeError("skin preview changed TemplateHeadV0 geometry or canonical UV")

    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "name": "TemplateHeadV0 licensed skin preview",
        "previewOnly": True,
        "notProductionThreeViewProjection": True,
        "template": str(template.relative_to(ROOT)),
        "geometryHash": geometry_after,
        "geometryChanged": False,
        "canonicalUvChanged": False,
        "canonicalUvMaximumDifference": uv_maximum_difference,
        "source": {
            "model": str(source_model.relative_to(ROOT)),
            "modelSha256": sha256_file(source_model),
            "albedo": str(source_albedo.relative_to(ROOT)),
            "albedoSha256": sha256_file(source_albedo),
            "license": str(source_license.relative_to(ROOT)),
            "licenseSha256": sha256_file(source_license),
        },
        "method": "closest-surface-source-uv-transfer-baked-to-canonical-uv",
        "atlasResolution": [resolution, resolution],
        **transfer_metrics,
        **atlas_metrics,
        "artifacts": {
            "model": model_path.relative_to(output).as_posix(),
            "modelSha256": sha256_file(model_path),
            "atlas": atlas_path.relative_to(output).as_posix(),
            "atlasSha256": sha256_file(atlas_path),
            "flatViews": flat_views,
        },
        "nextProductionInput": "authorized front/left45/right45 photos",
    }
    atomic_write_json(output / "manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bake the licensed Lee Perry-Smith albedo onto TemplateHeadV0 canonical UV."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=int, default=2048, choices=(1024, 2048, 4096))
    arguments = parser.parse_args()
    print(
        json.dumps(
            create(arguments.output, resolution=arguments.resolution),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
