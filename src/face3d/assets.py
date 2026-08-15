from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import cKDTree

from face3d.config import Face3DConfig
from face3d.errors import fail
from face3d.io import atomic_write_bytes, atomic_write_json, read_json, sha256_file
from face3d.profiles.face_v1 import MEDIAPIPE_TO_IBUG68
from face3d.skin import _duplicate_wrap_seam, _head_cylindrical_coordinates
from face3d.stages.flame import FlameModel, FlameRegionMasks

ASSET_LICENSES = {
    "flameModel": "FLAME-2023-Open-CC-BY-4.0",
    "flameLandmarks": "FLAME-2023-Open-CC-BY-4.0",
    "flameMasks": "FLAME-2023-Open-CC-BY-4.0",
    "flamePrepared": "Derived-from-FLAME-2023-Open-CC-BY-4.0",
    "templateHead": "Lee-Perry-Smith-CC-BY-3.0-derived",
    "templateLandmarks": "Lee-Perry-Smith-CC-BY-3.0-derived",
    "templateManifest": "Lee-Perry-Smith-CC-BY-3.0-derived",
    "faceLandmarker": "Apache-2.0",
    "canonicalFaceModel": "Apache-2.0",
}

CORE_ASSETS = ("faceLandmarker", "canonicalFaceModel")
V2_ASSETS = (
    "faceLandmarker",
    "canonicalFaceModel",
    "flameModel",
    "flameLandmarks",
    "flameMasks",
    "flamePrepared",
)
V3_ASSETS = (
    "faceLandmarker",
    "canonicalFaceModel",
    "templateHead",
    "templateLandmarks",
    "templateManifest",
)
OPTIONAL_ASSETS = (
    "flameModel",
    "flameLandmarks",
    "flameMasks",
    "flamePrepared",
    "templateHead",
    "templateLandmarks",
    "templateManifest",
)


def asset_paths(config: Face3DConfig) -> dict[str, Path]:
    configured = {
        "flameModel": config.assets.flame_model,
        "flameLandmarks": config.assets.flame_landmarks,
        "flameMasks": config.assets.flame_masks,
        "flamePrepared": config.assets.flame_prepared,
        "templateHead": config.assets.template_head,
        "templateLandmarks": config.assets.template_landmarks,
        "templateManifest": config.assets.template_manifest,
        "faceLandmarker": config.resolve_asset(config.assets.face_landmarker),
        "canonicalFaceModel": config.resolve_asset(config.assets.canonical_face_model),
    }
    return {
        name: value
        if isinstance(value, Path) and value.is_absolute()
        else config.resolve_asset(value)
        for name, value in configured.items()
        if value is not None
    }


def required_asset_names(config: Face3DConfig, *, prepared: bool = True) -> tuple[str, ...]:
    if config.is_v3:
        return V3_ASSETS
    if config.is_v2:
        return (
            V2_ASSETS if prepared else tuple(name for name in V2_ASSETS if name != "flamePrepared")
        )
    return CORE_ASSETS


def manifest_path(config: Face3DConfig) -> Path:
    return config.project_root / ".local" / "models" / "manifest.json"


def record_assets(config: Face3DConfig) -> dict[str, Any]:
    paths = asset_paths(config)
    required = required_asset_names(config, prepared=False)
    missing = [
        str(paths[name]) for name in required if name not in paths or not paths[name].is_file()
    ]
    if missing:
        fail(
            "asset-missing",
            "无法记录模型哈希：缺少配置要求的模型资产",
            stage="assets",
            details={"missing": missing},
        )
    models = {
        name: {
            "path": path.relative_to(config.project_root).as_posix()
            if path.is_relative_to(config.project_root)
            else str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "license": ASSET_LICENSES[name],
        }
        for name, path in sorted(paths.items())
        if path.is_file()
    }
    existing_models: dict[str, Any] = {}
    destination = manifest_path(config)
    if destination.is_file():
        existing = read_json(destination).get("models", {})
        if isinstance(existing, dict):
            existing_models = dict(existing)
    existing_models.update(models)
    payload = {
        "schemaVersion": 1,
        "recordedAt": datetime.now(UTC).isoformat(),
        "models": dict(sorted(existing_models.items())),
    }
    atomic_write_json(destination, payload)
    return payload


def asset_status(config: Face3DConfig, *, require_recorded: bool = False) -> dict[str, Any]:
    recorded: dict[str, Any] = {}
    path = manifest_path(config)
    if path.is_file():
        recorded = read_json(path).get("models", {})
    statuses: dict[str, Any] = {}
    required = set(required_asset_names(config))
    core_ready = True
    optional_ready = True
    for name, model_path in asset_paths(config).items():
        exists = model_path.is_file()
        actual_hash = sha256_file(model_path) if exists else None
        expected_hash = recorded.get(name, {}).get("sha256") if isinstance(recorded, dict) else None
        hash_matches = bool(actual_hash and expected_hash and actual_hash == expected_hash)
        item_ready = exists and (hash_matches or not require_recorded)
        if name in required:
            core_ready = core_ready and item_ready
        else:
            optional_ready = optional_ready and item_ready
        statuses[name] = {
            "path": str(model_path),
            "exists": exists,
            "sha256": actual_hash,
            "recordedSha256": expected_hash,
            "hashMatches": hash_matches,
            "license": ASSET_LICENSES[name],
        }
    return {
        "ready": core_ready,
        "coreReady": core_ready,
        "optionalReady": optional_ready,
        "coreAssets": sorted(required),
        "optionalAssets": sorted(set(statuses) - required),
        "manifest": str(path),
        "models": statuses,
    }


def require_assets(
    config: Face3DConfig,
    names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    names = required_asset_names(config) if names is None else names
    status = asset_status(config, require_recorded=True)
    missing = [
        name
        for name in names
        if name not in status["models"]
        or not status["models"][name]["exists"]
        or not status["models"][name]["hashMatches"]
    ]
    if missing:
        fail(
            "asset-missing",
            "本地模型缺失、未记录或哈希不一致",
            stage="assets",
            details={**status, "required": list(names), "failed": missing},
        )
    return status


def prepare_face_v2(config: Face3DConfig) -> dict[str, Any]:
    if not config.is_v2:
        fail("config-invalid", "prepare-face-v2 只接受 Face v2 配置", stage="assets")
    require_assets(config, names=required_asset_names(config, prepared=False))
    try:
        import xatlas
    except ImportError:
        fail(
            "dependency-missing",
            "Face v2 UV 准备需要 xatlas；请先执行 uv sync",
            stage="assets",
        )
    paths = asset_paths(config)
    flame = FlameModel.load(
        paths["flameModel"], paths["flameLandmarks"], config.fit.shape_coefficients
    )
    regions = FlameRegionMasks.load(paths["flameMasks"], flame.vertices_template)
    canonical = trimesh.load(paths["canonicalFaceModel"], force="mesh", process=False)
    canonical_vertices = np.asarray(canonical.vertices, dtype=np.float64)
    if len(canonical_vertices) < 468:
        fail(
            "asset-invalid",
            "MediaPipe canonical face model 顶点数不足 468",
            stage="assets",
            details={"measured": int(len(canonical_vertices))},
        )
    canonical_vertices = canonical_vertices[:468]
    source_68 = canonical_vertices[np.asarray(MEDIAPIPE_TO_IBUG68, dtype=np.int64)]
    target_68 = flame.landmark_vertices(flame.vertices_template, 0.0)
    source_center = source_68.mean(axis=0)
    target_center = target_68.mean(axis=0)
    rotation, scale_sum = orthogonal_procrustes(
        source_68 - source_center,
        target_68 - target_center,
    )
    scale = float(scale_sum / np.sum((source_68 - source_center) ** 2))
    aligned_canonical = (canonical_vertices - source_center) @ rotation * scale + target_center
    dense_vertex_index = cKDTree(flame.vertices_template).query(aligned_canonical, k=1)[1]
    eye_centers = np.stack(flame.eye_centers(flame.vertices_template))
    base_regions = regions.as_dict()
    vertices = np.asarray(flame.vertices_template, dtype=np.float64)
    faces = np.asarray(flame.faces, dtype=np.int64)
    for _ in range(config.anatomy.subdivision_levels):
        vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    source_index = cKDTree(flame.vertices_template).query(vertices, k=1)[1].astype(np.int32)
    expanded_regions = {
        name: np.flatnonzero(np.isin(source_index, indices)).astype(np.int32)
        for name, indices in base_regions.items()
    }
    xatlas_mapping, xatlas_faces, xatlas_uv = xatlas.parametrize(
        vertices.astype(np.float32), faces.astype(np.uint32)
    )
    if not len(xatlas_mapping) or not len(xatlas_faces) or not np.isfinite(xatlas_uv).all():
        fail("asset-invalid", "xatlas 未生成有效 Face v2 UV", stage="assets")
    _, cylindrical_u, cylindrical_v = _head_cylindrical_coordinates(vertices)
    canonical_uv = np.column_stack((cylindrical_u, cylindrical_v)).astype(np.float32)
    canonical_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    wrapped_mesh, uv, render_to_subdiv = _duplicate_wrap_seam(canonical_mesh, canonical_uv)
    render_faces = np.asarray(wrapped_mesh.faces, dtype=np.int32)
    render_to_subdiv = np.asarray(render_to_subdiv, dtype=np.int32)
    radii = []
    for center, indices in zip(
        eye_centers,
        (regions.left_eyelid, regions.right_eyelid),
        strict=True,
    ):
        distances = np.linalg.norm(flame.vertices_template[indices] - center, axis=1)
        radii.append(float(np.quantile(distances, 0.35)))
    output = io.BytesIO()
    np.savez_compressed(
        output,
        template_vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        source_index=source_index,
        render_to_subdiv=render_to_subdiv,
        render_faces=render_faces,
        uv=uv,
        uv_layout=np.asarray("central-face-separated-ear-columns"),
        uv_parameterizer=np.asarray("canonical-cylindrical-validated-by-xatlas"),
        eye_centers=eye_centers.astype(np.float32),
        eye_radii=np.asarray(radii, dtype=np.float32),
        dense_vertex_index=np.asarray(dense_vertex_index, dtype=np.int32),
        **{f"region_{name}": value for name, value in expanded_regions.items()},
    )
    destination = paths["flamePrepared"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(destination, output.getvalue())
    recorded = record_assets(config)
    return {
        "ok": True,
        "output": str(destination),
        "sha256": sha256_file(destination),
        "vertices": int(len(vertices)),
        "triangles": int(len(faces)),
        "renderVertices": int(len(render_to_subdiv)),
        "recorded": recorded,
    }
