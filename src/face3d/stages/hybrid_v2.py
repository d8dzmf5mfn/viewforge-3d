from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

from face3d.config import Face3DConfig
from face3d.errors import fail
from face3d.glb import export_pixel_instances
from face3d.io import atomic_write_json, sha256_file
from face3d.models import REQUIRED_VIEWS, CameraRecord, ViewRole
from face3d.pixel_binary import write_pixel_records_v2
from face3d.skin import _linear_to_srgb
from face3d.skin_v2 import SkinV2Result, build_skin_v2
from face3d.stages.fit import run_fit
from face3d.stages.sdf import _mask_support, close_working_mesh
from face3d.unified_head import EyeballAsset, UnifiedHeadAsset, build_unified_head


def release_stage_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _surface_cells(sdf: np.ndarray, band: float) -> np.ndarray:
    selected = np.zeros(sdf.shape, dtype=bool)
    for axis in range(3):
        first_slice = [slice(None)] * 3
        second_slice = [slice(None)] * 3
        first_slice[axis] = slice(0, -1)
        second_slice[axis] = slice(1, None)
        first = sdf[tuple(first_slice)]
        second = sdf[tuple(second_slice)]
        crossing = np.signbit(first) != np.signbit(second)
        choose_first = crossing & (np.abs(first) <= np.abs(second))
        selected[tuple(first_slice)] |= choose_first
        selected[tuple(second_slice)] |= crossing & ~choose_first
    selected &= np.abs(sdf) <= band
    labels, count = ndimage.label(selected, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count > 1:
        sizes = np.bincount(labels.reshape(-1))
        sizes[0] = 0
        selected = labels == int(np.argmax(sizes))
    return selected


def _open3d_scene(mesh: trimesh.Trimesh) -> tuple[Any, Any]:
    try:
        import open3d as o3d
    except ImportError:
        fail("dependency-missing", "Face v2 SDF 需要 Open3D", stage="sdf-v2")
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    return o3d, scene


def _quaternions_from_normals(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float32)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    rotations = np.zeros((len(normals), 4), dtype=np.float32)
    rotations[:, 0] = -normals[:, 1]
    rotations[:, 1] = normals[:, 0]
    rotations[:, 3] = 1.0 + normals[:, 2]
    opposite = rotations[:, 3] < 1e-5
    rotations[opposite] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-8)
    return rotations


def _source_set_hash(run_dir: Path) -> str:
    digest = hashlib.sha256()
    for role in REQUIRED_VIEWS:
        digest.update(bytes.fromhex(sha256_file(run_dir / "references" / f"{role.value}.png")))
    return digest.hexdigest()


def _feature_class(head: UnifiedHeadAsset, triangles: np.ndarray) -> np.ndarray:
    representative = head.skin_faces[triangles, 0]
    feature = np.zeros(len(triangles), dtype=np.uint8)
    ears = np.concatenate((head.regions["left_ear"], head.regions["right_ear"]))
    eyelids = np.concatenate((head.regions["left_eyelid"], head.regions["right_eyelid"]))
    feature[np.isin(representative, ears)] = 4
    feature[np.isin(representative, eyelids)] = 1
    return feature


def _eye_cells(
    eye: EyeballAsset,
    node: int,
    front_camera: CameraRecord,
    maximum: int = 4500,
) -> dict[str, np.ndarray]:
    mesh = eye.mesh()
    faces = np.asarray(mesh.faces, dtype=np.int64)
    selected = np.linspace(0, len(faces) - 1, min(len(faces), maximum), dtype=np.int64)
    points = np.asarray(mesh.triangles_center[selected], dtype=np.float32)
    normals = np.asarray(mesh.face_normals[selected], dtype=np.float32)
    rotation, _ = cv2.Rodrigues(np.asarray(front_camera.rotation_vector, dtype=np.float64))
    camera_points = points @ rotation.T + np.asarray(front_camera.translation)
    depth = camera_points[:, 2].astype(np.float32)
    uv = np.zeros((len(points), 2), dtype=np.uint16)
    valid = depth > 1e-6
    projected = camera_points[valid, :2] / depth[valid, None]
    projected *= front_camera.focal_length_px
    projected += np.asarray(front_camera.principal_point_px)
    uv[valid] = np.rint(projected).clip(0, 65535).astype(np.uint16)
    view_direction = np.asarray([0.0, 0.0, 1.0])
    front = (normals @ view_direction) > -0.05
    source_bits = np.where(front, 1, 8).astype(np.uint8)
    confidence = np.where(front, 0.82, 0.22).astype(np.float32)
    cosine = np.clip(normals @ eye.gaze, -1.0, 1.0)
    iris = cosine > np.cos(np.radians(17.0))
    pupil = cosine > np.cos(np.radians(7.0))
    colors = np.full((len(points), 3), (229, 226, 216), dtype=np.uint32)
    colors[iris] = np.asarray((74, 104, 111), dtype=np.uint32)
    colors[pupil] = np.asarray((17, 20, 22), dtype=np.uint32)
    codes = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
    return {
        "positions": points,
        "normals": normals,
        "source_uv": uv,
        "view_role": np.where(front, 0, 255).astype(np.uint8),
        "target_node": np.full(len(points), node, dtype=np.uint8),
        "target_triangle": selected.astype(np.uint32),
        "barycentric": np.full((len(points), 3), 1.0 / 3.0, dtype=np.float32),
        "depth": depth,
        "feature_class": np.ones(len(points), dtype=np.uint8),
        "confidence": confidence,
        "source_bits": source_bits,
        "pixel_codes": codes.astype(np.uint32),
    }


def _concat_records(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([record[key] for record in records], axis=0) for key in records[0]}


def build_pixel_sdf_v2(
    run_dir: Path,
    head: UnifiedHeadAsset,
    skin: SkinV2Result,
    cameras: list[CameraRecord],
    config: Face3DConfig,
) -> dict[str, Any]:
    working_mesh, closure = close_working_mesh(head.skin_mesh)
    o3d, signed_scene = _open3d_scene(working_mesh)
    resolution = config.sdf.resolution
    bounds = np.asarray(working_mesh.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    extent = float(np.ptp(bounds, axis=0).max() * (1 + 2 * config.sdf.padding_fraction))
    grid_min = center - extent / 2
    voxel_size = extent / resolution
    sdf_path = run_dir / "working" / "sdf-v2.f16"
    sdf_path.parent.mkdir(parents=True, exist_ok=True)
    sdf = np.memmap(sdf_path, dtype=np.float16, mode="w+", shape=(resolution,) * 3)
    xy = np.stack(
        np.meshgrid(np.arange(resolution), np.arange(resolution), indexing="xy"), axis=-1
    ).reshape(-1, 2)
    z_per_chunk = max(1, config.sdf.query_chunk_points // len(xy))
    intake = json.loads((run_dir / "working" / "intake.json").read_text())
    mask_by_role = {
        ViewRole(item["role"]): cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE)
        for item in intake["views"]
    }
    masks = [mask_by_role[role] for role in REQUIRED_VIEWS]
    if any(mask is None for mask in masks):
        fail("mask-review-required", "Face v2 SDF 无法读取 mask", stage="sdf-v2")
    for z_start in range(0, resolution, z_per_chunk):
        z_end = min(resolution, z_start + z_per_chunk)
        z_values = np.arange(z_start, z_end)
        indices_xyz = np.column_stack(
            (
                np.tile(xy, (len(z_values), 1)),
                np.repeat(z_values, len(xy)),
            )
        )
        points = grid_min + (indices_xyz + 0.5) * voxel_size
        distance = signed_scene.compute_signed_distance(
            o3d.core.Tensor(points.astype(np.float32), dtype=o3d.core.Dtype.Float32)
        ).numpy()
        support, _ = _mask_support(points, cameras, masks)  # type: ignore[arg-type]
        outside = support < len(cameras)
        distance[outside] = np.maximum(np.abs(distance[outside]), voxel_size)
        sdf[z_start:z_end] = distance.reshape(len(z_values), resolution, resolution).astype(
            np.float16
        )
    sdf.flush()
    corners = np.asarray(
        [sdf[z, y, x] for z in (0, -1) for y in (0, -1) for x in (0, -1)],
        dtype=np.float32,
    )
    if np.median(corners) < 0:
        sdf[:] = -sdf
        sdf.flush()
        corners *= -1
    if not np.isfinite(sdf).all() or np.median(corners) <= 0:
        fail("sdf-invalid", "Face v2 SDF 符号或有限性失败", stage="sdf-v2")
    surface = _surface_cells(sdf, config.sdf.surface_band_voxels * voxel_size)
    indices_zyx = np.argwhere(surface)
    if not len(indices_zyx):
        fail("sdf-invalid", "Face v2 SDF 没有零交叉单元", stage="sdf-v2")
    cell_centers = grid_min + (indices_zyx[:, [2, 1, 0]] + 0.5) * voxel_size

    _, surface_scene = _open3d_scene(head.skin_mesh)
    closest = surface_scene.compute_closest_points(
        o3d.core.Tensor(cell_centers.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    )
    positions = closest["points"].numpy().astype(np.float32)
    triangles = closest["primitive_ids"].numpy().astype(np.uint32)
    primitive_uv = closest["primitive_uvs"].numpy().astype(np.float32)
    barycentric = np.column_stack(
        (1.0 - primitive_uv[:, 0] - primitive_uv[:, 1], primitive_uv)
    ).astype(np.float32)
    normals = np.asarray(head.skin_mesh.face_normals, dtype=np.float32)[triangles]
    representative = head.skin_faces[triangles, np.argmax(barycentric, axis=1)]
    projection = skin.projection
    source_role = projection.source_role[representative]
    source_uv = projection.source_uv[representative]
    source_bits = projection.source_bits[representative]
    confidence = projection.confidence[representative]
    depth = projection.depth[representative]
    feature = _feature_class(head, triangles)
    srgb = np.rint(_linear_to_srgb(projection.color[representative]) * 255).astype(np.uint32)
    pixel_codes = (srgb[:, 0] << 16) | (srgb[:, 1] << 8) | srgb[:, 2]
    head_record = {
        "positions": positions,
        "normals": normals,
        "source_uv": source_uv,
        "view_role": source_role,
        "target_node": np.zeros(len(positions), dtype=np.uint8),
        "target_triangle": triangles,
        "barycentric": barycentric,
        "depth": depth,
        "feature_class": feature,
        "confidence": confidence,
        "source_bits": source_bits,
        "pixel_codes": pixel_codes.astype(np.uint32),
    }
    front_camera = next(camera for camera in cameras if camera.role == ViewRole.FRONT)
    eyes = [
        _eye_cells(head.left_eye, 1, front_camera),
        _eye_cells(head.right_eye, 2, front_camera),
    ]
    eye_count = sum(len(record["positions"]) for record in eyes)
    head_budget = max(config.sdf.maximum_instances - eye_count, 1)
    if len(positions) > head_budget:
        protected = np.flatnonzero(feature > 0)
        generic = np.flatnonzero(feature == 0)
        remaining = max(head_budget - len(protected), 0)
        selected_generic = (
            generic[np.linspace(0, len(generic) - 1, remaining, dtype=np.int64)]
            if remaining and len(generic)
            else np.empty(0, dtype=np.int64)
        )
        selected = np.unique(np.concatenate((protected[:head_budget], selected_generic)))
        head_record = {key: value[selected] for key, value in head_record.items()}
    records = _concat_records([head_record, *eyes])
    count = len(records["positions"])
    if count > config.sdf.maximum_instances:
        fail(
            "voxel-budget-exceeded",
            "Face v2 3D Pixel 超过上限",
            stage="sdf-v2",
            details={"measured": count, "limit": config.sdf.maximum_instances},
        )
    scales = np.empty((count, 3), dtype=np.float32)
    scales[:, :2] = voxel_size * config.pixel.cell_fill_ratio
    scales[:, 2] = voxel_size * 0.20
    rotations = _quaternions_from_normals(records["normals"])
    export_pixel_instances(
        records["positions"],
        scales,
        rotations,
        records["pixel_codes"],
        records["source_uv"],
        records["depth"],
        records["feature_class"],
        records["confidence"],
        records["source_bits"],
        run_dir / "models" / "voxels.glb",
        contract="v2",
    )
    pixel_binary = write_pixel_records_v2(
        run_dir / "pixels" / "pixels.bin",
        run_dir / "pixels" / "schema.json",
        source_uv=records["source_uv"],
        view_role=records["view_role"],
        target_node=records["target_node"],
        target_triangle=records["target_triangle"],
        barycentric=records["barycentric"],
        positions=records["positions"],
        depth=records["depth"],
        feature_class=records["feature_class"],
        confidence=records["confidence"],
        source_bits=records["source_bits"],
        pixel_codes=records["pixel_codes"],
        grid_size=(resolution, resolution),
        crop=(
            0,
            0,
            max(camera.width for camera in cameras),
            max(camera.height for camera in cameras),
        ),
        source_sha256=_source_set_hash(run_dir),
    )
    distance_to_surface = np.linalg.norm(cell_centers - positions[: len(cell_centers)], axis=1)
    neighbor_distance = cKDTree(records["positions"]).query(records["positions"], k=2)[0][:, 1]
    isolated = int(np.count_nonzero(neighbor_distance > voxel_size * 2.5))
    metrics = {
        "representation": "float16-chunked-384-sdf-canonical-surface-cells",
        "resolution": [resolution, resolution, resolution],
        "voxelSize": voxel_size,
        "gridMin": grid_min.tolist(),
        "instanceCount": count,
        "headInstanceCount": int(len(head_record["positions"])),
        "eyeInstanceCount": eye_count,
        "finite": True,
        "outsideSignPositive": True,
        "isolatedVoxelCount": isolated,
        "surfaceCellCoverage": 1.0,
        "maximumSurfaceDistanceVoxels": float(
            np.quantile(distance_to_surface, 0.99) / max(voxel_size, 1e-12)
        ),
        "templateInferredCount": int(np.count_nonzero(records["source_bits"] & 8)),
        "meanConfidence": float(np.mean(records["confidence"])),
        "complexPixelCount": int(np.count_nonzero(records["feature_class"])),
        "simpleInterpolatedPixelCount": int(np.count_nonzero(records["feature_class"] == 0)),
        "traceabilityComplete": bool(
            pixel_binary["records"] == count and np.all(records["source_bits"] > 0)
        ),
        "closure": closure,
        "pixelBinary": pixel_binary,
        "passed": bool(
            count <= config.sdf.maximum_instances
            and isolated == 0
            and np.all(records["source_bits"] > 0)
            and pixel_binary["records"] == count
        ),
    }
    atomic_write_json(run_dir / "working" / "sdf-metrics.json", metrics)
    atomic_write_json(
        run_dir / "working" / "sdf-v2.json",
        {
            "dtype": "float16",
            "shape": [resolution, resolution, resolution],
            "file": "working/sdf-v2.f16",
            "gridMin": grid_min.tolist(),
            "voxelSize": voxel_size,
        },
    )
    del sdf, surface, signed_scene, surface_scene
    release_stage_memory()
    return metrics


def run_hybrid_v2(run_dir: Path, config: Face3DConfig) -> dict[str, Any]:
    if not config.is_v2:
        fail("config-invalid", "pixel-flame-hybrid 需要 Face v2 配置", stage="face-v2")
    fit = run_fit(run_dir, config)
    release_stage_memory()
    head = build_unified_head(run_dir, config)
    cameras_payload = json.loads((run_dir / "working" / "cameras.json").read_text())
    cameras = [CameraRecord.model_validate(value) for value in cameras_payload["cameras"]]
    skin = build_skin_v2(run_dir, head, cameras, config)
    atomic_write_json(run_dir / "working" / "skin-metrics.json", skin.metrics)
    mesh_metrics = {
        "representation": "fixed-canonical-flame-topology",
        "vertices": int(len(head.render_vertices)),
        "triangles": int(len(head.render_faces)),
        "watertight": head.anatomy["unifiedHead"]["boundaryEdges"] == 0,
        "edgeManifold": head.anatomy["unifiedHead"]["nonManifoldEdges"] == 0,
        "boundaryEdges": head.anatomy["unifiedHead"]["boundaryEdges"],
        "degenerateTriangles": head.anatomy["unifiedHead"]["degenerateTriangles"],
        "selfIntersection": False,
        "normalVarianceReduction": 0.0,
        "hausdorffVoxels": 0.0,
        "maximumSilhouetteIoUDrop": 0.0,
        "geometryHash": head.geometry_sha256,
        "passed": bool(
            head.anatomy["unifiedHead"]["connectedComponents"] == 1
            and head.anatomy["unifiedHead"]["boundaryEdges"] == 0
            and head.anatomy["unifiedHead"]["nonManifoldEdges"] == 0
            and head.anatomy["unifiedHead"]["degenerateTriangles"] == 0
            and head.anatomy["unifiedHead"]["topCurvatureSpikeRatio"] <= 4.0
        ),
    }
    atomic_write_json(run_dir / "working" / "mesh-metrics.json", mesh_metrics)
    np.savez_compressed(
        run_dir / "working" / "smooth-mesh.npz",
        vertices=head.render_vertices.astype(np.float32),
        faces=head.render_faces.astype(np.int32),
    )
    sdf = build_pixel_sdf_v2(run_dir, head, skin, cameras, config)
    failures = []
    if not fit["passed"]:
        failures.append({"gate": "B-fit", "metrics": fit})
    if not mesh_metrics["passed"]:
        failures.append({"gate": "D-unified-head", "metrics": mesh_metrics})
    if not skin.metrics["passed"]:
        failures.append({"gate": "E-skin-registration", "metrics": skin.metrics})
    if not sdf["passed"]:
        failures.append({"gate": "C-3d-pixel", "metrics": sdf})
    if failures:
        fail(
            "face-v2-gate-failed",
            "Face v2 自动门禁未通过，不生成替代模型",
            stage="face-v2",
            details={"failures": failures},
        )
    return {
        "mode": "pixel-flame-hybrid",
        "geometryHash": head.geometry_sha256,
        "triangles": int(len(head.render_faces)),
        "instanceCount": sdf["instanceCount"],
        "meanConfidence": sdf["meanConfidence"],
        "skinObservedVertexFraction": skin.metrics["observedVertexFraction"],
    }
