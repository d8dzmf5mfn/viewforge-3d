from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy import ndimage

from face3d.config import Face3DConfig
from face3d.errors import fail
from face3d.glb import export_instanced_voxels, export_neutral_mesh
from face3d.io import atomic_write_json
from face3d.models import REQUIRED_VIEWS, CameraRecord, ViewRole


def _boundary_directed_edges(faces: np.ndarray) -> np.ndarray:
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    return directed[counts[inverse] == 1]


def _boundary_loops(edges: np.ndarray) -> list[list[int]]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    directed_lookup: dict[tuple[int, int], tuple[int, int]] = {}
    for first, second in edges.tolist():
        adjacency[first].append(second)
        adjacency[second].append(first)
        directed_lookup[tuple(sorted((first, second)))] = (first, second)
    unused = {tuple(sorted(edge)) for edge in edges.tolist()}
    loops: list[list[int]] = []
    while unused:
        edge = next(iter(unused))
        start, current = edge
        loop = [start, current]
        unused.discard(edge)
        previous = start
        while current != start:
            candidates = [value for value in adjacency[current] if value != previous]
            next_vertex = next(
                (value for value in candidates if tuple(sorted((current, value))) in unused),
                start if start in candidates else None,
            )
            if next_vertex is None:
                break
            unused.discard(tuple(sorted((current, next_vertex))))
            previous, current = current, next_vertex
            if current != start:
                loop.append(current)
            if len(loop) > len(edges) + 1:
                break
        if current == start and len(loop) >= 3:
            first_directed = directed_lookup[tuple(sorted((loop[0], loop[1])))]
            if first_directed != (loop[0], loop[1]):
                loop.reverse()
            loops.append(loop)
    return loops


def close_working_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh, multibody=True)
    boundary_edges = _boundary_directed_edges(np.asarray(mesh.faces))
    loops = _boundary_loops(boundary_edges)
    vertices = np.asarray(mesh.vertices).tolist()
    faces = np.asarray(mesh.faces).tolist()
    capped = 0
    for loop in loops:
        centroid_index = len(vertices)
        vertices.append(np.mean(np.asarray(mesh.vertices)[loop], axis=0).tolist())
        for index, first in enumerate(loop):
            second = loop[(index + 1) % len(loop)]
            faces.append([second, first, centroid_index])
        capped += 1
    closed = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=True,
        validate=True,
    )
    trimesh.repair.fix_normals(closed, multibody=True)
    if not closed.is_watertight:
        fail(
            "sdf-invalid",
            "FLAME 工作网格封口后仍非闭合",
            stage="sdf",
            details={"boundaryLoops": len(loops)},
        )
    return closed, {"cappedBoundaryLoops": capped, "inputBoundaryEdges": int(len(boundary_edges))}


def _camera_projection(points: np.ndarray, camera: CameraRecord) -> tuple[np.ndarray, np.ndarray]:
    rotation, _ = cv2.Rodrigues(np.asarray(camera.rotation_vector, dtype=np.float64))
    camera_points = points @ rotation.T + np.asarray(camera.translation)
    valid = camera_points[:, 2] > 1e-6
    pixels = np.zeros((len(points), 2), dtype=np.float64)
    pixels[valid] = camera_points[valid, :2] / camera_points[
        valid, 2:3
    ] * camera.focal_length_px + np.asarray(camera.principal_point_px)
    return pixels, valid


def _mask_support(
    points: np.ndarray,
    cameras: list[CameraRecord],
    masks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    source_bits = np.zeros(len(points), dtype=np.uint8)
    support_count = np.zeros(len(points), dtype=np.uint8)
    for index, (camera, mask) in enumerate(zip(cameras, masks, strict=True)):
        pixels, valid = _camera_projection(points, camera)
        rounded = np.rint(pixels).astype(np.int64)
        inside = (
            valid
            & (rounded[:, 0] >= 0)
            & (rounded[:, 0] < camera.width)
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < camera.height)
        )
        supported = np.zeros(len(points), dtype=bool)
        indices = np.flatnonzero(inside)
        supported[indices] = mask[rounded[indices, 1], rounded[indices, 0]] > 127
        support_count += supported.astype(np.uint8)
        source_bits |= supported.astype(np.uint8) << index
    return support_count, source_bits


def _surface_cells(sdf: np.ndarray, band: float) -> np.ndarray:
    selected = np.zeros(sdf.shape, dtype=bool)
    for axis in range(3):
        first_slice = [slice(None)] * 3
        second_slice = [slice(None)] * 3
        first_slice[axis] = slice(0, -1)
        second_slice[axis] = slice(1, None)
        first = sdf[tuple(first_slice)]
        second = sdf[tuple(second_slice)]
        crossings = np.signbit(first) != np.signbit(second)
        choose_first = crossings & (np.abs(first) <= np.abs(second))
        choose_second = crossings & ~choose_first
        selected[tuple(first_slice)] |= choose_first
        selected[tuple(second_slice)] |= choose_second
    selected &= np.abs(sdf) <= band
    labels, count = ndimage.label(selected, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count > 1:
        sizes = np.bincount(labels.reshape(-1))
        sizes[0] = 0
        selected = labels == int(np.argmax(sizes))
    return selected


def run_sdf(run_dir: Path, config: Face3DConfig) -> dict[str, Any]:
    fit = np.load(run_dir / "working" / "fit.npz")
    fitted_mesh = trimesh.Trimesh(vertices=fit["vertices"], faces=fit["faces"], process=False)
    working_mesh, closure = close_working_mesh(fitted_mesh)
    export_neutral_mesh(working_mesh, run_dir / "models" / "working-closed.glb")
    cameras_payload = json.loads((run_dir / "working" / "cameras.json").read_text())
    cameras = [CameraRecord.model_validate(item) for item in cameras_payload["cameras"]]
    intake = json.loads((run_dir / "working" / "intake.json").read_text())
    masks_by_role = {
        ViewRole(item["role"]): cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE)
        for item in intake["views"]
    }
    masks = [masks_by_role[role] for role in REQUIRED_VIEWS]
    if any(mask is None for mask in masks):
        fail("mask-review-required", "SDF 阶段无法读取已确认 mask", stage="sdf")

    resolution = config.sdf.resolution
    bounds = np.asarray(working_mesh.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    extent = float(np.ptp(bounds, axis=0).max() * (1 + 2 * config.sdf.padding_fraction))
    grid_min = center - extent / 2
    voxel_size = extent / resolution
    scene = o3d.t.geometry.RaycastingScene()
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(working_mesh.vertices)),
        o3d.utility.Vector3iVector(np.asarray(working_mesh.faces, dtype=np.int32)),
    )
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    sdf = np.empty((resolution, resolution, resolution), dtype=np.float32)
    xy = np.stack(
        np.meshgrid(np.arange(resolution), np.arange(resolution), indexing="xy"), axis=-1
    ).reshape(-1, 2)
    z_per_chunk = max(1, config.sdf.query_chunk_points // len(xy))
    for z_start in range(0, resolution, z_per_chunk):
        z_end = min(resolution, z_start + z_per_chunk)
        z_values = np.arange(z_start, z_end)
        tiled_xy = np.tile(xy, (len(z_values), 1))
        repeated_z = np.repeat(z_values, len(xy))
        indices_xyz = np.column_stack((tiled_xy[:, 0], tiled_xy[:, 1], repeated_z))
        points = grid_min + (indices_xyz + 0.5) * voxel_size
        distances = scene.compute_signed_distance(
            o3d.core.Tensor(points.astype(np.float32), dtype=o3d.core.Dtype.Float32)
        ).numpy()
        support, _ = _mask_support(points, cameras, masks)
        outside_hull = support < len(cameras)
        distances[outside_hull] = np.maximum(np.abs(distances[outside_hull]), voxel_size)
        chunk = distances.reshape(len(z_values), resolution, resolution)
        sdf[z_start:z_end] = chunk
    corner_values = np.asarray([sdf[z, y, x] for z in (0, -1) for y in (0, -1) for x in (0, -1)])
    if np.median(corner_values) < 0:
        sdf *= -1
        corner_values *= -1
    if not np.isfinite(sdf).all() or np.median(corner_values) == 0:
        fail("sdf-invalid", "SDF 包含非有限值或符号不确定", stage="sdf")
    surface = _surface_cells(sdf, config.sdf.surface_band_voxels * voxel_size)
    indices_zyx = np.argwhere(surface)
    if not len(indices_zyx):
        fail("sdf-invalid", "SDF 没有零交叉表面单元", stage="sdf")
    if len(indices_zyx) > config.sdf.maximum_instances:
        fail(
            "voxel-budget-exceeded",
            "表面 3D Pixel 数量超过配置上限",
            stage="sdf",
            details={"instances": int(len(indices_zyx)), "limit": config.sdf.maximum_instances},
        )
    indices_xyz = indices_zyx[:, [2, 1, 0]]
    translations = grid_min + (indices_xyz + 0.5) * voxel_size
    support, source_bits = _mask_support(translations, cameras, masks)
    confidence = support.astype(np.float32) / len(cameras)
    normalized = (translations - bounds[0]) / np.maximum(bounds[1] - bounds[0], 1e-9)
    template_inferred = (normalized[:, 2] < 0.30) | (normalized[:, 1] < 0.12)
    confidence[template_inferred] = np.minimum(confidence[template_inferred], 0.25)
    source_bits[template_inferred] |= 8
    export_instanced_voxels(
        translations,
        voxel_size,
        confidence,
        source_bits,
        run_dir / "models" / "voxels.glb",
    )
    sdf_path = run_dir / "working" / "sdf.npz"
    np.savez_compressed(
        sdf_path,
        sdf=sdf,
        grid_min=grid_min.astype(np.float64),
        voxel_size=np.asarray(voxel_size, dtype=np.float64),
        resolution=np.asarray(resolution, dtype=np.int32),
    )
    confidence_path = run_dir / "working" / "voxel-confidence.npz"
    np.savez_compressed(
        confidence_path,
        indices_zyx=indices_zyx.astype(np.uint16),
        confidence=confidence.astype(np.float16),
        source_bits=source_bits,
    )
    metrics = {
        "resolution": resolution,
        "voxelSize": voxel_size,
        "gridMin": grid_min.tolist(),
        "instanceCount": int(len(translations)),
        "finite": bool(np.isfinite(sdf).all()),
        "outsideSignPositive": bool(np.median(corner_values) > 0),
        "mainComponents": 1,
        "isolatedVoxelCount": 0,
        "templateInferredCount": int(np.count_nonzero(template_inferred)),
        "meanConfidence": float(np.mean(confidence)),
        "closure": closure,
        "passed": True,
    }
    atomic_write_json(run_dir / "working" / "sdf-metrics.json", metrics)
    return metrics
