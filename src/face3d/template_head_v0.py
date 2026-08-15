from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image, UnidentifiedImageError
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from face3d.errors import Face3DError, fail
from face3d.glb import export_neutral_mesh
from face3d.io import atomic_write_bytes, atomic_write_json, sha256_file
from face3d.models import REQUIRED_VIEWS, CameraRecord, ViewRole
from face3d.render import render_flat_mesh

TEMPLATE_ID = "TemplateHeadV0"
RAW_SCHEMA_VERSION = "0.1.0"
PACKAGE_ENTRIES = {
    "compute": "models/smooth.glb",
    "render": "models/skin.glb",
    "manifest": "manifest.json",
}
REQUIRED_PACKAGE_ENTRIES = {
    PACKAGE_ENTRIES["compute"],
    PACKAGE_ENTRIES["manifest"],
}


def _geometry_hash(vertices: np.ndarray, faces: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(vertices, dtype="<f4").tobytes())
    digest.update(np.ascontiguousarray(faces, dtype="<u4").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RawTemplateHeadV0:
    compute_vertices: np.ndarray
    compute_faces: np.ndarray
    render_to_compute: np.ndarray
    render_faces: np.ndarray
    uv: np.ndarray
    geometry_sha256: str
    metadata: dict[str, Any]

    @property
    def compute_mesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(
            vertices=self.compute_vertices,
            faces=self.compute_faces,
            process=False,
            validate=False,
        )

    @property
    def render_vertices(self) -> np.ndarray:
        return self.compute_vertices[self.render_to_compute]

    def save(self, destination: Path) -> None:
        output = io.BytesIO()
        np.savez_compressed(
            output,
            schema_version=np.asarray(RAW_SCHEMA_VERSION),
            template_id=np.asarray(TEMPLATE_ID),
            compute_vertices=np.asarray(self.compute_vertices, dtype=np.float32),
            compute_faces=np.asarray(self.compute_faces, dtype=np.int32),
            render_to_compute=np.asarray(self.render_to_compute, dtype=np.int32),
            render_faces=np.asarray(self.render_faces, dtype=np.int32),
            uv=np.asarray(self.uv, dtype=np.float32),
            geometry_sha256=np.asarray(self.geometry_sha256),
            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )
        atomic_write_bytes(destination, output.getvalue())

    @classmethod
    def load(cls, source: Path) -> RawTemplateHeadV0:
        with np.load(source, allow_pickle=False) as payload:
            schema_version = str(payload["schema_version"])
            template_id = str(payload["template_id"])
            if schema_version != RAW_SCHEMA_VERSION or template_id != TEMPLATE_ID:
                fail(
                    "template-schema-mismatch",
                    "TemplateHeadV0 原始资产 schema 不兼容",
                    stage="template-head-v0",
                    details={
                        "expectedSchemaVersion": RAW_SCHEMA_VERSION,
                        "actualSchemaVersion": schema_version,
                        "expectedTemplateId": TEMPLATE_ID,
                        "actualTemplateId": template_id,
                    },
                )
            asset = cls(
                compute_vertices=np.asarray(payload["compute_vertices"], dtype=np.float64),
                compute_faces=np.asarray(payload["compute_faces"], dtype=np.int64),
                render_to_compute=np.asarray(payload["render_to_compute"], dtype=np.int64),
                render_faces=np.asarray(payload["render_faces"], dtype=np.int64),
                uv=np.asarray(payload["uv"], dtype=np.float32),
                geometry_sha256=str(payload["geometry_sha256"]),
                metadata=json.loads(str(payload["metadata_json"])),
            )
        if _geometry_hash(asset.compute_vertices, asset.compute_faces) != asset.geometry_sha256:
            fail(
                "template-geometry-hash-mismatch",
                "TemplateHeadV0 几何哈希校验失败",
                stage="template-head-v0",
            )
        return asset


def _read_package(package: Path) -> tuple[dict[str, Any], bytes, bytes | None]:
    try:
        with zipfile.ZipFile(package) as archive:
            names = set(archive.namelist())
            missing = sorted(REQUIRED_PACKAGE_ENTRIES - names)
            if missing:
                fail(
                    "template-source-entry-missing",
                    "结果包缺少 TemplateHeadV0 提取所需文件",
                    stage="template-head-v0",
                    details={"missing": missing},
                )
            manifest_value = json.loads(archive.read(PACKAGE_ENTRIES["manifest"]))
            if not isinstance(manifest_value, dict):
                raise TypeError("manifest must be an object")
            return (
                manifest_value,
                archive.read(PACKAGE_ENTRIES["compute"]),
                (
                    archive.read(PACKAGE_ENTRIES["render"])
                    if PACKAGE_ENTRIES["render"] in names
                    else None
                ),
            )
    except Face3DError:
        raise
    except (OSError, TypeError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        fail(
            "template-source-package-invalid",
            "无法读取 TemplateHeadV0 源结果包",
            stage="template-head-v0",
            details={"reason": str(exc)},
        )


def _load_single_mesh(payload: bytes, *, label: str) -> trimesh.Trimesh:
    try:
        scene = trimesh.load(
            io.BytesIO(payload),
            file_type="glb",
            force="scene",
            process=False,
        )
        if not isinstance(scene, trimesh.Scene) or len(scene.geometry) != 1:
            count = len(scene.geometry) if isinstance(scene, trimesh.Scene) else 0
            fail(
                "template-source-mesh-count-invalid",
                f"{label} 必须只包含一个网格",
                stage="template-head-v0",
                details={"meshCount": count, "source": label},
            )
        mesh = trimesh.load(
            io.BytesIO(payload),
            file_type="glb",
            force="mesh",
            process=False,
        )
    except Face3DError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        fail(
            "template-source-mesh-invalid",
            f"无法解析 {label}",
            stage="template-head-v0",
            details={"reason": str(exc), "source": label},
        )
    if not isinstance(mesh, trimesh.Trimesh):
        fail(
            "template-source-mesh-invalid",
            f"{label} 不是三角网格",
            stage="template-head-v0",
            details={"source": label},
        )
    return mesh


def _edge_and_component_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> dict[str, int | bool]:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        fail(
            "template-vertex-shape-invalid",
            "模板顶点必须为 Nx3",
            stage="template-head-v0",
            details={"shape": list(vertices.shape)},
        )
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        fail(
            "template-face-shape-invalid",
            "模板面必须为非空 Mx3 三角形",
            stage="template-head-v0",
            details={"shape": list(faces.shape)},
        )
    if not np.isfinite(vertices).all():
        fail(
            "template-non-finite-geometry",
            "模板几何包含 NaN 或 Inf",
            stage="template-head-v0",
        )
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        fail(
            "template-face-index-invalid",
            "模板面索引越界",
            stage="template-head-v0",
        )

    edges = np.sort(
        np.concatenate(
            (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]),
            axis=0,
        ),
        axis=1,
    )
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    rows = np.concatenate((unique_edges[:, 0], unique_edges[:, 1]))
    columns = np.concatenate((unique_edges[:, 1], unique_edges[:, 0]))
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(len(vertices), len(vertices)),
    )
    component_count = int(connected_components(graph, directed=False, return_labels=False))
    repeated_index_faces = np.any(
        np.column_stack(
            (
                faces[:, 0] == faces[:, 1],
                faces[:, 1] == faces[:, 2],
                faces[:, 2] == faces[:, 0],
            )
        ),
        axis=1,
    )
    triangles = vertices[faces]
    doubled_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    diagonal = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1e-12)
    # Blender's exact boolean can create very slender but non-zero triangles on
    # coplanar intersections. Treat only numerically zero area as degenerate;
    # aspect ratio is reported separately and must not create a false hole.
    zero_area_tolerance = diagonal * diagonal * 1e-16
    zero_area_faces = doubled_area <= zero_area_tolerance
    sorted_faces = np.sort(faces, axis=1)
    duplicate_face_count = len(sorted_faces) - len(np.unique(sorted_faces, axis=0))
    duplicate_vertex_count = len(vertices) - len(np.unique(vertices, axis=0))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    return {
        "componentCount": component_count,
        "boundaryEdgeCount": int(np.count_nonzero(edge_counts == 1)),
        "nonManifoldEdgeCount": int(np.count_nonzero(edge_counts > 2)),
        "degenerateFaceCount": int(np.count_nonzero(repeated_index_faces | zero_area_faces)),
        "duplicateFaceCount": int(duplicate_face_count),
        "duplicateVertexCount": int(duplicate_vertex_count),
        "minimumDoubledTriangleArea": float(np.min(doubled_area)),
        "zeroAreaTolerance": float(zero_area_tolerance),
        "watertight": bool(mesh.is_watertight),
        "windingConsistent": bool(mesh.is_winding_consistent),
    }


def _is_locked_topology(metrics: dict[str, int | bool]) -> bool:
    expected: dict[str, int | bool] = {
        "componentCount": 1,
        "boundaryEdgeCount": 0,
        "nonManifoldEdgeCount": 0,
        "degenerateFaceCount": 0,
        "duplicateFaceCount": 0,
        "duplicateVertexCount": 0,
        "watertight": True,
        "windingConsistent": True,
    }
    return all(metrics[name] == value for name, value in expected.items())


def _unify_compute_mesh(
    source_mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    source_vertices = np.asarray(source_mesh.vertices, dtype=np.float64)
    source_faces = np.asarray(source_mesh.faces, dtype=np.int64)
    source_metrics = _edge_and_component_metrics(source_vertices, source_faces)
    if source_metrics["componentCount"] == 1:
        return source_mesh, {
            "required": False,
            "method": "source-already-continuous",
            "sdfUsed": False,
            "input": source_metrics,
            "output": source_metrics,
        }

    components = sorted(
        source_mesh.split(only_watertight=False),
        key=lambda mesh: (
            -len(mesh.faces),
            tuple(np.asarray(mesh.bounds, dtype=np.float64).reshape(-1)),
        ),
    )
    component_records = [
        {
            "vertexCount": len(component.vertices),
            "faceCount": len(component.faces),
            "bounds": np.asarray(component.bounds, dtype=float).tolist(),
            "watertight": bool(component.is_watertight),
            "windingConsistent": bool(component.is_winding_consistent),
        }
        for component in components
    ]
    if any(
        not record["watertight"] or not record["windingConsistent"] for record in component_records
    ):
        fail(
            "template-source-components-invalid",
            "源 smooth.glb 包含无法精确焊接的开放或反向组件",
            stage="template-head-v0",
            details={"components": component_records},
        )
    try:
        unified = trimesh.boolean.union(
            components,
            engine="blender",
            check_volume=True,
        )
    except Exception as exc:
        fail(
            "template-component-union-failed",
            "头颈、双耳和肩部的精确拓扑焊接失败",
            stage="template-head-v0",
            details={
                "engine": "blender-exact-boolean",
                "reason": str(exc),
            },
        )
    if not isinstance(unified, trimesh.Trimesh):
        fail(
            "template-component-union-invalid",
            "精确拓扑焊接没有生成单一三角网格",
            stage="template-head-v0",
        )
    unified_vertices = np.asarray(unified.vertices, dtype=np.float64)
    unified_faces = np.asarray(unified.faces, dtype=np.int64)
    output_metrics = _edge_and_component_metrics(unified_vertices, unified_faces)
    if not _is_locked_topology(output_metrics):
        fail(
            "template-component-union-invalid",
            "精确拓扑焊接结果未达到单连通闭合流形要求",
            stage="template-head-v0",
            details=output_metrics,
        )
    return unified, {
        "required": True,
        "method": "blender-exact-boolean-union",
        "sdfUsed": False,
        "input": source_metrics,
        "components": component_records,
        "output": output_metrics,
    }


def _weld_uv_seams(
    render_mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    render_vertices = np.asarray(render_mesh.vertices, dtype=np.float64)
    render_faces = np.asarray(render_mesh.faces, dtype=np.int64)
    compute_mesh = trimesh.Trimesh(
        vertices=render_vertices,
        faces=render_faces,
        process=True,
        validate=True,
    )
    compute_vertices = np.asarray(compute_mesh.vertices, dtype=np.float64)
    compute_faces = np.asarray(compute_mesh.faces, dtype=np.int64)
    if len(compute_faces) != len(render_faces):
        fail(
            "template-source-weld-changed-surface",
            "源 GLB 的 UV 缝焊接删除了表面三角形",
            stage="template-head-v0",
            details={
                "renderFaceCount": len(render_faces),
                "computeFaceCount": len(compute_faces),
            },
        )
    distances, mapping = cKDTree(compute_vertices).query(render_vertices, k=1)
    mapped_faces = mapping[render_faces]
    if float(np.max(distances)) != 0.0 or not np.array_equal(mapped_faces, compute_faces):
        fail(
            "template-source-weld-not-zero-drift",
            "源 GLB 无法只通过零漂移 UV 缝焊接转换为计算拓扑",
            stage="template-head-v0",
            details={
                "maximumPositionDifference": float(np.max(distances)),
                "mismatchedFaceCount": int(
                    np.count_nonzero(np.any(mapped_faces != compute_faces, axis=1))
                ),
            },
        )
    metrics = _edge_and_component_metrics(compute_vertices, compute_faces)
    if not _is_locked_topology(metrics):
        fail(
            "template-source-weld-topology-invalid",
            "UV 缝焊接后仍不是单连通闭合流形头模",
            stage="template-head-v0",
            details=metrics,
        )
    return compute_mesh, {
        "required": True,
        "method": "exact-position-uv-seam-weld",
        "sdfUsed": False,
        "input": {
            "renderVertexCount": len(render_vertices),
            "renderFaceCount": len(render_faces),
        },
        "weldedDuplicateVertexCount": len(render_vertices) - len(compute_vertices),
        "maximumPositionDifference": float(np.max(distances)),
        "output": metrics,
    }


def _extract_raw_asset(
    compute_mesh: trimesh.Trimesh,
    render_mesh: trimesh.Trimesh | None,
) -> tuple[RawTemplateHeadV0, dict[str, Any], dict[str, Any], str]:
    compute_vertices = np.asarray(compute_mesh.vertices, dtype=np.float64)
    compute_faces = np.asarray(compute_mesh.faces, dtype=np.int64)
    uv_method = "source-skin-glb"
    uv_repair: dict[str, Any] = {
        "repairedFaceCount": 0,
        "maximumRepairFraction": 0.02,
    }
    if render_mesh is None:
        try:
            import xatlas
        except ImportError:
            fail(
                "template-uv-dependency-missing",
                "源 smooth.glb 没有 UV；生成稳定 UV 需要 xatlas",
                stage="template-head-v0",
                details={"install": "uv sync"},
            )
        mapping, parameterized_faces, parameterized_uv = xatlas.parametrize(
            compute_vertices.astype(np.float32),
            compute_faces.astype(np.uint32),
        )
        mapping, parameterized_faces, parameterized_uv, uv_repair = (
            _repair_degenerate_xatlas_charts(
                compute_faces,
                np.asarray(mapping, dtype=np.int64),
                np.asarray(parameterized_faces, dtype=np.int64),
                np.asarray(parameterized_uv, dtype=np.float64),
            )
        )
        render_mesh = trimesh.Trimesh(
            vertices=compute_vertices[np.asarray(mapping, dtype=np.int64)],
            faces=np.asarray(parameterized_faces, dtype=np.int64),
            process=False,
            validate=False,
        )
        render_mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.asarray(parameterized_uv, dtype=np.float32),
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=(0.5, 0.52, 0.56, 1.0),
                metallicFactor=0.0,
                roughnessFactor=0.76,
            ),
        )
        uv_method = "xatlas-0.0.11"
        if uv_repair["repairedFaceCount"]:
            uv_method += "+deterministic-degenerate-chart-repair"
    render_vertices = np.asarray(render_mesh.vertices, dtype=np.float64)
    render_faces = np.asarray(render_mesh.faces, dtype=np.int64)
    topology = _edge_and_component_metrics(compute_vertices, compute_faces)
    if not _is_locked_topology(topology):
        fail(
            "template-compute-topology-invalid",
            "smooth.glb 不是可锁定的一体化闭合计算拓扑",
            stage="template-head-v0",
            details=topology,
        )
    if len(render_faces) != len(compute_faces):
        fail(
            "template-render-face-count-mismatch",
            "skin.glb 与 smooth.glb 面数不一致",
            stage="template-head-v0",
            details={
                "computeFaceCount": len(compute_faces),
                "renderFaceCount": len(render_faces),
            },
        )
    uv_value = getattr(render_mesh.visual, "uv", None)
    if uv_value is None:
        fail(
            "template-render-uv-missing",
            "skin.glb 缺少 UV",
            stage="template-head-v0",
        )
    uv = np.asarray(uv_value, dtype=np.float64)
    if uv.shape != (len(render_vertices), 2) or not np.isfinite(uv).all():
        fail(
            "template-render-uv-invalid",
            "skin.glb UV 形状错误或包含 NaN/Inf",
            stage="template-head-v0",
            details={"shape": list(uv.shape)},
        )

    diagonal = max(float(np.linalg.norm(np.ptp(compute_vertices, axis=0))), 1e-12)
    mapping_tolerance = max(diagonal * 1e-8, 1e-10)
    distances, render_to_compute = cKDTree(compute_vertices).query(render_vertices, k=1)
    maximum_distance = float(np.max(distances))
    if maximum_distance > mapping_tolerance:
        fail(
            "template-render-position-mismatch",
            "skin.glb 顶点无法零漂移映射到 smooth.glb",
            stage="template-head-v0",
            details={
                "maximumDistance": maximum_distance,
                "tolerance": mapping_tolerance,
            },
        )
    unique_compute = np.unique(render_to_compute)
    if len(unique_compute) != len(compute_vertices):
        fail(
            "template-render-coverage-incomplete",
            "skin.glb 未覆盖 smooth.glb 的全部计算顶点",
            stage="template-head-v0",
            details={
                "coveredComputeVertexCount": len(unique_compute),
                "computeVertexCount": len(compute_vertices),
            },
        )
    mapped_faces = render_to_compute[render_faces]
    mismatch_count = int(np.count_nonzero(np.any(mapped_faces != compute_faces, axis=1)))
    if mismatch_count:
        fail(
            "template-render-topology-mismatch",
            "skin.glb UV 拓扑与 smooth.glb 计算拓扑不一致",
            stage="template-head-v0",
            details={"mismatchedFaceCount": mismatch_count},
        )

    uv_triangles = uv[render_faces]
    uv_doubled_area = np.abs(
        (uv_triangles[:, 1, 0] - uv_triangles[:, 0, 0])
        * (uv_triangles[:, 2, 1] - uv_triangles[:, 0, 1])
        - (uv_triangles[:, 1, 1] - uv_triangles[:, 0, 1])
        * (uv_triangles[:, 2, 0] - uv_triangles[:, 0, 0])
    )
    outside_unit = np.any((uv < 0.0) | (uv > 1.0), axis=1)
    uv_metrics = {
        "renderVertexCount": len(render_vertices),
        "seamDuplicateVertexCount": len(render_vertices) - len(compute_vertices),
        "minimum": np.min(uv, axis=0).astype(float).tolist(),
        "maximum": np.max(uv, axis=0).astype(float).tolist(),
        "outsideUnitVertexCount": int(np.count_nonzero(outside_unit)),
        "degenerateFaceCount": int(np.count_nonzero(uv_doubled_area <= 1e-12)),
        "sourceUvStable": bool(not np.any(outside_unit) and not np.any(uv_doubled_area <= 1e-12)),
        "repair": uv_repair,
        "method": uv_method,
    }
    mapping_metrics = {
        "maximumPositionDifference": maximum_distance,
        "tolerance": mapping_tolerance,
        "coveredComputeVertexCount": len(unique_compute),
        "mappedFaceMismatchCount": mismatch_count,
    }
    geometry_sha256 = _geometry_hash(compute_vertices, compute_faces)
    metadata = {
        "schemaVersion": RAW_SCHEMA_VERSION,
        "templateId": TEMPLATE_ID,
        "state": "raw-extracted",
        "geometrySha256": geometry_sha256,
        "topology": topology,
        "uv": uv_metrics,
        "mapping": mapping_metrics,
    }
    return (
        RawTemplateHeadV0(
            compute_vertices=compute_vertices,
            compute_faces=compute_faces,
            render_to_compute=np.asarray(render_to_compute, dtype=np.int64),
            render_faces=render_faces,
            uv=np.asarray(uv, dtype=np.float32),
            geometry_sha256=geometry_sha256,
            metadata=metadata,
        ),
        topology,
        uv_metrics,
        uv_method,
    )


def _repair_degenerate_xatlas_charts(
    compute_faces: np.ndarray,
    render_to_compute: np.ndarray,
    render_faces: np.ndarray,
    uv: np.ndarray,
    *,
    doubled_area_epsilon: float = 1e-12,
    maximum_repair_fraction: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Give xatlas-rejected sliver faces deterministic, isolated UV charts.

    Exact booleans can create geometrically valid sub-pixel triangles around an
    eyelid ring. xatlas leaves those render-only charts at (0, 0). Rebuilding
    the surface to satisfy the UV unwrap would change the locked head geometry,
    so the rejected faces instead receive independent charts in a reserved
    atlas band. The compute topology and every render-to-compute position stay
    unchanged.
    """

    compute_faces = np.asarray(compute_faces, dtype=np.int64)
    mapping = np.asarray(render_to_compute, dtype=np.int64).copy()
    faces = np.asarray(render_faces, dtype=np.int64).copy()
    coordinates = np.asarray(uv, dtype=np.float64).copy()
    triangles = coordinates[faces]
    doubled_area = np.abs(
        (triangles[:, 1, 0] - triangles[:, 0, 0]) * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 1, 1] - triangles[:, 0, 1]) * (triangles[:, 2, 0] - triangles[:, 0, 0])
    )
    rejected = np.flatnonzero(doubled_area <= doubled_area_epsilon)
    if not len(rejected):
        return (
            mapping,
            faces,
            coordinates.astype(np.float32),
            {
                "repairedFaceCount": 0,
                "maximumRepairFraction": maximum_repair_fraction,
            },
        )
    repair_fraction = len(rejected) / max(len(faces), 1)
    if repair_fraction > maximum_repair_fraction:
        fail(
            "template-uv-repair-limit-exceeded",
            "xatlas 未参数化的三角面过多，拒绝隐藏系统性 UV 失败",
            stage="template-head-v0",
            details={
                "rejectedFaceCount": len(rejected),
                "repairFraction": repair_fraction,
                "maximumRepairFraction": maximum_repair_fraction,
            },
        )
    if not np.array_equal(mapping[faces], compute_faces):
        fail(
            "template-uv-repair-topology-mismatch",
            "UV 修复前渲染面无法映射回计算拓扑",
            stage="template-head-v0",
        )

    # Keep the original atlas isotropic while reserving a square-cell band for
    # the tiny rejected faces. At 4K, each repaired cell remains roughly 90 px.
    main_scale = 0.86
    coordinates *= main_scale
    coordinates[:, 0] += (1.0 - main_scale) * 0.5
    coordinates[:, 1] += 0.005
    band_min = 0.88
    band_max = 0.995
    band_height = band_max - band_min
    columns = max(1, int(np.ceil(np.sqrt(len(rejected) / band_height))))
    rows = int(np.ceil(len(rejected) / columns))
    cell_width = 1.0 / columns
    cell_height = band_height / rows
    padding = 0.12

    appended_mapping: list[int] = []
    appended_uv: list[tuple[float, float]] = []
    next_vertex = len(mapping)
    for chart_index, face_index in enumerate(rejected):
        row, column = divmod(chart_index, columns)
        x0 = column * cell_width
        y0 = band_min + row * cell_height
        x1 = x0 + cell_width
        y1 = y0 + cell_height
        face_uv = (
            (x0 + cell_width * padding, y0 + cell_height * padding),
            (x1 - cell_width * padding, y0 + cell_height * padding),
            (x0 + cell_width * padding, y1 - cell_height * padding),
        )
        compute_triangle = compute_faces[face_index]
        appended_mapping.extend(int(value) for value in compute_triangle)
        appended_uv.extend(face_uv)
        faces[face_index] = np.arange(next_vertex, next_vertex + 3, dtype=np.int64)
        next_vertex += 3

    mapping = np.concatenate((mapping, np.asarray(appended_mapping, dtype=np.int64)))
    coordinates = np.vstack((coordinates, np.asarray(appended_uv, dtype=np.float64)))
    referenced = np.unique(faces.reshape(-1))
    compact = np.full(len(mapping), -1, dtype=np.int64)
    compact[referenced] = np.arange(len(referenced), dtype=np.int64)
    mapping = mapping[referenced]
    coordinates = coordinates[referenced]
    faces = compact[faces]
    if not np.array_equal(mapping[faces], compute_faces):
        fail(
            "template-uv-repair-topology-mismatch",
            "UV 修复改变了渲染面到计算拓扑的映射",
            stage="template-head-v0",
        )
    repaired_triangles = coordinates[faces]
    repaired_area = np.abs(
        (repaired_triangles[:, 1, 0] - repaired_triangles[:, 0, 0])
        * (repaired_triangles[:, 2, 1] - repaired_triangles[:, 0, 1])
        - (repaired_triangles[:, 1, 1] - repaired_triangles[:, 0, 1])
        * (repaired_triangles[:, 2, 0] - repaired_triangles[:, 0, 0])
    )
    if np.any(repaired_area <= doubled_area_epsilon):
        fail(
            "template-uv-repair-failed",
            "修复后仍存在退化 UV 三角面",
            stage="template-head-v0",
        )
    return (
        mapping,
        faces,
        coordinates.astype(np.float32),
        {
            "repairedFaceCount": len(rejected),
            "repairFraction": repair_fraction,
            "maximumRepairFraction": maximum_repair_fraction,
            "reservedBand": [band_min, band_max],
            "chartColumns": columns,
            "chartRows": rows,
        },
    )


def _normalized_baseline(source: bytes) -> tuple[bytes, dict[str, Any]]:
    try:
        with Image.open(io.BytesIO(source)) as opened:
            opened.load()
            image = opened.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        fail(
            "template-quality-baseline-invalid",
            "质量基准不是可读取的图片",
            stage="template-head-v0",
            details={"reason": str(exc)},
        )
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue(), {
        "width": image.width,
        "height": image.height,
        "sourceSha256": hashlib.sha256(source).hexdigest(),
    }


def _source_cameras(manifest: dict[str, Any]) -> dict[ViewRole, CameraRecord]:
    values = manifest.get("cameras")
    if not isinstance(values, list):
        fail(
            "template-source-cameras-missing",
            "源结果包缺少三视图相机",
            stage="template-head-v0",
        )
    try:
        cameras = [CameraRecord.model_validate(value) for value in values]
    except (TypeError, ValueError) as exc:
        fail(
            "template-source-cameras-invalid",
            "源结果包相机参数无效",
            stage="template-head-v0",
            details={"reason": str(exc)},
        )
    by_role = {camera.role: camera for camera in cameras}
    missing = [role.value for role in REQUIRED_VIEWS if role not in by_role]
    if missing or len(by_role) != len(cameras):
        fail(
            "template-source-cameras-invalid",
            "源结果包相机角色缺失或重复",
            stage="template-head-v0",
            details={"missing": missing},
        )
    return by_role


def _canonical_cameras(mesh: trimesh.Trimesh) -> dict[ViewRole, CameraRecord]:
    width = height = 1024
    focal = 1320.0
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    vertical_span = float(np.ptp(bounds[:, 1]))
    depth_radius = float(np.ptp(bounds[:, 2])) * 0.5
    distance = focal * vertical_span / (height * 0.72) + depth_radius
    yaw_by_role = {
        ViewRole.FRONT: 0.0,
        ViewRole.LEFT45: -45.0,
        ViewRole.RIGHT45: 45.0,
    }
    cameras: dict[ViewRole, CameraRecord] = {}
    flip = trimesh.transformations.rotation_matrix(np.pi, (1.0, 0.0, 0.0))[:3, :3]
    for role, yaw_degrees in yaw_by_role.items():
        yaw = trimesh.transformations.rotation_matrix(
            np.radians(yaw_degrees),
            (0.0, 1.0, 0.0),
        )[:3, :3]
        rotation = flip @ yaw
        rotation_vector, _ = cv2.Rodrigues(rotation)
        translation = np.asarray([0.0, 0.0, distance]) - center @ rotation.T
        cameras[role] = CameraRecord(
            role=role,
            width=width,
            height=height,
            focal_length_px=focal,
            principal_point_px=(width / 2, height / 2),
            rotation_vector=tuple(float(value) for value in rotation_vector.reshape(3)),
            translation=tuple(float(value) for value in translation),
            yaw_deg=yaw_degrees,
            pitch_deg=0.0,
            roll_deg=0.0,
        )
    return cameras


def _side_camera(front: CameraRecord) -> CameraRecord:
    front_rotation, _ = cv2.Rodrigues(np.asarray(front.rotation_vector, dtype=np.float64))
    yaw_rotation = trimesh.transformations.rotation_matrix(np.radians(-90.0), (0.0, 1.0, 0.0))[
        :3, :3
    ]
    rotation_vector, _ = cv2.Rodrigues(front_rotation @ yaw_rotation)
    return front.model_copy(
        update={
            "rotation_vector": tuple(float(value) for value in rotation_vector.reshape(3)),
            "yaw_deg": -90.0,
        }
    )


def prepare_template_head_v0(
    package: Path | None,
    quality_baseline: Path,
    output: Path,
    *,
    source_glb: Path | None = None,
    source_license: Path | None = None,
) -> dict[str, Any]:
    quality_baseline = quality_baseline.expanduser().resolve()
    output = output.expanduser().resolve()
    if (package is None) == (source_glb is None):
        fail(
            "template-source-selection-invalid",
            "必须且只能指定 --package 或 --source-glb 之一",
            stage="template-head-v0",
        )
    if not quality_baseline.is_file():
        fail(
            "template-quality-baseline-missing",
            "TemplateHeadV0 质量基准图不存在",
            stage="template-head-v0",
            details={"path": str(quality_baseline)},
        )
    if output.exists():
        fail(
            "template-output-exists",
            "TemplateHeadV0 输出目录已存在；为避免覆盖，请指定新目录",
            stage="template-head-v0",
            details={"path": str(output)},
        )

    source_payloads: dict[str, bytes]
    source_artifacts: dict[str, str]
    source_descriptor: dict[str, Any]
    if package is not None:
        package = package.expanduser().resolve()
        if not package.is_file():
            fail(
                "template-source-package-missing",
                "TemplateHeadV0 源结果包不存在",
                stage="template-head-v0",
                details={"path": str(package)},
            )
        source_manifest, compute_glb, render_glb = _read_package(package)
        source_compute_mesh = _load_single_mesh(
            compute_glb,
            label=PACKAGE_ENTRIES["compute"],
        )
        compute_mesh, topology_preparation = _unify_compute_mesh(source_compute_mesh)
        render_mesh = (
            _load_single_mesh(render_glb, label=PACKAGE_ENTRIES["render"])
            if render_glb is not None and not topology_preparation["required"]
            else None
        )
        cameras = _source_cameras(source_manifest)
        source_payloads = {
            "source/smooth.glb": compute_glb,
            "source/package-manifest.json": (
                json.dumps(
                    source_manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            ),
        }
        source_artifacts = {
            "sourceCompute": "source/smooth.glb",
            "sourceManifest": "source/package-manifest.json",
        }
        source_hashes = {
            "package": sha256_file(package),
            "computeGlb": hashlib.sha256(compute_glb).hexdigest(),
        }
        if render_glb is not None:
            source_payloads["source/skin.glb"] = render_glb
            source_artifacts["sourceRender"] = "source/skin.glb"
            source_hashes["renderGlb"] = hashlib.sha256(render_glb).hexdigest()
        source_descriptor = {
            "kind": "viewforge3d-package",
            "packageSchemaVersion": source_manifest.get("schemaVersion"),
            "hashes": source_hashes,
        }
    else:
        assert source_glb is not None
        source_glb = source_glb.expanduser().resolve()
        if not source_glb.is_file():
            fail(
                "template-source-glb-missing",
                "TemplateHeadV0 源 GLB 不存在",
                stage="template-head-v0",
                details={"path": str(source_glb)},
            )
        if source_license is None:
            fail(
                "template-source-license-missing",
                "直接提取 GLB 必须提供许可证文件",
                stage="template-head-v0",
            )
        source_license = source_license.expanduser().resolve()
        if not source_license.is_file() or source_license.stat().st_size == 0:
            fail(
                "template-source-license-missing",
                "TemplateHeadV0 源许可证不存在或为空",
                stage="template-head-v0",
                details={"path": str(source_license)},
            )
        source_bytes = source_glb.read_bytes()
        license_bytes = source_license.read_bytes()
        source_render_mesh = _load_single_mesh(source_bytes, label=source_glb.name)
        compute_mesh, topology_preparation = _weld_uv_seams(source_render_mesh)
        # Preserve the licensed source GLB verbatim, but generate a fresh atlas
        # for the template. The scan UV contains collapsed triangles that would
        # leave unprojectable skin regions after non-rigid deformation.
        render_mesh = None
        cameras = _canonical_cameras(compute_mesh)
        source_payloads = {
            "source/original.glb": source_bytes,
            "source/LICENSE.txt": license_bytes,
        }
        source_artifacts = {
            "sourceOriginal": "source/original.glb",
            "sourceLicense": "source/LICENSE.txt",
        }
        source_descriptor = {
            "kind": "direct-glb",
            "name": source_glb.name,
            "licenseFile": source_license.name,
            "uvPolicy": "regenerate-stable-xatlas-from-compute-topology",
            "hashes": {
                "sourceGlb": hashlib.sha256(source_bytes).hexdigest(),
                "license": hashlib.sha256(license_bytes).hexdigest(),
            },
        }

    asset, topology, uv_metrics, uv_method = _extract_raw_asset(
        compute_mesh,
        render_mesh,
    )
    baseline_bytes = quality_baseline.read_bytes()
    normalized_baseline, baseline_metrics = _normalized_baseline(baseline_bytes)
    baseline_metrics["normalizedSha256"] = hashlib.sha256(normalized_baseline).hexdigest()

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as root:
        staging = Path(root) / output.name
        staging.mkdir()
        for relative, payload in source_payloads.items():
            atomic_write_bytes(staging / relative, payload)
        atomic_write_bytes(
            staging / "reference" / "quality-baseline.png",
            normalized_baseline,
        )
        asset.save(staging / "template" / "template-head-v0.raw.npz")
        export_neutral_mesh(
            asset.compute_mesh,
            staging / "template" / "template-head-v0.glb",
        )

        qa_cameras: dict[str, CameraRecord] = {role.value: cameras[role] for role in REQUIRED_VIEWS}
        qa_cameras["side"] = _side_camera(cameras[ViewRole.FRONT])
        for name, camera in qa_cameras.items():
            render_flat_mesh(
                asset.compute_mesh,
                camera,
                staging / "qa" / f"fixed-view-{name}.png",
                width=720,
                height=720,
            )

        artifacts = {
            "rawTemplate": "template/template-head-v0.raw.npz",
            "neutralTemplate": "template/template-head-v0.glb",
            "qualityBaseline": "reference/quality-baseline.png",
            "fixedViews": {name: f"qa/fixed-view-{name}.png" for name in qa_cameras},
            **source_artifacts,
        }
        artifact_hashes = {
            name: sha256_file(staging / relative)
            for name, relative in artifacts.items()
            if isinstance(relative, str)
        }
        manifest = {
            "schemaVersion": RAW_SCHEMA_VERSION,
            "templateId": TEMPLATE_ID,
            "state": "raw-extracted",
            "source": source_descriptor,
            "qualityBaseline": {
                **baseline_metrics,
                "role": "minimum-visual-quality",
                "requiredViews": ["front", "left45", "right45", "side"],
            },
            "geometry": {
                "sha256": asset.geometry_sha256,
                "computeVertexCount": len(asset.compute_vertices),
                "computeFaceCount": len(asset.compute_faces),
                "bounds": np.asarray(asset.compute_mesh.bounds, dtype=float).tolist(),
                **topology,
            },
            "topologyPreparation": topology_preparation,
            "renderTopology": {
                "renderVertexCount": len(asset.render_to_compute),
                "renderFaceCount": len(asset.render_faces),
                "maximumPositionDifference": asset.metadata["mapping"]["maximumPositionDifference"],
                "mappedFaceMismatchCount": asset.metadata["mapping"]["mappedFaceMismatchCount"],
            },
            "uv": {
                **uv_metrics,
                "method": uv_method,
                "state": (
                    "source-uv-extracted"
                    if uv_metrics["sourceUvStable"]
                    else "source-uv-requires-normalization"
                ),
            },
            "readiness": {
                "extractionReady": True,
                "continuousHeadNeckTopologyReady": True,
                "stableUvReady": bool(uv_metrics["sourceUvStable"]),
                "visualBaselineReviewed": False,
                "semanticRegionsReady": False,
                "openEyelidRingsReady": False,
                "completeEyeballsReady": False,
                "nonRigidDeformationReady": False,
            },
            "route": {
                "surfaceSource": "template-non-rigid-deformation",
                "sdfRole": "qa-only",
                "forbidden": [
                    "cube-or-voxel-surface-generation",
                    "separate-cranium-generation",
                    "detached-ear-carrier",
                ],
            },
            "cameras": {
                name: camera.model_dump(mode="json") for name, camera in qa_cameras.items()
            },
            "artifacts": artifacts,
            "artifactSha256": artifact_hashes,
        }
        atomic_write_json(staging / "manifest.json", manifest)
        os.replace(staging, output)

    return {
        "ok": True,
        "templateId": TEMPLATE_ID,
        "state": "raw-extracted",
        "output": str(output),
        "geometrySha256": asset.geometry_sha256,
        "computeVertexCount": len(asset.compute_vertices),
        "renderVertexCount": len(asset.render_to_compute),
        "faceCount": len(asset.compute_faces),
        "sourceUvStable": uv_metrics["sourceUvStable"],
        "uvMethod": uv_method,
        "baselineSourceSha256": baseline_metrics["sourceSha256"],
    }
