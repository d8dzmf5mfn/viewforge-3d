#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometry_hash(vertices: np.ndarray, faces: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(vertices, dtype="<f4").tobytes())
    digest.update(np.ascontiguousarray(faces, dtype="<u4").tobytes())
    return digest.hexdigest()


def topology_hash(vertex_count: int, faces: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray([vertex_count], dtype="<u8").tobytes())
    digest.update(np.ascontiguousarray(faces, dtype="<u4").tobytes())
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_npz(path: Path) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        if "vertices" not in payload.files or "faces" not in payload.files:
            raise ValueError("NPZ must contain vertices and faces")
        vertices = np.asarray(payload["vertices"], dtype=np.float64)
        faces = np.asarray(payload["faces"], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh, {
        "format": "npz",
        "geometryCount": 1,
        "geometryNodeCount": 1,
        "identitySceneTransform": True,
        "material": None,
    }


def material_record(mesh: trimesh.Trimesh) -> dict[str, Any] | None:
    material = getattr(mesh.visual, "material", None)
    if material is None:
        return None
    fields: dict[str, Any] = {"type": type(material).__name__}
    for name in (
        "name",
        "baseColorFactor",
        "metallicFactor",
        "roughnessFactor",
        "emissiveFactor",
        "alphaMode",
        "alphaCutoff",
        "doubleSided",
    ):
        value = getattr(material, name, None)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if value is None or isinstance(value, (str, int, float, bool, list)):
            fields[name] = value
    return fields


def load_glb(path: Path) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    scene = trimesh.load(path, force="scene", process=False)
    if not isinstance(scene, trimesh.Scene):
        raise ValueError("GLB did not load as a scene")
    geometry_count = len(scene.geometry)
    nodes = list(scene.graph.nodes_geometry)
    if geometry_count != 1 or len(nodes) != 1:
        raise ValueError(
            f"GLB must contain one geometry and one geometry node, found "
            f"{geometry_count} and {len(nodes)}"
        )
    transform, geometry_name = scene.graph.get(nodes[0])
    identity = bool(np.allclose(transform, np.eye(4), atol=1e-9))
    if not identity:
        raise ValueError("GLB geometry node transform is not identity")
    mesh = scene.geometry[geometry_name]
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("GLB geometry is not a triangle mesh")
    return mesh, {
        "format": "glb",
        "geometryCount": geometry_count,
        "geometryNodeCount": len(nodes),
        "geometryName": geometry_name,
        "nodeName": nodes[0],
        "identitySceneTransform": identity,
        "material": material_record(mesh),
    }


def load_mesh(path: Path) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_npz(path)
    if suffix == ".glb":
        return load_glb(path)
    raise ValueError("input must use .npz or .glb")


def boundary_counts(faces: np.ndarray) -> tuple[int, int]:
    edges = np.sort(
        np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts == 1)), int(np.count_nonzero(counts > 2))


def topology_record(mesh: trimesh.Trimesh) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"invalid vertex array: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"invalid face array: {faces.shape}")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("mesh must not be empty")
    if int(np.min(faces)) < 0 or int(np.max(faces)) >= len(vertices):
        raise ValueError("face array contains an out-of-range index")
    boundary_edges, non_manifold_edges = boundary_counts(faces)
    scale = float(max(np.max(mesh.extents), 1e-9))
    threshold = max(scale * scale * 1e-14, 1e-18)
    canonical_faces = np.sort(faces, axis=1)
    return {
        "vertices": int(len(vertices)),
        "triangles": int(len(faces)),
        "finite": bool(np.all(np.isfinite(vertices))),
        "connectedComponents": int(len(mesh.split(only_watertight=False))),
        "boundaryEdges": boundary_edges,
        "nonManifoldEdges": non_manifold_edges,
        "degenerateFaces": int(np.count_nonzero(mesh.area_faces <= threshold)),
        "duplicateFaces": int(len(faces) - len(np.unique(canonical_faces, axis=0))),
        "watertight": bool(mesh.is_watertight),
        "windingConsistent": bool(mesh.is_winding_consistent),
        "positiveVolume": bool(mesh.volume > 0.0),
        "volume": float(mesh.volume),
        "bounds": np.asarray(mesh.bounds, dtype=np.float64).tolist(),
        "geometryHash": geometry_hash(vertices, faces),
        "topologyHash": topology_hash(len(vertices), faces),
    }


def topology_failures(record: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "finite": True,
        "connectedComponents": 1,
        "boundaryEdges": 0,
        "nonManifoldEdges": 0,
        "degenerateFaces": 0,
        "duplicateFaces": 0,
        "watertight": True,
        "windingConsistent": True,
        "positiveVolume": True,
    }
    return {
        name: record.get(name)
        for name, wanted in expected.items()
        if record.get(name) != wanted
    }


def exact_self_intersections(mesh: trimesh.Trimesh) -> dict[str, Any]:
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required for exact self-intersection checks") from error
    open3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    pairs = np.asarray(open3d_mesh.get_self_intersecting_triangles(), dtype=np.int64)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int64)
    unique_faces = np.unique(pairs) if len(pairs) else np.empty(0, dtype=np.int64)
    centroids = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[unique_faces]].mean(axis=1)
    return {
        "performed": True,
        "method": "Open3D TriangleMesh.get_self_intersecting_triangles",
        "passed": len(pairs) == 0,
        "selfIntersectingTrianglePairs": int(len(pairs)),
        "pairs": pairs.tolist(),
        "affectedFaceCentroids": centroids.tolist(),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed topology and optional exact self-intersection checks for NPZ/GLB."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--exact-self-intersections", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    source = arguments.input.expanduser().resolve()
    metrics = arguments.metrics.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if metrics.exists():
        raise FileExistsError(metrics)
    if source == metrics:
        raise ValueError("input and metrics paths must be distinct")

    mesh, container = load_mesh(source)
    topology = topology_record(mesh)
    failures = topology_failures(topology)
    self_intersections = (
        exact_self_intersections(mesh)
        if arguments.exact_self_intersections
        else {
            "performed": False,
            "passed": None,
            "selfIntersectingTrianglePairs": None,
        }
    )
    if arguments.exact_self_intersections and not self_intersections["passed"]:
        failures["selfIntersectingTrianglePairs"] = self_intersections[
            "selfIntersectingTrianglePairs"
        ]
    complete = arguments.exact_self_intersections
    passed = not failures and complete
    record = {
        "schemaVersion": 1,
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "container": container,
        "topology": topology,
        "exactSelfIntersections": self_intersections,
        "gates": {
            "failures": failures,
            "basicTopologyPassed": not topology_failures(topology),
            "completeSurfaceGate": complete,
            "automaticPassed": passed,
        },
    }
    atomic_write_json(metrics, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
