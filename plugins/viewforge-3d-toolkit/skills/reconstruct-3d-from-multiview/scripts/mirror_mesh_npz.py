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

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<u8").tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write_bytes(path, data)


def atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    if "vertices" not in arrays or "faces" not in arrays:
        raise ValueError("NPZ must contain vertices and faces")
    vertices = arrays["vertices"]
    faces = arrays["faces"]
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), found {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), found {faces.shape}")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("faces must use an integer dtype")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("mesh arrays must not be empty")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("vertices contain non-finite values")
    if int(np.min(faces)) < 0 or int(np.max(faces)) >= len(vertices):
        raise ValueError("faces contain an out-of-range vertex index")
    return arrays


def boundary_counts(faces: np.ndarray) -> tuple[int, int]:
    edges = np.sort(
        np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts == 1)), int(np.count_nonzero(counts > 2))


def topology_record(mesh: trimesh.Trimesh) -> dict[str, Any]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    boundary_edges, non_manifold_edges = boundary_counts(faces)
    scale = float(max(np.max(mesh.extents), 1e-9))
    degenerate_threshold = max(scale * scale * 1e-14, 1e-18)
    canonical_faces = np.sort(faces, axis=1)
    duplicate_faces = int(len(faces) - len(np.unique(canonical_faces, axis=0)))
    return {
        "finite": bool(np.all(np.isfinite(mesh.vertices))),
        "connectedComponents": int(len(mesh.split(only_watertight=False))),
        "boundaryEdges": boundary_edges,
        "nonManifoldEdges": non_manifold_edges,
        "degenerateFaces": int(np.count_nonzero(mesh.area_faces <= degenerate_threshold)),
        "duplicateFaces": duplicate_faces,
        "watertight": bool(mesh.is_watertight),
        "windingConsistent": bool(mesh.is_winding_consistent),
        "positiveVolume": bool(mesh.volume > 0.0),
        "volume": float(mesh.volume),
    }


def require_safe(label: str, record: dict[str, Any]) -> None:
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
    failures = {
        name: record.get(name)
        for name, wanted in expected.items()
        if record.get(name) != wanted
    }
    if failures:
        raise ValueError(f"{label} topology gates failed: {failures}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror one closed NPZ triangle mesh and preserve vertex-index metadata."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--axis", choices=tuple(AXIS_INDEX), default="x")
    parser.add_argument("--origin", type=float, default=0.0)
    parser.add_argument("--translate-x", type=float, default=0.0)
    parser.add_argument("--translate-y", type=float, default=0.0)
    parser.add_argument("--translate-z", type=float, default=0.0)
    parser.add_argument("--metrics", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    source = arguments.input.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    metrics = (
        arguments.metrics.expanduser().resolve()
        if arguments.metrics is not None
        else output.with_suffix(".metrics.json")
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".npz" or output.suffix.lower() != ".npz":
        raise ValueError("input and output must use .npz")
    if output.exists() or metrics.exists():
        raise FileExistsError("output and metrics paths must not already exist")
    if len({source, output, metrics}) != 3:
        raise ValueError("input, output, and metrics paths must be distinct")

    arrays = load_npz(source)
    source_vertices = np.asarray(arrays["vertices"], dtype=np.float64)
    source_faces = np.asarray(arrays["faces"], dtype=np.int64)
    source_mesh = trimesh.Trimesh(
        vertices=source_vertices,
        faces=source_faces,
        process=False,
    )
    source_topology = topology_record(source_mesh)
    require_safe("source", source_topology)

    axis_index = AXIS_INDEX[arguments.axis]
    translation = np.asarray(
        [arguments.translate_x, arguments.translate_y, arguments.translate_z],
        dtype=np.float64,
    )
    mirrored_vertices = source_vertices.copy()
    mirrored_vertices[:, axis_index] = 2.0 * arguments.origin - source_vertices[:, axis_index]
    mirrored_vertices += translation
    mirrored_faces = source_faces[:, [0, 2, 1]].copy()
    mirrored_mesh = trimesh.Trimesh(
        vertices=mirrored_vertices,
        faces=mirrored_faces,
        process=False,
    )
    mirrored_topology = topology_record(mirrored_mesh)
    require_safe("mirrored", mirrored_topology)

    expected_vertices = source_vertices.copy()
    expected_vertices[:, axis_index] = 2.0 * arguments.origin - source_vertices[:, axis_index]
    expected_vertices += translation
    maximum_vertex_residual = float(np.max(np.abs(mirrored_vertices - expected_vertices)))
    if maximum_vertex_residual != 0.0:
        raise ValueError(f"mirror residual is not exact: {maximum_vertex_residual}")
    if not np.array_equal(mirrored_faces, source_faces[:, [0, 2, 1]]):
        raise ValueError("mirrored face winding does not match the required reversal")

    output_arrays = dict(arrays)
    output_arrays["vertices"] = mirrored_vertices.astype(arrays["vertices"].dtype, copy=False)
    output_arrays["faces"] = mirrored_faces.astype(arrays["faces"].dtype, copy=False)
    atomic_write_npz(output, output_arrays)
    persisted = load_npz(output)
    for name, expected in output_arrays.items():
        if not np.array_equal(persisted[name], expected):
            raise ValueError(f"persisted array changed: {name}")
    persisted_mesh = trimesh.Trimesh(
        vertices=np.asarray(persisted["vertices"], dtype=np.float64),
        faces=np.asarray(persisted["faces"], dtype=np.int64),
        process=False,
    )
    persisted_topology = topology_record(persisted_mesh)
    require_safe("persisted", persisted_topology)

    record = {
        "schemaVersion": 1,
        "route": "profile-loft-preview",
        "state": "preview",
        "previewOnly": True,
        "identityAcceptanceAllowed": False,
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "output": {"path": str(output), "sha256": sha256_file(output)},
        "transform": {
            "operation": "plane reflection followed by translation",
            "axis": arguments.axis,
            "origin": float(arguments.origin),
            "translation": translation.tolist(),
            "determinantSign": -1,
            "triangleWindingReversed": True,
            "maximumCanonicalResidual": maximum_vertex_residual,
        },
        "arrays": {
            "vertices": len(mirrored_vertices),
            "triangles": len(mirrored_faces),
            "sourceVertexHash": array_hash(source_vertices),
            "sourceFaceHash": array_hash(source_faces),
            "outputVertexHash": array_hash(mirrored_vertices),
            "outputFaceHash": array_hash(mirrored_faces),
            "preservedExtraArrays": sorted(
                name for name in arrays if name not in {"vertices", "faces"}
            ),
        },
        "gates": {
            "source": source_topology,
            "mirrored": mirrored_topology,
            "persisted": persisted_topology,
            "vertexOrderPreserved": True,
            "faceWindingReversed": True,
            "extraArraysPreserved": True,
            "automaticPassed": True,
            "userSignoff": False,
        },
    }
    atomic_write_json(metrics, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
