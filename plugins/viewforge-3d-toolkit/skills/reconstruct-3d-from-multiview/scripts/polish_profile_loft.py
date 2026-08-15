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


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def visual_signature(mesh: trimesh.Trimesh) -> dict[str, Any]:
    visual = mesh.visual
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    fields: dict[str, Any] = {}
    for name in (
        "name",
        "baseColorFactor",
        "metallicFactor",
        "roughnessFactor",
        "emissiveFactor",
        "alphaMode",
        "alphaCutoff",
        "doubleSided",
        "diffuse",
        "ambient",
        "specular",
        "glossiness",
    ):
        value = getattr(material, name, None)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if value is None or isinstance(value, (str, int, float, bool, list)):
            fields[name] = value
    texture_hashes: dict[str, str] = {}
    for name in (
        "image",
        "baseColorTexture",
        "metallicRoughnessTexture",
        "normalTexture",
        "occlusionTexture",
        "emissiveTexture",
    ):
        texture = getattr(material, name, None)
        if texture is not None:
            texture_hashes[name] = array_hash(np.asarray(texture))
    material_payload = json.dumps(
        {
            "type": type(material).__name__ if material is not None else None,
            "fields": fields,
            "textures": texture_hashes,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "kind": str(getattr(visual, "kind", None)),
        "uvHash": array_hash(np.asarray(uv, dtype="<f4")) if uv is not None else None,
        "materialHash": hashlib.sha256(material_payload).hexdigest(),
    }


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
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    atomic_write_bytes(path, data)


def load_single_identity_scene(path: Path) -> tuple[trimesh.Scene, str, trimesh.Trimesh]:
    scene = trimesh.load(path, force="scene", process=False)
    if not isinstance(scene, trimesh.Scene) or len(scene.geometry) != 1:
        raise ValueError(f"expected one mesh geometry: {path}")
    nodes = list(scene.graph.nodes_geometry)
    if len(nodes) != 1:
        raise ValueError(f"expected one geometry node, found {len(nodes)}")
    transform, geometry_name = scene.graph.get(nodes[0])
    if not np.allclose(transform, np.eye(4), atol=1e-9):
        raise ValueError("input scene transform must be identity")
    mesh = scene.geometry[geometry_name]
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("scene geometry is not a triangular mesh")
    return scene, geometry_name, mesh


def boundary_counts(faces: np.ndarray) -> tuple[int, int]:
    edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts == 1)), int(np.count_nonzero(counts > 2))


def topology_gates(mesh: trimesh.Trimesh) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    boundary_edges, non_manifold_edges = boundary_counts(faces)
    components = mesh.split(only_watertight=False)
    scale = float(max(np.max(mesh.extents), 1e-9))
    degenerate_threshold = max(scale * scale * 1e-14, 1e-18)
    return {
        "finite": bool(np.all(np.isfinite(vertices))),
        "connectedComponents": int(len(components)),
        "boundaryEdges": boundary_edges,
        "nonManifoldEdges": non_manifold_edges,
        "degenerateFaces": int(np.count_nonzero(mesh.area_faces <= degenerate_threshold)),
        "watertight": bool(mesh.is_watertight),
        "windingConsistent": bool(mesh.is_winding_consistent),
    }


def require_safe_topology(label: str, gates: dict[str, Any]) -> None:
    required = {
        "finite": True,
        "connectedComponents": 1,
        "boundaryEdges": 0,
        "nonManifoldEdges": 0,
        "degenerateFaces": 0,
        "watertight": True,
        "windingConsistent": True,
    }
    failures = {
        key: gates.get(key) for key, expected in required.items() if gates.get(key) != expected
    }
    if failures:
        raise ValueError(f"{label} topology gates failed: {failures}")


def loft_grid(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, int, int]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    if len(vertices) < 10:
        raise ValueError("mesh has too few vertices for a capped regular loft")
    first_y = vertices[0, 1]
    transitions = np.flatnonzero(np.abs(vertices[:, 1] - first_y) > 1e-8)
    if not len(transitions):
        raise ValueError("cannot infer radial sample count")
    radial_samples = int(transitions[0])
    if radial_samples < 8 or (len(vertices) - 2) % radial_samples:
        raise ValueError("vertex layout is not rings followed by two cap vertices")
    vertical_samples = (len(vertices) - 2) // radial_samples
    if vertical_samples < 4:
        raise ValueError("loft has too few vertical rings")
    grid = vertices[: vertical_samples * radial_samples].reshape(
        vertical_samples, radial_samples, 3
    )
    y_values = grid[:, 0, 1].copy()
    if not np.allclose(grid[:, :, 1], y_values[:, None], atol=1e-8):
        raise ValueError("vertices inside each loft ring must share one Y coordinate")
    if not np.all(np.diff(y_values) > 0.0):
        raise ValueError("loft rings must be ordered bottom-to-top")
    return vertices, grid, vertical_samples, radial_samples


def polish(
    mesh: trimesh.Trimesh,
    *,
    lower_y: float,
    upper_y: float,
    support_rows: int,
    strength: float,
) -> dict[str, Any]:
    if support_rows < 1:
        raise ValueError("support rows must be positive")
    if not 0.0 < strength <= 1.0:
        raise ValueError("strength must be in (0, 1]")
    vertices, grid, vertical_samples, radial_samples = loft_grid(mesh)
    original = grid.copy()
    y_values = grid[:, 0, 1]
    lower = int(np.argmin(np.abs(y_values - lower_y)))
    upper = int(np.argmin(np.abs(y_values - upper_y)))
    if lower >= upper:
        raise ValueError("lower band boundary must precede upper boundary")
    if lower - support_rows < 0 or upper + support_rows >= vertical_samples:
        raise ValueError("support rows extend outside the loft")

    y0 = float(y_values[lower])
    y1 = float(y_values[upper])
    span = y1 - y0
    p0 = original[lower, :, (0, 2)]
    p1 = original[upper, :, (0, 2)]
    m0 = (p0 - original[lower - support_rows, :, (0, 2)]) / (
        y0 - float(y_values[lower - support_rows])
    )
    m1 = (original[upper + support_rows, :, (0, 2)] - p1) / (
        float(y_values[upper + support_rows]) - y1
    )
    for row in range(lower, upper + 1):
        t = float((y_values[row] - y0) / span)
        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h10 = t**3 - 2.0 * t**2 + t
        h01 = -2.0 * t**3 + 3.0 * t**2
        h11 = t**3 - t**2
        target = h00 * p0 + h10 * span * m0 + h01 * p1 + h11 * span * m1
        current = original[row, :, (0, 2)]
        grid[row, :, (0, 2)] = current + strength * (target - current)

    mesh.vertices = vertices
    displacement = np.linalg.norm(grid - original, axis=2)
    edited = displacement > 1e-10
    outside = edited.copy()
    outside[lower : upper + 1] = False
    y_changed = not np.array_equal(grid[:, :, 1], original[:, :, 1])
    return {
        "requestedBandY": [float(lower_y), float(upper_y)],
        "resolvedBandY": [y0, y1],
        "resolvedRows": [lower, upper],
        "supportRows": int(support_rows),
        "strength": float(strength),
        "verticalSamples": int(vertical_samples),
        "radialSamples": int(radial_samples),
        "editedVertices": int(np.count_nonzero(edited)),
        "maximumDisplacement": float(displacement.max()),
        "meanEditedDisplacement": float(displacement[edited].mean()) if np.any(edited) else 0.0,
        "verticesChangedOutsideBand": int(np.count_nonzero(outside)),
        "yCoordinatesChanged": bool(y_changed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Polish one Y band of a regular profile loft without changing topology."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lower-y", required=True, type=float)
    parser.add_argument("--upper-y", required=True, type=float)
    parser.add_argument("--support-rows", type=int, default=8)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--maximum-displacement", type=float)
    parser.add_argument("--metrics", type=Path)
    arguments = parser.parse_args()

    source = arguments.input.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    metrics_path = (
        arguments.metrics.expanduser().resolve()
        if arguments.metrics
        else output.with_suffix(".metrics.json")
    )
    if not source.is_file():
        parser.error(f"input does not exist: {source}")
    if output.suffix.lower() != ".glb":
        parser.error("output must use .glb")
    if output.exists() or metrics_path.exists():
        parser.error("output and metrics paths must not already exist")
    if output == source or metrics_path in (source, output):
        parser.error("input, output, and metrics paths must be distinct")

    scene, geometry_name, mesh = load_single_identity_scene(source)
    original_faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    original_vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    original_face_normals = np.asarray(mesh.face_normals, dtype=np.float64).copy()
    original_visual = visual_signature(mesh)
    input_gates = topology_gates(mesh)
    require_safe_topology("input", input_gates)
    before_geometry = geometry_hash(original_vertices, original_faces)
    before_topology = topology_hash(len(original_vertices), original_faces)

    polished = mesh.copy()
    edit = polish(
        polished,
        lower_y=arguments.lower_y,
        upper_y=arguments.upper_y,
        support_rows=arguments.support_rows,
        strength=arguments.strength,
    )
    if edit["editedVertices"] == 0:
        raise ValueError("polish produced no geometry change")
    if (
        arguments.maximum_displacement is not None
        and edit["maximumDisplacement"] > arguments.maximum_displacement
    ):
        raise ValueError(
            f"maximum displacement {edit['maximumDisplacement']:.9g} exceeds "
            f"{arguments.maximum_displacement:.9g}"
        )
    if edit["verticesChangedOutsideBand"] != 0 or edit["yCoordinatesChanged"]:
        raise ValueError("polish escaped the selected XZ band")
    if not np.array_equal(original_faces, np.asarray(polished.faces)):
        raise ValueError("in-memory face topology changed")
    if not np.array_equal(original_vertices[:, 1], np.asarray(polished.vertices)[:, 1]):
        raise ValueError("Y coordinates changed")
    opposed_normals = int(
        np.count_nonzero(np.einsum("ij,ij->i", original_face_normals, polished.face_normals) < 0.0)
    )
    if opposed_normals:
        raise ValueError(f"{opposed_normals} face normals reversed direction")

    output_gates = topology_gates(polished)
    require_safe_topology("polished", output_gates)
    after_geometry = geometry_hash(polished.vertices, polished.faces)
    after_topology = topology_hash(len(polished.vertices), polished.faces)
    if before_geometry == after_geometry or before_topology != after_topology:
        raise ValueError("geometry/topology hash invariants failed")

    scene.geometry[geometry_name] = polished
    payload = scene.export(file_type="glb")
    if not isinstance(payload, bytes):
        raise ValueError("GLB exporter did not return bytes")

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".glb", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _, _, persisted = load_single_identity_scene(temporary)
        persisted_gates = topology_gates(persisted)
        require_safe_topology("persisted", persisted_gates)
        persisted_faces = np.asarray(persisted.faces, dtype=np.int64)
        if not np.array_equal(original_faces, persisted_faces):
            raise ValueError("persisted face array changed")
        persisted_topology = topology_hash(len(persisted.vertices), persisted_faces)
        if persisted_topology != before_topology:
            raise ValueError("persisted topology hash changed")
        persisted_visual = visual_signature(persisted)
        if persisted_visual != original_visual:
            raise ValueError(
                f"persisted visual/material signature changed: "
                f"{original_visual} != {persisted_visual}"
            )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    result = {
        "schemaVersion": 1,
        "route": "profile-loft-preview",
        "state": "preview",
        "previewOnly": True,
        "identityAcceptanceAllowed": False,
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "output": {"path": str(output), "sha256": sha256_file(output)},
        "geometryHashBefore": before_geometry,
        "geometryHashAfter": after_geometry,
        "topologyHashBefore": before_topology,
        "topologyHashAfter": after_topology,
        "edit": edit,
        "gates": {
            "input": input_gates,
            "polished": output_gates,
            "persisted": persisted_gates,
            "geometryChanged": before_geometry != after_geometry,
            "topologyUnchanged": before_topology == after_topology == persisted_topology,
            "facesUnchanged": True,
            "opposedFaceNormals": opposed_normals,
            "visualLineagePreserved": persisted_visual == original_visual,
        },
        "visualSignature": original_visual,
        "requiredVisualReviewYawDegrees": [0, 45, 90],
        "userSignoff": False,
    }
    atomic_write_json(metrics_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
