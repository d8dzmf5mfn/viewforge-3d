#!/usr/bin/env python3
"""Build a QA-only low-poly ray-picking proxy in the source model coordinate system."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-triangles", type=int, default=60_000)
    parser.add_argument("--source-version", default="V24")
    args = parser.parse_args()
    source_version = args.source_version.upper()
    if not re.fullmatch(r"V[0-9]+", source_version):
        raise SystemExit("source version must have the form V<number>")

    source = np.load(args.source_npz)
    vertices = np.asarray(source["outer_vertices"], dtype=np.float64)
    faces = np.asarray(source["outer_faces"], dtype=np.int32)
    if not np.isfinite(vertices).all():
        raise SystemExit("source vertices contain non-finite values")
    if args.target_triangles < 10_000 or args.target_triangles >= len(faces):
        raise SystemExit("target triangle count must be between 10,000 and the source count")

    source_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces),
    )
    source_mesh.remove_duplicated_vertices()
    source_mesh.remove_degenerate_triangles()
    proxy = source_mesh.simplify_quadric_decimation(args.target_triangles)
    proxy.remove_degenerate_triangles()
    proxy.remove_duplicated_triangles()
    proxy.compute_vertex_normals()

    proxy_vertices = np.asarray(proxy.vertices, dtype=np.float64)
    proxy_faces = np.asarray(proxy.triangles, dtype=np.int64)
    if not np.isfinite(proxy_vertices).all() or len(proxy_faces) == 0:
        raise SystemExit("decimation produced an invalid proxy")

    mesh = trimesh.Trimesh(proxy_vertices, proxy_faces, process=False, validate=False)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.tile(
            np.array([236, 127, 111, 255], dtype=np.uint8),
            (len(mesh.vertices), 1),
        ),
    )
    scene = trimesh.Scene()
    proxy_name = f"{source_version}AnnotationPickProxy"
    scene.add_geometry(mesh, node_name=proxy_name, geom_name=proxy_name)
    atomic_write(args.output, trimesh.exchange.gltf.export_glb(scene, include_normals=True))

    persisted = trimesh.load(args.output, force="mesh", process=False)
    if not isinstance(persisted, trimesh.Trimesh):
        raise SystemExit("persisted proxy is not a single mesh")
    source_bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    proxy_bounds = np.asarray(persisted.bounds)
    bounds_delta = np.abs(proxy_bounds - source_bounds)
    report = {
        "schemaVersion": 1,
        "role": "qa-only-ray-picking-proxy",
        "surfaceGenerated": False,
        "coordinateSystem": f"identical to {source_version} source model",
        "source": {
            "npz": str(args.source_npz.resolve()),
            "npzSha256": sha256(args.source_npz),
            "model": str(args.source_model.resolve()),
            "modelSha256": sha256(args.source_model),
            "outerVertices": int(len(vertices)),
            "outerTriangles": int(len(faces)),
        },
        "proxy": {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output),
            "vertices": int(len(persisted.vertices)),
            "triangles": int(len(persisted.faces)),
            "finite": bool(np.isfinite(persisted.vertices).all()),
            "maximumBoundsDelta": float(bounds_delta.max()),
        },
        "warning": (
            "Use only for interactive surface hit sampling. Never modify, render, validate, "
            f"or deliver this proxy as the {source_version} surface."
        ),
    }
    atomic_write(args.report, (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode())
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
