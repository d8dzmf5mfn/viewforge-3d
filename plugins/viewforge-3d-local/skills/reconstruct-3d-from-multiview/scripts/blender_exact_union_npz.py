#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import bmesh
import bpy
import numpy as np


def parse_arguments() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(
        description="Sequentially union closed NPZ meshes with Blender's exact Boolean solver."
    )
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument(
        "--operand",
        required=True,
        action="append",
        metavar="LABEL=PATH",
        help="Repeat in the exact union order, for example bridge=bridge.npz",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--seam-cleanup", action="store_true")
    return parser.parse_args(argv)


def parse_operand(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"operand must use LABEL=PATH: {value}")
    label, raw_path = value.split("=", 1)
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]{0,63}", label):
        raise ValueError(f"invalid operand label: {label}")
    return label, Path(raw_path).expanduser().resolve()


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablock in tuple(bpy.data.meshes):
        if datablock.users == 0:
            bpy.data.meshes.remove(datablock)


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if "vertices" not in payload.files or "faces" not in payload.files:
            raise ValueError(f"NPZ lacks vertices or faces: {path}")
        vertices = np.asarray(payload["vertices"], dtype=np.float64)
        faces = np.asarray(payload["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"invalid vertex array in {path}: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"invalid face array in {path}: {faces.shape}")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"empty mesh in {path}")
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"non-finite vertices in {path}")
    if int(np.min(faces)) < 0 or int(np.max(faces)) >= len(vertices):
        raise ValueError(f"out-of-range face index in {path}")
    signed_volume = float(
        np.einsum(
            "ij,ij->i",
            vertices[faces[:, 0]],
            np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]]),
        ).sum()
        / 6.0
    )
    if signed_volume <= 0.0:
        raise ValueError(f"mesh must have positive outward winding: {path}")
    return vertices, faces


def load_object(name: str, path: Path) -> bpy.types.Object:
    vertices, faces = load_npz(path)
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def mesh_counts(obj: bpy.types.Object) -> dict[str, int]:
    return {
        "vertices": len(obj.data.vertices),
        "edges": len(obj.data.edges),
        "polygons": len(obj.data.polygons),
        "triangles": sum(max(1, len(poly.vertices) - 2) for poly in obj.data.polygons),
    }


def exact_union(
    base: bpy.types.Object,
    operand: bpy.types.Object,
    label: str,
) -> dict[str, Any]:
    before = {"base": mesh_counts(base), "operand": mesh_counts(operand)}
    bpy.context.view_layer.objects.active = base
    base.select_set(True)
    operand.select_set(False)
    modifier = base.modifiers.new(name=f"ExactUnion_{label}", type="BOOLEAN")
    modifier.operation = "UNION"
    modifier.solver = "EXACT"
    modifier.operand_type = "OBJECT"
    modifier.object = operand
    modifier.use_self = False
    modifier.use_hole_tolerant = True
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.data.objects.remove(operand, do_unlink=True)
    base.data.update(calc_edges=True)
    return {
        "label": label,
        "operation": "UNION",
        "solver": "EXACT",
        "holeTolerant": True,
        "before": before,
        "after": mesh_counts(base),
    }


def clean_and_triangulate(
    obj: bpy.types.Object,
    *,
    merge_seams: bool,
) -> dict[str, Any]:
    before = mesh_counts(obj)
    merge_distance = 1e-7
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    if merge_seams:
        bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=merge_distance)
        bmesh.ops.dissolve_degenerate(bm, dist=merge_distance, edges=list(bm.edges))
    bmesh.ops.triangulate(
        bm,
        faces=list(bm.faces),
        quad_method="BEAUTY",
        ngon_method="BEAUTY",
    )
    bm.normal_update()
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update(calc_edges=True)
    return {
        "method": (
            "merge exact/near-exact seam duplicates then dissolve collapsed edges"
            if merge_seams
            else "triangulate exact Boolean polygons without seam merging"
        ),
        "mergeSeams": merge_seams,
        "mergeDistance": merge_distance if merge_seams else None,
        "before": before,
        "after": mesh_counts(obj),
    }


def export_npz_atomic(obj: bpy.types.Object, path: Path) -> None:
    if any(len(poly.vertices) != 3 for poly in obj.data.polygons):
        raise ValueError("final Boolean mesh is not triangulated")
    vertices = np.asarray([vertex.co[:] for vertex in obj.data.vertices], dtype=np.float32)
    faces = np.asarray([poly.vertices[:] for poly in obj.data.polygons], dtype=np.int32)
    if not np.all(np.isfinite(vertices)):
        raise ValueError("final Boolean mesh contains non-finite vertices")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("final Boolean mesh is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, vertices=vertices, faces=faces)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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


def main() -> int:
    arguments = parse_arguments()
    base_path = arguments.base.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    stats_path = arguments.stats.expanduser().resolve()
    operands = [parse_operand(value) for value in arguments.operand]
    source_paths = [base_path, *(path for _, path in operands)]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if len(set(source_paths)) != len(source_paths):
        raise ValueError("base and operand paths must be distinct")
    if output.suffix.lower() != ".npz":
        raise ValueError("output must use .npz")
    if output.exists() or stats_path.exists():
        raise FileExistsError("output or stats path already exists")
    if output in source_paths or stats_path in {*source_paths, output}:
        raise ValueError("source, output, and stats paths must be distinct")

    clear_scene()
    base = load_object("AcceptedBase", base_path)
    stages: list[dict[str, Any]] = []
    for index, (label, path) in enumerate(operands, start=1):
        operand = load_object(f"Operand{index}_{label}", path)
        stages.append(exact_union(base, operand, label))
    cleanup = clean_and_triangulate(base, merge_seams=arguments.seam_cleanup)
    export_npz_atomic(base, output)
    stats = {
        "schemaVersion": 1,
        "backend": "Blender Boolean modifier",
        "blenderVersion": bpy.app.version_string,
        "backgroundMode": bool(bpy.app.background),
        "surfaceMethod": "exact triangle-mesh Boolean union",
        "base": str(base_path),
        "operandOrder": [
            {"label": label, "path": str(path)} for label, path in operands
        ],
        "solver": "EXACT",
        "voxelOrSdfUsed": False,
        "marchingCubesUsed": False,
        "stages": stages,
        "cleanup": cleanup,
        "final": mesh_counts(base),
    }
    write_json_atomic(stats_path, stats)
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
