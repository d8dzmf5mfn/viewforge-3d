from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector


def _arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--output-blend", type=Path, required=True)
    parser.add_argument("--output-glb", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    return parser.parse_args(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source(source: Path) -> None:
    if source.suffix.lower() == ".blend":
        current = Path(bpy.data.filepath).resolve() if bpy.data.filepath else None
        if current != source:
            raise RuntimeError("Blender did not load the staged Blend source.")
        return
    if source.suffix.lower() == ".glb":
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(source))
        return
    raise ValueError("Smoothing input must be a Blend or GLB file.")


def _load_options(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schemaVersion") != 1:
        raise ValueError("Smoothing options must use schemaVersion 1.")
    allowed = {
        "schemaVersion",
        "objectNames",
        "vertexGroup",
        "iterations",
        "strength",
        "preserveVolume",
        "preserveBoundaries",
        "maxDisplacementRatio",
        "shadeSmooth",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown smoothing option fields: {unknown}")
    return payload


def _mesh_topology_hash(mesh: bpy.types.Mesh) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<QQQ", len(mesh.vertices), len(mesh.edges), len(mesh.polygons)))
    for polygon in mesh.polygons:
        digest.update(struct.pack("<II", len(polygon.vertices), polygon.material_index))
        for vertex_index in polygon.vertices:
            digest.update(struct.pack("<I", vertex_index))
    return digest.hexdigest()


def _mesh_uv_hash(mesh: bpy.types.Mesh) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<I", len(mesh.uv_layers)))
    for layer in mesh.uv_layers:
        digest.update(layer.name.encode("utf-8"))
        digest.update(struct.pack("<I", len(layer.data)))
        for item in layer.data:
            digest.update(struct.pack("<dd", float(item.uv.x), float(item.uv.y)))
    return digest.hexdigest()


def _mesh_material_hash(object_: bpy.types.Object) -> str:
    digest = hashlib.sha256()
    for slot in object_.material_slots:
        digest.update((slot.name or "").encode("utf-8"))
        digest.update(b"\0")
    for polygon in object_.data.polygons:
        digest.update(struct.pack("<I", polygon.material_index))
    return digest.hexdigest()


def _adjacency(mesh: bpy.types.Mesh) -> list[set[int]]:
    neighbors = [set() for _ in mesh.vertices]
    for edge in mesh.edges:
        first, second = edge.vertices
        neighbors[first].add(second)
        neighbors[second].add(first)
    return neighbors


def _boundary_vertices(mesh: bpy.types.Mesh) -> set[int]:
    uses: dict[tuple[int, int], int] = {}
    for polygon in mesh.polygons:
        vertices = list(polygon.vertices)
        for index, first in enumerate(vertices):
            second = vertices[(index + 1) % len(vertices)]
            edge = (min(first, second), max(first, second))
            uses[edge] = uses.get(edge, 0) + 1
    return {vertex for edge, count in uses.items() if count == 1 for vertex in edge}


def _weights(
    object_: bpy.types.Object,
    vertex_group_name: str | None,
    boundary: set[int],
    preserve_boundaries: bool,
) -> list[float]:
    if vertex_group_name is None:
        weights = [1.0 for _ in object_.data.vertices]
    else:
        group = object_.vertex_groups.get(vertex_group_name)
        if group is None:
            raise ValueError(
                f"Object {object_.name!r} does not contain vertex group {vertex_group_name!r}."
            )
        weights = []
        for vertex in object_.data.vertices:
            try:
                weight = float(group.weight(vertex.index))
            except RuntimeError:
                weight = 0.0
            weights.append(min(max(weight, 0.0), 1.0))
    if preserve_boundaries:
        for index in boundary:
            weights[index] = 0.0
    return weights


def _laplacian_pass(
    coordinates: list[Vector],
    neighbors: list[set[int]],
    weights: list[float],
    coefficient: float,
) -> list[Vector]:
    result = [coordinate.copy() for coordinate in coordinates]
    for index, weight in enumerate(weights):
        if weight <= 0.0 or not neighbors[index]:
            continue
        average = sum((coordinates[item] for item in neighbors[index]), Vector())
        average /= len(neighbors[index])
        result[index] = coordinates[index] + (average - coordinates[index]) * (
            coefficient * weight
        )
    return result


def _world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [
        object_.matrix_world @ vertex.co
        for object_ in objects
        for vertex in object_.data.vertices
    ]
    if not points:
        raise RuntimeError("No mesh vertices are available for smoothing.")
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    if not all(math.isfinite(value) for value in (*minimum, *maximum)):
        raise RuntimeError("The source model has non-finite bounds.")
    if (maximum - minimum).length <= 1e-9:
        raise RuntimeError("The source model bounds are degenerate.")
    return minimum, maximum


def _selected_meshes(options: dict[str, Any]) -> list[bpy.types.Object]:
    available = {
        object_.name: object_
        for object_ in bpy.context.scene.objects
        if object_.type == "MESH"
    }
    requested = options.get("objectNames")
    if requested is None:
        selected = [available[name] for name in sorted(available)]
    else:
        if not isinstance(requested, list) or not requested:
            raise ValueError("objectNames must be null or a non-empty list.")
        if any(not isinstance(name, str) for name in requested):
            raise ValueError("Every objectNames entry must be a string.")
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Requested mesh objects are unavailable: {missing}")
        selected = [available[name] for name in requested]
    if not selected:
        raise RuntimeError("The source contains no mesh objects to smooth.")
    for object_ in selected:
        if object_.data.shape_keys is not None:
            raise RuntimeError(f"Object {object_.name!r} has shape keys and was not modified.")
        if object_.data.users != 1:
            raise RuntimeError(
                f"Object {object_.name!r} shares mesh data and was not modified."
            )
    return selected


def main() -> None:
    arguments = _arguments()
    source = arguments.input.resolve()
    options_path = arguments.options.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not options_path.is_file():
        raise FileNotFoundError(options_path)
    options = _load_options(options_path)
    _load_source(source)
    selected = _selected_meshes(options)
    all_meshes = [object_ for object_ in bpy.context.scene.objects if object_.type == "MESH"]
    bounds_minimum, bounds_maximum = _world_bounds(all_meshes)
    bounds_diagonal = (bounds_maximum - bounds_minimum).length

    iterations = int(options.get("iterations", 3))
    strength = float(options.get("strength", 0.35))
    preserve_volume = bool(options.get("preserveVolume", True))
    preserve_boundaries = bool(options.get("preserveBoundaries", True))
    max_displacement_ratio = float(options.get("maxDisplacementRatio", 0.02))
    shade_smooth = bool(options.get("shadeSmooth", True))
    vertex_group = options.get("vertexGroup")
    if vertex_group is not None and not isinstance(vertex_group, str):
        raise ValueError("vertexGroup must be null or a string.")
    if not 1 <= iterations <= 20:
        raise ValueError("iterations must be between 1 and 20.")
    if not 0.01 <= strength <= 1.0:
        raise ValueError("strength must be between 0.01 and 1.0.")
    if not 0.0001 <= max_displacement_ratio <= 0.25:
        raise ValueError("maxDisplacementRatio must be between 0.0001 and 0.25.")

    snapshots: dict[str, dict[str, Any]] = {}
    selected_vertex_count = 0
    for object_ in selected:
        mesh = object_.data
        mesh.update()
        boundary = _boundary_vertices(mesh)
        weights = _weights(object_, vertex_group, boundary, preserve_boundaries)
        selected_vertex_count += sum(weight > 0.0 for weight in weights)
        snapshots[object_.name] = {
            "coordinates": [vertex.co.copy() for vertex in mesh.vertices],
            "normals": [polygon.normal.copy() for polygon in mesh.polygons],
            "topologyHash": _mesh_topology_hash(mesh),
            "uvHash": _mesh_uv_hash(mesh),
            "materialHash": _mesh_material_hash(object_),
            "transform": tuple(float(value) for row in object_.matrix_world for value in row),
            "boundary": boundary,
            "weights": weights,
        }
    if selected_vertex_count == 0:
        if preserve_boundaries:
            raise ValueError(
                "The smoothing selection contains no movable vertices because every selected "
                "vertex is a topological boundary. This is common in seam-split GLB data; use "
                "the source Blend or explicitly disable preserveBoundaries after boundary review."
            )
        raise ValueError("The smoothing selection contains no movable vertices.")

    for object_ in selected:
        mesh = object_.data
        snapshot = snapshots[object_.name]
        coordinates = [coordinate.copy() for coordinate in snapshot["coordinates"]]
        neighbors = _adjacency(mesh)
        weights = snapshot["weights"]
        for _ in range(iterations):
            coordinates = _laplacian_pass(
                coordinates,
                neighbors,
                weights,
                0.5 * strength,
            )
            if preserve_volume:
                coordinates = _laplacian_pass(
                    coordinates,
                    neighbors,
                    weights,
                    -0.53 * strength,
                )
        if any(
            not math.isfinite(component)
            for coordinate in coordinates
            for component in coordinate
        ):
            raise RuntimeError("The smoothing pass produced non-finite coordinates.")
        for vertex, coordinate in zip(mesh.vertices, coordinates, strict=True):
            vertex.co = coordinate
        mesh.update()

    displacement_limit = bounds_diagonal * max_displacement_ratio
    object_reports: list[dict[str, Any]] = []
    all_displacements: list[float] = []
    gate_failures: list[str] = []
    for object_ in selected:
        mesh = object_.data
        mesh.update()
        snapshot = snapshots[object_.name]
        before_coordinates = snapshot["coordinates"]
        matrix = object_.matrix_world.to_3x3()
        displacements = [
            (matrix @ (vertex.co - before)).length
            for vertex, before in zip(mesh.vertices, before_coordinates, strict=True)
        ]
        all_displacements.extend(displacements)
        boundary_maximum = max(
            (displacements[index] for index in snapshot["boundary"]),
            default=0.0,
        )
        outside_maximum = max(
            (
                displacements[index]
                for index, weight in enumerate(snapshot["weights"])
                if weight <= 0.0
            ),
            default=0.0,
        )
        topology_unchanged = _mesh_topology_hash(mesh) == snapshot["topologyHash"]
        uv_unchanged = _mesh_uv_hash(mesh) == snapshot["uvHash"]
        materials_unchanged = _mesh_material_hash(object_) == snapshot["materialHash"]
        transform_unchanged = (
            tuple(float(value) for row in object_.matrix_world for value in row)
            == snapshot["transform"]
        )
        flipped_faces = sum(
            before.dot(polygon.normal) < 0.0
            for before, polygon in zip(snapshot["normals"], mesh.polygons, strict=True)
        )
        degenerate_faces = sum(polygon.area <= 1e-12 for polygon in mesh.polygons)
        if not topology_unchanged:
            gate_failures.append(f"{object_.name}: topology changed")
        if not uv_unchanged:
            gate_failures.append(f"{object_.name}: UV data changed")
        if not materials_unchanged:
            gate_failures.append(f"{object_.name}: materials changed")
        if not transform_unchanged:
            gate_failures.append(f"{object_.name}: transform changed")
        if flipped_faces:
            gate_failures.append(f"{object_.name}: {flipped_faces} faces flipped")
        if degenerate_faces:
            gate_failures.append(f"{object_.name}: {degenerate_faces} faces degenerated")
        if preserve_boundaries and boundary_maximum > 1e-9:
            gate_failures.append(f"{object_.name}: boundary vertices moved")
        if outside_maximum > 1e-9:
            gate_failures.append(f"{object_.name}: protected vertices moved")
        object_reports.append(
            {
                "name": object_.name,
                "vertexCount": len(mesh.vertices),
                "polygonCount": len(mesh.polygons),
                "selectedVertexCount": sum(weight > 0.0 for weight in snapshot["weights"]),
                "topologyHashBefore": snapshot["topologyHash"],
                "topologyHashAfter": _mesh_topology_hash(mesh),
                "topologyUnchanged": topology_unchanged,
                "uvUnchanged": uv_unchanged,
                "materialsUnchanged": materials_unchanged,
                "transformUnchanged": transform_unchanged,
                "boundaryMaximumDisplacementMeters": boundary_maximum,
                "outsideSelectionMaximumDisplacementMeters": outside_maximum,
                "maximumDisplacementMeters": max(displacements, default=0.0),
                "flippedFaceCount": flipped_faces,
                "degenerateFaceCount": degenerate_faces,
            }
        )

    maximum_displacement = max(all_displacements, default=0.0)
    if maximum_displacement > displacement_limit:
        gate_failures.append(
            "maximum displacement exceeded the configured model-diagonal budget"
        )
    if gate_failures:
        for object_ in selected:
            for vertex, coordinate in zip(
                object_.data.vertices,
                snapshots[object_.name]["coordinates"],
                strict=True,
            ):
                vertex.co = coordinate
            object_.data.update()
        raise RuntimeError("Smoothing QA failed: " + "; ".join(gate_failures))

    if shade_smooth:
        for object_ in selected:
            for polygon in object_.data.polygons:
                polygon.use_smooth = True

    for path in (arguments.output_blend, arguments.output_glb, arguments.qa):
        path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene["viewforgeSurfaceSmooth"] = True
    bpy.context.scene["viewforgeSurfaceSmoothSourceSha256"] = _sha256(source)
    bpy.ops.wm.save_as_mainfile(filepath=str(arguments.output_blend.resolve()), compress=True)
    bpy.ops.export_scene.gltf(
        filepath=str(arguments.output_glb.resolve()),
        export_format="GLB",
        export_cameras=False,
        export_lights=False,
        export_apply=False,
    )

    sorted_displacements = sorted(all_displacements)
    percentile_index = max(0, math.ceil(len(sorted_displacements) * 0.99) - 1)
    qa = {
        "schemaVersion": 1,
        "route": "topology-preserving-surface-smooth-v1",
        "status": "pendingUserSignoff",
        "source": {
            "name": source.name,
            "sha256": _sha256(source),
            "modified": False,
        },
        "operation": {
            "method": "taubin" if preserve_volume else "laplacian",
            "iterations": iterations,
            "strength": strength,
            "preserveVolume": preserve_volume,
            "preserveBoundaries": preserve_boundaries,
            "vertexGroup": vertex_group,
            "shadeSmooth": shade_smooth,
            "maxDisplacementRatio": max_displacement_ratio,
        },
        "boundsDiagonalMeters": bounds_diagonal,
        "displacementBudgetMeters": displacement_limit,
        "selectedVertexCount": selected_vertex_count,
        "maximumDisplacementMeters": maximum_displacement,
        "meanDisplacementMeters": (
            sum(all_displacements) / len(all_displacements) if all_displacements else 0.0
        ),
        "p99DisplacementMeters": (
            sorted_displacements[percentile_index] if sorted_displacements else 0.0
        ),
        "objects": object_reports,
        "gates": {
            "finiteCoordinates": True,
            "topologyUnchanged": True,
            "uvsUnchanged": True,
            "materialsUnchanged": True,
            "transformsUnchanged": True,
            "protectedVerticesLocked": True,
            "facesNotFlipped": True,
            "facesNotDegenerate": True,
            "displacementBudgetPassed": True,
        },
        "outputs": {
            "blendSha256": _sha256(arguments.output_blend),
            "glbSha256": _sha256(arguments.output_glb),
        },
    }
    arguments.qa.write_text(
        json.dumps(qa, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
