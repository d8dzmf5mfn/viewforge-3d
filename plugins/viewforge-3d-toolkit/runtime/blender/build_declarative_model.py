from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector

MAX_OBJECTS = 256
MAX_VERTICES = 100_000
MAX_FACES = 200_000


def _arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-blend", type=Path, required=True)
    parser.add_argument("--output-glb", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    return parser.parse_args(arguments)


def _number(value: Any, name: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    measured = float(value)
    if not math.isfinite(measured) or not minimum <= measured <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return measured


def _integer(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _vector(
    value: Any,
    name: str,
    *,
    default: tuple[float, float, float],
    minimum: float,
    maximum: float,
) -> tuple[float, float, float]:
    if value is None:
        return default
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain three numbers")
    return tuple(
        _number(component, f"{name}[{index}]", minimum=minimum, maximum=maximum)
        for index, component in enumerate(value)
    )


def _load_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schemaVersion") != 1:
        raise ValueError("declarative model spec must use schemaVersion 1")
    allowed_root = {"schemaVersion", "name", "units", "objects", "world"}
    unknown = sorted(set(payload) - allowed_root)
    if unknown:
        raise ValueError(f"unknown root fields: {unknown}")
    if payload.get("units", "meters") != "meters":
        raise ValueError("declarative model units must be meters")
    objects = payload.get("objects")
    if not isinstance(objects, list) or not 1 <= len(objects) <= MAX_OBJECTS:
        raise ValueError(f"objects must contain 1..{MAX_OBJECTS} entries")
    return payload


def _material(definition: Any, object_name: str) -> bpy.types.Material | None:
    if definition is None:
        return None
    if not isinstance(definition, dict):
        raise ValueError(f"{object_name}.material must be an object")
    unknown = sorted(set(definition) - {"color", "metallic", "roughness"})
    if unknown:
        raise ValueError(f"{object_name}.material contains unknown fields: {unknown}")
    color_value = definition.get("color", [0.7, 0.7, 0.7, 1.0])
    if not isinstance(color_value, list) or len(color_value) != 4:
        raise ValueError(f"{object_name}.material.color must contain RGBA")
    color = tuple(
        _number(value, f"{object_name}.material.color", minimum=0, maximum=1)
        for value in color_value
    )
    material = bpy.data.materials.new(name=f"{object_name}_Material")
    material.diffuse_color = color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Metallic"].default_value = _number(
            definition.get("metallic", 0.0),
            f"{object_name}.material.metallic",
            minimum=0,
            maximum=1,
        )
        principled.inputs["Roughness"].default_value = _number(
            definition.get("roughness", 0.5),
            f"{object_name}.material.roughness",
            minimum=0,
            maximum=1,
        )
    return material


def _explicit_mesh(definition: dict[str, Any], name: str) -> bpy.types.Object:
    vertices = definition.get("vertices")
    faces = definition.get("faces")
    if not isinstance(vertices, list) or not 3 <= len(vertices) <= MAX_VERTICES:
        raise ValueError(f"{name}.vertices must contain 3..{MAX_VERTICES} points")
    if not isinstance(faces, list) or not 1 <= len(faces) <= MAX_FACES:
        raise ValueError(f"{name}.faces must contain 1..{MAX_FACES} polygons")
    parsed_vertices = [
        _vector(
            value,
            f"{name}.vertices[{index}]",
            default=(0, 0, 0),
            minimum=-10_000,
            maximum=10_000,
        )
        for index, value in enumerate(vertices)
    ]
    parsed_faces: list[tuple[int, ...]] = []
    for index, face in enumerate(faces):
        if not isinstance(face, list) or not 3 <= len(face) <= 64:
            raise ValueError(f"{name}.faces[{index}] must contain 3..64 indices")
        parsed = tuple(
            _integer(value, f"{name}.faces[{index}]", minimum=0, maximum=len(vertices) - 1)
            for value in face
        )
        if len(set(parsed)) != len(parsed):
            raise ValueError(f"{name}.faces[{index}] contains duplicate indices")
        parsed_faces.append(parsed)
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(parsed_vertices, [], parsed_faces)
    mesh.validate(verbose=False)
    mesh.update()
    object_ = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(object_)
    return object_


def _primitive(definition: dict[str, Any], name: str) -> bpy.types.Object:
    primitive = definition.get("primitive")
    if primitive == "mesh":
        return _explicit_mesh(definition, name)
    parameters = definition.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError(f"{name}.parameters must be an object")
    if primitive == "cube":
        bpy.ops.mesh.primitive_cube_add(
            size=_number(parameters.get("size", 1.0), f"{name}.size", minimum=0.0001, maximum=1000)
        )
    elif primitive == "uv_sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=_integer(
                parameters.get("segments", 32), f"{name}.segments", minimum=3, maximum=256
            ),
            ring_count=_integer(
                parameters.get("rings", 16), f"{name}.rings", minimum=3, maximum=128
            ),
            radius=_number(
                parameters.get("radius", 0.5), f"{name}.radius", minimum=0.0001, maximum=1000
            ),
        )
    elif primitive == "ico_sphere":
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=_integer(
                parameters.get("subdivisions", 2), f"{name}.subdivisions", minimum=1, maximum=6
            ),
            radius=_number(
                parameters.get("radius", 0.5), f"{name}.radius", minimum=0.0001, maximum=1000
            ),
        )
    elif primitive == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=_integer(
                parameters.get("vertices", 32), f"{name}.vertices", minimum=3, maximum=256
            ),
            radius=_number(
                parameters.get("radius", 0.5), f"{name}.radius", minimum=0.0001, maximum=1000
            ),
            depth=_number(
                parameters.get("depth", 1.0), f"{name}.depth", minimum=0.0001, maximum=1000
            ),
        )
    elif primitive == "cone":
        bpy.ops.mesh.primitive_cone_add(
            vertices=_integer(
                parameters.get("vertices", 32), f"{name}.vertices", minimum=3, maximum=256
            ),
            radius1=_number(
                parameters.get("radius1", 0.5), f"{name}.radius1", minimum=0, maximum=1000
            ),
            radius2=_number(
                parameters.get("radius2", 0.0), f"{name}.radius2", minimum=0, maximum=1000
            ),
            depth=_number(
                parameters.get("depth", 1.0), f"{name}.depth", minimum=0.0001, maximum=1000
            ),
        )
    elif primitive == "torus":
        bpy.ops.mesh.primitive_torus_add(
            major_segments=_integer(
                parameters.get("majorSegments", 48), f"{name}.majorSegments", minimum=3, maximum=256
            ),
            minor_segments=_integer(
                parameters.get("minorSegments", 16), f"{name}.minorSegments", minimum=3, maximum=128
            ),
            major_radius=_number(
                parameters.get("majorRadius", 0.5),
                f"{name}.majorRadius",
                minimum=0.0001,
                maximum=1000,
            ),
            minor_radius=_number(
                parameters.get("minorRadius", 0.1),
                f"{name}.minorRadius",
                minimum=0.0001,
                maximum=1000,
            ),
        )
    else:
        raise ValueError(
            f"{name}.primitive must be cube, uv_sphere, ico_sphere, cylinder, cone, torus, or mesh"
        )
    object_ = bpy.context.object
    object_.name = name
    return object_


def _build_object(definition: Any, index: int) -> bpy.types.Object:
    if not isinstance(definition, dict):
        raise ValueError(f"objects[{index}] must be an object")
    allowed = {
        "name",
        "primitive",
        "parameters",
        "vertices",
        "faces",
        "location",
        "rotationDegrees",
        "scale",
        "material",
        "bevel",
        "shadeSmooth",
    }
    unknown = sorted(set(definition) - allowed)
    if unknown:
        raise ValueError(f"objects[{index}] contains unknown fields: {unknown}")
    name = definition.get("name", f"Object_{index + 1:03d}")
    if (
        not isinstance(name, str)
        or not name
        or len(name) > 80
        or any(ord(char) < 32 for char in name)
    ):
        raise ValueError(f"objects[{index}].name is invalid")
    object_ = _primitive(definition, name)
    object_.location = _vector(
        definition.get("location"),
        f"{name}.location",
        default=(0, 0, 0),
        minimum=-10_000,
        maximum=10_000,
    )
    rotation = _vector(
        definition.get("rotationDegrees"),
        f"{name}.rotationDegrees",
        default=(0, 0, 0),
        minimum=-360_000,
        maximum=360_000,
    )
    object_.rotation_euler = tuple(math.radians(value) for value in rotation)
    object_.scale = _vector(
        definition.get("scale"),
        f"{name}.scale",
        default=(1, 1, 1),
        minimum=0.000001,
        maximum=10_000,
    )
    material = _material(definition.get("material"), name)
    if material is not None:
        object_.data.materials.append(material)
    bevel = definition.get("bevel")
    if bevel is not None:
        if not isinstance(bevel, dict) or sorted(set(bevel) - {"width", "segments"}):
            raise ValueError(f"{name}.bevel accepts only width and segments")
        modifier = object_.modifiers.new(name="ViewForge Bevel", type="BEVEL")
        modifier.width = _number(
            bevel.get("width", 0.01), f"{name}.bevel.width", minimum=0, maximum=100
        )
        modifier.segments = _integer(
            bevel.get("segments", 2), f"{name}.bevel.segments", minimum=1, maximum=16
        )
        bpy.context.view_layer.objects.active = object_
        object_.select_set(True)
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    if definition.get("shadeSmooth", False):
        for polygon in object_.data.polygons:
            polygon.use_smooth = True
    return object_


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounds(object_: bpy.types.Object) -> list[list[float]]:
    corners = [object_.matrix_world @ Vector(corner) for corner in object_.bound_box]
    return [
        [float(min(point[axis] for point in corners)) for axis in range(3)],
        [float(max(point[axis] for point in corners)) for axis in range(3)],
    ]


def main() -> None:
    arguments = _arguments()
    spec = _load_spec(arguments.spec.resolve())
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)

    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    world = spec.get("world", {})
    if not isinstance(world, dict) or sorted(set(world) - {"backgroundColor"}):
        raise ValueError("world accepts only backgroundColor")
    background = world.get("backgroundColor", [0.05, 0.05, 0.05, 1.0])
    if not isinstance(background, list) or len(background) != 4:
        raise ValueError("world.backgroundColor must contain RGBA")
    scene.world.color = tuple(
        _number(value, "world.backgroundColor", minimum=0, maximum=1) for value in background[:3]
    )

    objects = [_build_object(definition, index) for index, definition in enumerate(spec["objects"])]
    scene["viewforgeSchemaVersion"] = 1
    scene["viewforgeDeclarativeModel"] = True
    scene["viewforgeSourceSpecSha256"] = _sha256(arguments.spec)

    for path in (arguments.output_blend, arguments.output_glb, arguments.qa):
        path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(arguments.output_blend.resolve()))
    bpy.ops.export_scene.gltf(
        filepath=str(arguments.output_glb.resolve()),
        export_format="GLB",
        export_cameras=False,
        export_lights=False,
        export_apply=True,
    )
    qa = {
        "schemaVersion": 1,
        "route": "declarative-blender-model-v1",
        "arbitraryCodeExecution": False,
        "objectCount": len(objects),
        "vertexCount": sum(len(object_.data.vertices) for object_ in objects),
        "triangleCount": sum(len(object_.data.loop_triangles) for object_ in objects),
        "objects": [
            {
                "name": object_.name,
                "vertexCount": len(object_.data.vertices),
                "polygonCount": len(object_.data.polygons),
                "boundsMeters": _bounds(object_),
            }
            for object_ in objects
        ],
        "specSha256": _sha256(arguments.spec),
        "blendSha256": _sha256(arguments.output_blend),
        "glbSha256": _sha256(arguments.output_glb),
    }
    arguments.qa.write_text(
        json.dumps(qa, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
