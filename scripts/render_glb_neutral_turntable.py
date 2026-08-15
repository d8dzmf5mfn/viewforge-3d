"""Render an immutable GLB under a neutral matte material at fixed yaw views.

Run with Blender:

    Blender --background --python scripts/render_glb_neutral_turntable.py -- \
      --model /absolute/model.glb --output /absolute/output

The imported scene is modified only in memory. The source GLB is never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--roughness", type=float, default=0.82)
    parser.add_argument("--specular-ior-level", type=float, default=0.28)
    parser.add_argument("--ior", type=float, default=1.45)
    parser.add_argument("--coat-weight", type=float, default=0.0)
    parser.add_argument("--coat-roughness", type=float, default=0.2)
    parser.add_argument("--subsurface-weight", type=float, default=0.0)
    parser.add_argument("--base-color", default="0.72,0.72,0.72")
    parser.add_argument(
        "--views",
        default="left90:-90,left45:-45,front:0,right45:45,right90:90",
    )
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def point_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def add_area_light(
    name: str,
    location: Vector,
    target: Vector,
    *,
    energy: float,
    size: float,
) -> None:
    data = bpy.data.lights.new(name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    point_at(obj, target)


def make_neutral_material(
    base_color: tuple[float, float, float],
    roughness: float,
    specular_ior_level: float,
    ior: float,
    coat_weight: float,
    coat_roughness: float,
    subsurface_weight: float,
) -> bpy.types.Material:
    material = bpy.data.materials.new("V24NeutralComparison")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = (*base_color, 1.0)
    shader.inputs["Metallic"].default_value = 0.0
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["IOR"].default_value = ior
    specular_input = shader.inputs.get("Specular IOR Level") or shader.inputs.get("IOR Level")
    if specular_input is None:
        raise RuntimeError("Principled BSDF exposes no specular IOR level input")
    specular_input.default_value = specular_ior_level
    shader.inputs["Coat Weight"].default_value = coat_weight
    shader.inputs["Coat Roughness"].default_value = coat_roughness
    shader.inputs["Subsurface Weight"].default_value = subsurface_weight
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def main() -> None:
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    base_color_values = tuple(float(value) for value in args.base_color.split(","))
    if len(base_color_values) != 3:
        raise ValueError("--base-color must contain exactly three comma-separated values")
    views = []
    for item in args.views.split(","):
        name, yaw = item.split(":", 1)
        views.append((name.strip(), float(yaw)))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.model.resolve()))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("imported GLB contains no mesh objects")

    material = make_neutral_material(
        base_color_values,
        args.roughness,
        args.specular_ior_level,
        args.ior,
        args.coat_weight,
        args.coat_roughness,
        args.subsurface_weight,
    )
    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(material)

    points = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    center = (minimum + maximum) * 0.5
    dimensions = maximum - minimum
    span = max(dimensions)

    camera_data = bpy.data.cameras.new("ComparisonCamera")
    camera = bpy.data.objects.new("ComparisonCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = span * 1.14
    camera_data.dof.use_dof = False

    add_area_light(
        "Key",
        center + Vector((-2.2 * span, 2.8 * span, 2.5 * span)),
        center,
        energy=12000.0,
        size=2.4 * span,
    )
    add_area_light(
        "Fill",
        center + Vector((2.4 * span, 1.8 * span, 0.7 * span)),
        center,
        energy=6500.0,
        size=2.8 * span,
    )
    add_area_light(
        "Rim",
        center + Vector((0.0, -2.4 * span, 2.0 * span)),
        center,
        energy=4200.0,
        size=2.0 * span,
    )

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = args.size
    scene.render.resolution_y = args.size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.view_settings.look = "AgX - Medium High Contrast"
    world = bpy.data.worlds.new("ComparisonWorld")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.002, 0.002, 0.002, 1.0)
    background.inputs["Strength"].default_value = 0.05
    scene.world = world

    distance = span * 3.0
    renders: dict[str, dict[str, object]] = {}
    for name, yaw_degrees in views:
        yaw = math.radians(yaw_degrees)
        camera.location = center + Vector((distance * math.sin(yaw), distance * math.cos(yaw), 0.0))
        point_at(camera, center)
        destination = args.output / f"{name}.png"
        scene.render.filepath = str(destination.resolve())
        bpy.ops.render.render(write_still=True)
        renders[name] = {
            "yawDegrees": yaw_degrees,
            "path": destination.name,
            "sha256": sha256_file(destination),
        }

    metrics = {
        "schemaVersion": 1,
        "purpose": "neutral-material geometry and reflection comparison",
        "source": {
            "path": str(args.model.resolve()),
            "sha256": sha256_file(args.model),
            "modified": False,
        },
        "scene": {
            "engine": scene.render.engine,
            "camera": "orthographic",
            "size": args.size,
            "boundsMin": list(minimum),
            "boundsMax": list(maximum),
            "orthoScale": camera_data.ortho_scale,
        },
        "material": {
            "baseColor": list(base_color_values),
            "metallic": 0.0,
            "roughness": args.roughness,
            "ior": args.ior,
            "specularIorLevel": args.specular_ior_level,
            "coatWeight": args.coat_weight,
            "coatRoughness": args.coat_roughness,
            "subsurfaceWeight": args.subsurface_weight,
        },
        "renders": renders,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
