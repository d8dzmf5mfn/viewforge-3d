#!/usr/bin/env python3
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

VIEWS = (
    ("left90", -90.0),
    ("left45", -45.0),
    ("front", 0.0),
    ("right45", 45.0),
    ("right90", 90.0),
    ("back", 180.0),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render fixed bone-overlay QA views.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--size", type=int, default=640)
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
    name: str, location: Vector, target: Vector, *, energy: float, size: float
) -> None:
    data = bpy.data.lights.new(name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    point_at(obj, target)


def principled_material(
    name: str,
    base_color: tuple[float, float, float, float],
    *,
    alpha: float = 1.0,
    emission_strength: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*base_color[:3], alpha)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = (*base_color[:3], alpha)
    shader.inputs["Metallic"].default_value = 0.0
    shader.inputs["Roughness"].default_value = 0.36
    shader.inputs["Alpha"].default_value = alpha
    if emission_strength > 0.0:
        emission_color = shader.inputs.get("Emission Color") or shader.inputs.get("Emission")
        if emission_color:
            emission_color.default_value = (*base_color[:3], 1.0)
        strength = shader.inputs.get("Emission Strength")
        if strength:
            strength.default_value = emission_strength
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    if alpha < 1.0:
        material.surface_render_method = "BLENDED"
        material.show_transparent_back = True
        material.use_transparency_overlap = False
    return material


def main() -> None:
    arguments = parse_arguments()
    output_dir = arguments.output_dir.expanduser().resolve()
    source_layer_dir = output_dir / "source"
    bone_layer_dir = output_dir / "bones"
    source_layer_dir.mkdir(parents=True, exist_ok=True)
    bone_layer_dir.mkdir(parents=True, exist_ok=True)
    collisions = [source_layer_dir / f"{name}.png" for name, _ in VIEWS]
    collisions.extend(bone_layer_dir / f"{name}.png" for name, _ in VIEWS)
    collisions.append(output_dir / "metrics.json")
    existing = [str(path) for path in collisions if path.exists()]
    if existing:
        raise FileExistsError(f"immutable QA render outputs already exist: {existing}")

    source_meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and not obj.name.startswith("VF_")
    ]
    bone_meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.get("vf_preview_kind") == "bone"
    ]
    joint_meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.get("vf_preview_kind") == "joint"
    ]
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if not source_meshes or not bone_meshes or not joint_meshes or len(armatures) != 1:
        raise RuntimeError("Blend file does not contain the expected source, preview, and Armature")

    source_material = principled_material("VF_QA_Source", (0.03, 0.28, 0.25, 1.0))
    bone_material = principled_material(
        "VF_QA_Bones", (1.0, 0.34, 0.01, 1.0), emission_strength=3.0
    )
    joint_material = principled_material(
        "VF_QA_Joints", (0.0, 0.86, 1.0, 1.0), emission_strength=3.0
    )
    for obj in source_meshes:
        obj.data.materials.clear()
        obj.data.materials.append(source_material)
    for obj in bone_meshes:
        obj.data.materials.clear()
        obj.data.materials.append(bone_material)
    for obj in joint_meshes:
        obj.data.materials.clear()
        obj.data.materials.append(joint_material)
    armatures[0].hide_render = True

    points = [
        obj.matrix_world @ Vector(corner)
        for obj in source_meshes
        for corner in obj.bound_box
    ]
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    center = (minimum + maximum) * 0.5
    dimensions = maximum - minimum
    span = max(dimensions)

    camera_data = bpy.data.cameras.new("VF_QA_Camera")
    camera = bpy.data.objects.new("VF_QA_Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = span * 1.14
    camera_data.dof.use_dof = False

    add_area_light(
        "VF_QA_Key",
        center + Vector((-2.2 * span, 2.8 * span, 2.5 * span)),
        center,
        energy=9000.0,
        size=2.4 * span,
    )
    add_area_light(
        "VF_QA_Fill",
        center + Vector((2.4 * span, 1.8 * span, 0.8 * span)),
        center,
        energy=5200.0,
        size=2.8 * span,
    )

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = arguments.size
    scene.render.resolution_y = arguments.size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.view_settings.look = "AgX - Medium High Contrast"
    world = bpy.data.worlds.new("VF_QA_World")
    world.use_nodes = True
    world.node_tree.nodes.clear()
    background = world.node_tree.nodes.new("ShaderNodeBackground")
    world_output = world.node_tree.nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(background.outputs["Background"], world_output.inputs["Surface"])
    background.inputs["Color"].default_value = (0.006, 0.009, 0.014, 1.0)
    background.inputs["Strength"].default_value = 0.08
    scene.world = world

    distance = span * 3.0
    renders: dict[str, dict[str, Any]] = {}
    for name, yaw_degrees in VIEWS:
        yaw = math.radians(yaw_degrees)
        camera.location = center + Vector(
            (distance * math.sin(yaw), distance * math.cos(yaw), 0.0)
        )
        point_at(camera, center)
        for obj in source_meshes:
            obj.hide_render = False
        for obj in [*bone_meshes, *joint_meshes]:
            obj.hide_render = True
        scene.render.film_transparent = False
        source_destination = source_layer_dir / f"{name}.png"
        scene.render.filepath = str(source_destination)
        bpy.ops.render.render(write_still=True)

        for obj in source_meshes:
            obj.hide_render = True
        for obj in [*bone_meshes, *joint_meshes]:
            obj.hide_render = False
        scene.render.film_transparent = True
        bone_destination = bone_layer_dir / f"{name}.png"
        scene.render.filepath = str(bone_destination)
        bpy.ops.render.render(write_still=True)
        renders[name] = {
            "yawDegrees": yaw_degrees,
            "sourceLayer": {
                "path": str(source_destination.relative_to(output_dir)),
                "sha256": sha256_file(source_destination),
            },
            "boneLayer": {
                "path": str(bone_destination.relative_to(output_dir)),
                "sha256": sha256_file(bone_destination),
            },
        }

    metrics = {
        "schemaVersion": 1,
        "purpose": "fixed-view bone-only Armature placement review",
        "sourceBlend": {
            "path": bpy.data.filepath,
            "sha256": sha256_file(Path(bpy.data.filepath)),
            "modifiedOnDisk": False,
        },
        "scene": {
            "engine": scene.render.engine,
            "camera": "orthographic",
            "size": arguments.size,
            "boundsMin": list(minimum),
            "boundsMax": list(maximum),
            "orthoScale": camera_data.ortho_scale,
        },
        "display": {
            "composition": "opaque source layer plus transparent bone layer",
            "sourceIsReferenceOnly": True,
            "bonePreviewMeshes": len(bone_meshes),
            "jointPreviewMeshes": len(joint_meshes),
        },
        "renders": renders,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
