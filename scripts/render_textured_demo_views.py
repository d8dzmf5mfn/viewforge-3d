"""Render licensed Lee Perry-Smith assets into calibrated 2D-only test inputs.

Run with Blender:
  Blender --background --python scripts/render_textured_demo_views.py -- \
    --model .local/demo-assets/LeePerrySmith.glb \
    --albedo .local/demo-assets/Map-COL.jpg \
    --normal .local/demo-assets/Infinite-Level_02_Tangent_SmoothUV.jpg \
    --output assets/demo/lee-textured-v1
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--albedo", type=Path, required=True)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1280)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def point_camera(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def make_material(albedo_path: Path, normal_path: Path) -> bpy.types.Material:
    material = bpy.data.materials.new("LeePerrySmithPhotoMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Roughness"].default_value = 0.48
    shader.inputs["IOR"].default_value = 1.4
    albedo = nodes.new("ShaderNodeTexImage")
    albedo.image = bpy.data.images.load(str(albedo_path.resolve()))
    albedo.image.colorspace_settings.name = "sRGB"
    normal = nodes.new("ShaderNodeTexImage")
    normal.image = bpy.data.images.load(str(normal_path.resolve()))
    normal.image.colorspace_settings.name = "Non-Color"
    normal_map = nodes.new("ShaderNodeNormalMap")
    normal_map.inputs["Strength"].default_value = 0.12

    links.new(albedo.outputs["Color"], shader.inputs["Base Color"])
    links.new(normal.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def add_area_light(
    name: str,
    location: tuple[float, float, float],
    energy: float,
    size: float,
) -> None:
    light_data = bpy.data.lights.new(name, type="AREA")
    light_data.energy = energy
    light_data.shape = "DISK"
    light_data.size = size
    light = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light)
    light.location = location
    point_camera(light, Vector((0.0, 0.0, 0.0)))


def main() -> None:
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.model.resolve()))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("imported GLB contains no mesh")
    material = make_material(args.albedo, args.normal)
    for mesh in meshes:
        mesh.data.materials.clear()
        mesh.data.materials.append(material)

    bounds = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    minimum = Vector(tuple(min(point[axis] for point in bounds) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in bounds) for axis in range(3)))
    center = (minimum + maximum) * 0.5
    root = bpy.data.objects.new("SubjectRoot", None)
    bpy.context.collection.objects.link(root)
    for mesh in meshes:
        mesh.parent = root
    root.location = -center

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera_data.lens = 78.0
    camera_data.sensor_width = 36.0
    camera_data.dof.use_dof = False

    add_area_light("Key", (-7.0, -12.0, 8.0), 820.0, 7.0)
    add_area_light("Fill", (8.0, -10.0, 4.0), 560.0, 6.0)
    add_area_light("Top", (0.0, 0.0, 12.0), 360.0, 5.0)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = args.size
    scene.render.resolution_y = args.size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.image_settings.color_depth = "8"
    scene.world = bpy.data.worlds.new("StudioWorld")
    scene.world.color = (0.22, 0.22, 0.22)
    scene.view_settings.look = "AgX - Medium High Contrast"

    distance = 22.0
    target = Vector((0.0, 0.0, 0.0))
    views = (("front", 0.0), ("left45", -45.0), ("right45", 45.0))
    for role, yaw_degrees in views:
        yaw = math.radians(yaw_degrees)
        camera.location = (distance * math.sin(yaw), -distance * math.cos(yaw), 0.0)
        point_camera(camera, target)
        scene.render.filepath = str((args.output / f"{role}.png").resolve())
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
