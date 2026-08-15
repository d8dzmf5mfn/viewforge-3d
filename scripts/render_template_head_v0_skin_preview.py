"""Render the prepared TemplateHeadV0 skin preview from Blender's GUI scripting workspace.

The Boolean preparation path can crash in Blender background mode on some macOS builds. This
render script is intentionally GUI-compatible and writes only preview images; it never edits the
source GLB.
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1024)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def point_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def add_area_light(
    name: str,
    location: tuple[float, float, float],
    energy: float,
    size: float,
) -> None:
    data = bpy.data.lights.new(name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = location
    point_at(light, Vector((0.0, 0.0, 0.0)))


def main() -> None:
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.model.resolve()))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("prepared preview GLB contains no mesh")

    bounds = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    minimum = Vector(tuple(min(point[axis] for point in bounds) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in bounds) for axis in range(3)))
    center = (minimum + maximum) * 0.5
    root = bpy.data.objects.new("TemplateHeadV0.PreviewRoot", None)
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

    add_area_light("Key", (-7.0, -12.0, 8.0), 780.0, 7.0)
    add_area_light("Fill", (8.0, -10.0, 4.0), 500.0, 6.0)
    add_area_light("Top", (0.0, 0.0, 12.0), 320.0, 5.0)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = args.size
    scene.render.resolution_y = args.size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("StudioWorld")
    scene.world.color = (0.018, 0.022, 0.026)
    scene.view_settings.look = "AgX - Medium High Contrast"

    distance = 22.0
    target = Vector((0.0, 0.0, 0.0))
    for role, yaw_degrees in (
        ("front", 0.0),
        ("left45", -45.0),
        ("right45", 45.0),
        ("side", -90.0),
    ):
        yaw = math.radians(yaw_degrees)
        camera.location = (
            distance * math.sin(yaw),
            -distance * math.cos(yaw),
            0.0,
        )
        point_at(camera, target)
        scene.render.filepath = str((args.output / f"skin-{role}.png").resolve())
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
