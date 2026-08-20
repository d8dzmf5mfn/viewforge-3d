from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector

VIEW_DIRECTIONS = {
    "perspective": (1.6, 1.6, 1.15),
    "front": (0.0, 1.0, 0.0),
    "back": (0.0, -1.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "top": (0.0, 0.0, 1.0),
    "bottom": (0.0, 0.0, -1.0),
}
MATERIAL_MODES = {"original", "neutral"}
BACKGROUNDS = {"studio_dark", "studio_light", "transparent"}


def _arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--material-mode", required=True)
    parser.add_argument("--background", required=True)
    return parser.parse_args(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source(source: Path) -> None:
    suffix = source.suffix.lower()
    if suffix == ".blend":
        current = Path(bpy.data.filepath).resolve() if bpy.data.filepath else None
        if current != source:
            raise RuntimeError("Blender did not load the staged Blend source.")
        return
    if suffix == ".glb":
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(source))
        return
    raise ValueError("Render input must be a Blend or GLB file.")


def _remove_scene_rig() -> None:
    for object_ in list(bpy.data.objects):
        if object_.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(object_, do_unlink=True)


def _neutral_material() -> bpy.types.Material:
    material = bpy.data.materials.new("ViewForge_NeutralPreview")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is None:
        raise RuntimeError("The neutral preview material could not be created.")
    principled.inputs["Base Color"].default_value = (0.62, 0.66, 0.72, 1.0)
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 0.62
    return material


def _renderable_meshes() -> list[bpy.types.Object]:
    meshes = [
        object_
        for object_ in bpy.context.scene.objects
        if object_.type == "MESH" and not object_.hide_render
    ]
    if not meshes:
        raise RuntimeError("The source contains no renderable mesh objects.")
    return meshes


def _bounds(meshes: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points: list[Vector] = []
    for object_ in meshes:
        evaluated = object_.evaluated_get(depsgraph)
        points.extend(evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box)
    if not points:
        raise RuntimeError("The source mesh bounds are unavailable.")
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    if not all(math.isfinite(value) for value in (*minimum, *maximum)):
        raise RuntimeError("The source mesh bounds are not finite.")
    if max(maximum - minimum) <= 1e-7:
        raise RuntimeError("The source mesh bounds are degenerate.")
    return minimum, maximum


def _point_at(object_: bpy.types.Object, target: Vector) -> None:
    object_.rotation_euler = (target - object_.location).to_track_quat("-Z", "Y").to_euler()


def _area_light(name: str, energy: float, size: float) -> bpy.types.Object:
    data = bpy.data.lights.new(name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    object_ = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(object_)
    return object_


def _set_rig_for_view(
    direction: Vector,
    center: Vector,
    span: float,
    lights: tuple[bpy.types.Object, bpy.types.Object, bpy.types.Object],
) -> None:
    reference_up = Vector((0.0, 0.0, 1.0))
    if abs(direction.dot(reference_up)) > 0.95:
        reference_up = Vector((0.0, 1.0, 0.0))
    right = reference_up.cross(direction).normalized()
    up = direction.cross(right).normalized()
    key, fill, rim = lights
    key.location = center + direction * span * 2.0 - right * span * 1.3 + up * span * 1.4
    fill.location = center + direction * span * 1.4 + right * span * 1.5 + up * span * 0.25
    rim.location = center - direction * span * 1.5 + up * span * 1.2
    for light in lights:
        _point_at(light, center)


def _configure_scene(
    resolution: int,
    background_name: str,
) -> tuple[bpy.types.Scene, bpy.types.Object]:
    scene = bpy.context.scene
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    else:
        raise RuntimeError("No supported Blender Eevee render engine is available.")

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.color_mode = (
        "RGBA" if background_name == "transparent" else "RGB"
    )
    scene.render.film_transparent = background_name == "transparent"
    with contextlib.suppress(TypeError):
        scene.view_settings.look = "AgX - Medium High Contrast"

    world = bpy.data.worlds.new("ViewForge_PreviewWorld")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is None:
        raise RuntimeError("The preview world background is unavailable.")
    if background_name == "studio_light":
        background.inputs["Color"].default_value = (0.68, 0.7, 0.74, 1.0)
        background.inputs["Strength"].default_value = 0.8
    else:
        background.inputs["Color"].default_value = (0.012, 0.014, 0.02, 1.0)
        background.inputs["Strength"].default_value = 0.2
    scene.world = world

    camera_data = bpy.data.cameras.new("ViewForge_PreviewCamera")
    camera = bpy.data.objects.new("ViewForge_PreviewCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    return scene, camera


def _validated_views(raw: str) -> list[str]:
    views = [value.strip() for value in raw.split(",") if value.strip()]
    if not 1 <= len(views) <= len(VIEW_DIRECTIONS):
        raise ValueError("Render views must contain between one and seven values.")
    if len(set(views)) != len(views):
        raise ValueError("Render views must be unique.")
    unknown = sorted(set(views) - set(VIEW_DIRECTIONS))
    if unknown:
        raise ValueError(f"Unsupported render views: {unknown}")
    return views


def main() -> None:
    arguments = _arguments()
    source = arguments.input.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    views = _validated_views(arguments.views)
    if not 256 <= arguments.resolution <= 1024:
        raise ValueError("Render resolution must be between 256 and 1024.")
    if arguments.material_mode not in MATERIAL_MODES:
        raise ValueError("Unsupported material mode.")
    if arguments.background not in BACKGROUNDS:
        raise ValueError("Unsupported background.")

    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _load_source(source)
    _remove_scene_rig()
    meshes = _renderable_meshes()
    if arguments.material_mode == "neutral":
        material = _neutral_material()
        for object_ in meshes:
            object_.data.materials.clear()
            object_.data.materials.append(material)

    minimum, maximum = _bounds(meshes)
    center = (minimum + maximum) * 0.5
    dimensions = maximum - minimum
    span = max(dimensions)
    radius = max(dimensions.length * 0.5, span * 0.5)
    scene, camera = _configure_scene(arguments.resolution, arguments.background)
    camera_data = camera.data
    camera_data.clip_start = max(span * 0.0001, 0.00001)
    camera_data.clip_end = max(span * 100.0, 1000.0)
    lights = (
        _area_light("ViewForge_Key", 1100.0, span * 1.7),
        _area_light("ViewForge_Fill", 650.0, span * 2.1),
        _area_light("ViewForge_Rim", 900.0, span * 1.5),
    )

    rendered: dict[str, dict[str, Any]] = {}
    for view in views:
        direction = Vector(VIEW_DIRECTIONS[view]).normalized()
        if view == "perspective":
            camera_data.type = "PERSP"
            camera_data.lens = 52.0
            distance = max(radius / math.sin(camera_data.angle * 0.5) * 1.3, span * 2.2)
        else:
            camera_data.type = "ORTHO"
            camera_data.ortho_scale = span * 1.24
            distance = span * 3.0
        camera.location = center + direction * distance
        _point_at(camera, center)
        _set_rig_for_view(direction, center, span, lights)
        destination = output_dir / f"render-{view}.png"
        scene.render.filepath = str(destination)
        bpy.ops.render.render(write_still=True)
        rendered[view] = {
            "name": destination.name,
            "sha256": _sha256(destination),
            "direction": [float(value) for value in direction],
        }

    manifest = {
        "schemaVersion": 1,
        "route": "render-model-preview-v1",
        "source": {
            "name": source.name,
            "extension": source.suffix.lower(),
            "sha256": _sha256(source),
            "modified": False,
        },
        "render": {
            "engine": scene.render.engine,
            "resolution": arguments.resolution,
            "materialMode": arguments.material_mode,
            "background": arguments.background,
            "frame": scene.frame_current,
            "embeddedAutoexecEnabled": False,
            "boundsMeters": {
                "minimum": [float(value) for value in minimum],
                "maximum": [float(value) for value in maximum],
            },
            "views": rendered,
        },
    }
    (output_dir / "render-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
