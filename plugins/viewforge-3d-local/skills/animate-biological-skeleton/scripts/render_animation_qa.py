#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _animation_common import atomic_write_json, load_json, sha256_file  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render fixed-front biological animation QA with a complete skeleton overlay."
    )
    parser.add_argument("--skeleton", required=True, type=Path)
    parser.add_argument("--coordinates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("auto", "bone-only", "rigid-bound"), default="auto")
    parser.add_argument("--frames", choices=("schedule", "all"), default="schedule")
    parser.add_argument("--size", default=640, type=int)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def point_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def add_area_light(name: str, location: Vector, target: Vector, energy: float, size: float) -> None:
    data = bpy.data.lights.new(name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    point_at(obj, target)


def source_material() -> bpy.types.Material:
    material = bpy.data.materials.new("VF_AnimationQA_Source")
    material.diffuse_color = (0.025, 0.33, 0.29, 1.0)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = (0.025, 0.33, 0.29, 1.0)
    shader.inputs["Metallic"].default_value = 0.0
    shader.inputs["Roughness"].default_value = 0.38
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def main() -> int:
    arguments = parse_arguments()
    skeleton_path = arguments.skeleton.expanduser().resolve()
    coordinates_path = arguments.coordinates.expanduser().resolve()
    output_dir = arguments.output_dir.expanduser().resolve()
    for path in (skeleton_path, coordinates_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    skeleton = load_json(skeleton_path)
    coordinates = load_json(coordinates_path)
    scene = bpy.context.scene
    source_names = set(skeleton["source"]["meshObjects"])
    source_objects = [
        obj for obj in scene.objects if obj.type == "MESH" and obj.name in source_names
    ]
    if {obj.name for obj in source_objects} != source_names:
        raise RuntimeError("source mesh set differs from skeleton record")
    bone_objects = [
        obj
        for obj in scene.objects
        if obj.type == "MESH" and obj.get("vf_preview_kind") in {"animated_bone", "animated_joint"}
    ]
    expected_bone_objects = len(skeleton["bones"]) * 2 + 1
    if len(bone_objects) != expected_bone_objects:
        raise RuntimeError(
            f"expected {expected_bone_objects} animated skeleton meshes, found {len(bone_objects)}"
        )
    armatures = [obj for obj in scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"expected one Armature, found {len(armatures)}")
    armature = armatures[0]
    action = armature.animation_data.action if armature.animation_data else None
    if action is None or action.name != coordinates["animation"]["actionName"]:
        raise RuntimeError("expected animation Action is not active")

    all_bound = all(
        obj.parent == armature and obj.parent_type == "BONE" and bool(obj.parent_bone)
        for obj in source_objects
    )
    mode = arguments.mode
    if mode == "auto":
        mode = "rigid-bound" if all_bound else "bone-only"
    if mode == "rigid-bound" and not all_bound:
        raise RuntimeError(
            "rigid-bound render requested but source components are not Bone Parent bound"
        )
    if mode == "bone-only" and any(obj.parent == armature for obj in source_objects):
        raise RuntimeError("bone-only render requested for a bound model")

    if arguments.frames == "all":
        render_frames = list(range(scene.frame_start, scene.frame_end + 1))
    else:
        render_frames = sorted(
            {int(entry["frame"]) for entry in coordinates["animation"]["schedule"]}
        )
    if not render_frames:
        raise ValueError("no render frames selected")
    metrics_path = output_dir / "render-metrics.json"
    if mode == "rigid-bound":
        source_dir = output_dir / "source"
        bones_dir = output_dir / "bones"
        planned_paths = [
            *(source_dir / f"frame-{frame:04d}.png" for frame in render_frames),
            *(bones_dir / f"frame-{frame:04d}.png" for frame in render_frames),
        ]
    else:
        frames_dir = output_dir / "frames"
        planned_paths = [frames_dir / f"frame-{frame:04d}.png" for frame in render_frames]
    collisions = [str(path) for path in [metrics_path, *planned_paths] if path.exists()]
    if collisions:
        raise FileExistsError(f"immutable render outputs already exist: {collisions[:4]}")
    for path in planned_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    material = source_material()
    for obj in source_objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)
        obj.hide_viewport = False
    for obj in scene.objects:
        if obj.get("vf_preview_kind") in {"bone", "joint"}:
            obj.hide_render = True
            obj.hide_viewport = True
    armature.hide_render = True

    sampled: list[Vector] = []
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        if mode == "rigid-bound":
            for obj in source_objects:
                sampled.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
        else:
            for pose_bone in armature.pose.bones:
                sampled.append(armature.matrix_world @ pose_bone.head)
                sampled.append(armature.matrix_world @ pose_bone.tail)
    minimum = Vector(tuple(min(point[axis] for point in sampled) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in sampled) for axis in range(3)))
    center = (minimum + maximum) * 0.5
    dimensions = maximum - minimum
    ortho_scale = max(dimensions.x, dimensions.z) * 1.18

    camera_data = bpy.data.cameras.new("VF_AnimationQA_Camera")
    camera = bpy.data.objects.new("VF_AnimationQA_Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = ortho_scale
    camera_data.dof.use_dof = False
    camera.location = center + Vector((0.0, ortho_scale * 3.0, 0.0))
    point_at(camera, center)

    add_area_light(
        "VF_AnimationQA_Key",
        center + Vector((-2.0 * ortho_scale, 2.8 * ortho_scale, 2.2 * ortho_scale)),
        center,
        8500.0,
        2.2 * ortho_scale,
    )
    add_area_light(
        "VF_AnimationQA_Fill",
        center + Vector((2.2 * ortho_scale, 1.8 * ortho_scale, 0.8 * ortho_scale)),
        center,
        4400.0,
        2.6 * ortho_scale,
    )
    world = bpy.data.worlds.new("VF_AnimationQA_World")
    world.use_nodes = True
    world.node_tree.nodes.clear()
    background = world.node_tree.nodes.new("ShaderNodeBackground")
    world_output = world.node_tree.nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(background.outputs["Background"], world_output.inputs["Surface"])
    background.inputs["Color"].default_value = (0.004, 0.007, 0.012, 1.0)
    background.inputs["Strength"].default_value = 0.06
    scene.world = world

    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = arguments.size
    scene.render.resolution_y = arguments.size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.use_file_extension = True
    scene.view_settings.look = "AgX - Medium High Contrast"

    records: list[dict[str, Any]] = []
    for frame in render_frames:
        scene.frame_set(frame)
        if mode == "rigid-bound":
            source_path = output_dir / "source" / f"frame-{frame:04d}.png"
            bone_path = output_dir / "bones" / f"frame-{frame:04d}.png"
            for obj in source_objects:
                obj.hide_render = False
            for obj in bone_objects:
                obj.hide_render = True
            scene.render.film_transparent = False
            scene.render.filepath = str(source_path)
            bpy.ops.render.render(write_still=True)
            for obj in source_objects:
                obj.hide_render = True
            for obj in bone_objects:
                obj.hide_render = False
            scene.render.film_transparent = True
            scene.render.filepath = str(bone_path)
            bpy.ops.render.render(write_still=True)
            records.append(
                {
                    "frame": frame,
                    "source": {
                        "path": str(source_path.relative_to(output_dir)),
                        "sha256": sha256_file(source_path),
                    },
                    "bones": {
                        "path": str(bone_path.relative_to(output_dir)),
                        "sha256": sha256_file(bone_path),
                    },
                }
            )
        else:
            frame_path = output_dir / "frames" / f"frame-{frame:04d}.png"
            for obj in source_objects:
                obj.hide_render = True
            for obj in bone_objects:
                obj.hide_render = False
            scene.render.film_transparent = False
            scene.render.filepath = str(frame_path)
            bpy.ops.render.render(write_still=True)
            records.append(
                {
                    "frame": frame,
                    "combined": {
                        "path": str(frame_path.relative_to(output_dir)),
                        "sha256": sha256_file(frame_path),
                    },
                }
            )
    scene.frame_set(scene.frame_start)
    metrics = {
        "schemaVersion": 1,
        "kind": "biological-animation-render-layers",
        "sourceBlend": bpy.data.filepath,
        "mode": mode,
        "frameSelection": arguments.frames,
        "fps": scene.render.fps,
        "frameStart": scene.frame_start,
        "frameEnd": scene.frame_end,
        "resolution": [arguments.size, arguments.size],
        "camera": {
            "projection": "orthographic",
            "view": "front",
            "orthoScale": ortho_scale,
            "boundsMin": list(minimum),
            "boundsMax": list(maximum),
        },
        "display": {
            "sourceComponents": len(source_objects),
            "animatedSkeletonMeshes": len(bone_objects),
            "completeSkeletonVisibleInFinalComposite": True,
            "skin": False,
        },
        "frames": records,
    }
    atomic_write_json(metrics_path, metrics)
    print(
        "VF_ANIMATION_RENDER_RESULT",
        json.dumps(
            {"frames": len(records), "mode": mode, "outputDir": str(output_dir)},
            sort_keys=True,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
