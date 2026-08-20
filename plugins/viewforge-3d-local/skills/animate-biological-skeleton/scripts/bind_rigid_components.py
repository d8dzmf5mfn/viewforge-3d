#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _animation_common import (  # noqa: E402
    atomic_write_json,
    load_json,
    sha256_file,
    validate_binding_profile,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rigidly bind segmented biological components without skin weights."
    )
    parser.add_argument("--input-blend", required=True, type=Path)
    parser.add_argument("--skeleton", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--qa", required=True, type=Path)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def mesh_digest(obj: bpy.types.Object) -> str:
    digest = hashlib.sha256()
    for vertex in obj.data.vertices:
        digest.update(struct.pack("<3d", *vertex.co))
    for polygon in obj.data.polygons:
        digest.update(struct.pack("<I", len(polygon.vertices)))
        for index in polygon.vertices:
            digest.update(struct.pack("<I", index))
    return digest.hexdigest()


def matrix_payload(matrix: Matrix) -> list[list[float]]:
    return [[round(float(value), 9) for value in row] for row in matrix]


def matrix_error(left: Matrix, right: Matrix) -> float:
    return max(
        abs(left[row][column] - right[row][column]) for row in range(4) for column in range(4)
    )


def bone_world_matrix(armature: bpy.types.Object, bone_name: str) -> Matrix:
    return armature.matrix_world @ armature.pose.bones[bone_name].matrix


def iter_action_fcurves(action: bpy.types.Action) -> Iterator[Any]:
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        yield from legacy
        return
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for channelbag in getattr(strip, "channelbags", []):
                yield from channelbag.fcurves


def action_keyframes(action: bpy.types.Action, frame_start: int, frame_end: int) -> list[int]:
    frames = {frame_start, frame_end}
    for curve in iter_action_fcurves(action):
        for point in curve.keyframe_points:
            frame = int(round(float(point.co.x)))
            if frame_start <= frame <= frame_end:
                frames.add(frame)
    return sorted(frames)


def main() -> int:
    arguments = parse_arguments()
    input_blend = arguments.input_blend.expanduser().resolve()
    skeleton_path = arguments.skeleton.expanduser().resolve()
    mapping_path = arguments.mapping.expanduser().resolve()
    output_blend = arguments.output_blend.expanduser().resolve()
    qa_path = arguments.qa.expanduser().resolve()
    for path in (input_blend, skeleton_path, mapping_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if Path(bpy.data.filepath).resolve() != input_blend:
        raise RuntimeError(f"Blender opened {bpy.data.filepath}, expected {input_blend}")
    for path in (output_blend, qa_path):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    source_hash_before = sha256_file(input_blend)
    skeleton = load_json(skeleton_path)
    mapping_document = load_json(mapping_path)
    validate_binding_profile(mapping_document)
    mapping: dict[str, str] = mapping_document["components"]
    expected_source_names = set(skeleton["source"]["meshObjects"])
    if (
        mapping_document.get("requireCompleteSourceSet", False)
        and set(mapping) != expected_source_names
    ):
        raise ValueError("binding map does not match the complete recorded source component set")
    if not set(mapping).issubset(expected_source_names):
        raise ValueError("binding map contains components absent from skeleton record")

    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"expected one Armature, found {len(armatures)}")
    armature = armatures[0]
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        raise RuntimeError("input Armature has no active animation Action")
    if action.name != mapping_document["expectedActionName"]:
        raise RuntimeError(
            f"expected Action {mapping_document['expectedActionName']}, found {action.name}"
        )
    missing_bones = sorted(set(mapping.values()) - set(armature.pose.bones.keys()))
    if missing_bones:
        raise ValueError(f"mapping references missing bones: {missing_bones}")

    source_objects = {name: bpy.data.objects.get(name) for name in mapping}
    if any(obj is None or obj.type != "MESH" for obj in source_objects.values()):
        raise RuntimeError("one or more mapped source components are missing mesh objects")
    typed_source_objects: dict[str, bpy.types.Object] = {
        name: obj for name, obj in source_objects.items() if obj is not None
    }
    if any(obj.parent == armature for obj in typed_source_objects.values()):
        raise RuntimeError("mapped components are already parented to the Armature")
    if any(obj.vertex_groups for obj in typed_source_objects.values()):
        raise RuntimeError("source components unexpectedly contain vertex groups")
    if any(
        modifier.type == "ARMATURE"
        for obj in typed_source_objects.values()
        for modifier in obj.modifiers
    ):
        raise RuntimeError("source components unexpectedly contain Armature modifiers")

    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)
    bpy.context.view_layer.update()
    geometry_before = {name: mesh_digest(obj) for name, obj in typed_source_objects.items()}
    rest_world_before = {
        name: obj.matrix_world.copy() for name, obj in typed_source_objects.items()
    }
    original_parents = {
        name: {
            "parent": obj.parent.name if obj.parent else None,
            "parentType": obj.parent_type,
        }
        for name, obj in typed_source_objects.items()
    }
    rest_offsets = {
        name: bone_world_matrix(armature, bone_name).inverted() @ rest_world_before[name]
        for name, bone_name in mapping.items()
    }

    for name, bone_name in mapping.items():
        obj = typed_source_objects[name]
        rest_world = rest_world_before[name].copy()
        obj.parent = armature
        obj.parent_type = "BONE"
        obj.parent_bone = bone_name
        obj.matrix_world = rest_world
        obj.hide_viewport = False
        obj.hide_render = False
        obj["vf_binding_mode"] = "rigid-bone-parent"
        obj["vf_binding_bone"] = bone_name
        obj["vf_geometry_sha256"] = geometry_before[name]
    bpy.context.view_layer.update()

    rest_world_after = {name: obj.matrix_world.copy() for name, obj in typed_source_objects.items()}
    max_rest_error = max(
        matrix_error(rest_world_before[name], rest_world_after[name]) for name in mapping
    )
    frame_samples = set(action_keyframes(action, int(scene.frame_start), int(scene.frame_end)))
    max_bind_offset_error = 0.0
    max_static_motion = 0.0
    moving_deltas: dict[str, float] = {name: 0.0 for name in mapping}
    animated_components = set(mapping_document["animatedComponents"])
    sample_records: list[dict[str, Any]] = []
    for frame in range(int(scene.frame_start), int(scene.frame_end) + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        frame_record: dict[str, Any] = {"frame": frame, "components": {}}
        for name, bone_name in mapping.items():
            obj = typed_source_objects[name]
            relative = bone_world_matrix(armature, bone_name).inverted() @ obj.matrix_world
            offset_error = matrix_error(rest_offsets[name], relative)
            max_bind_offset_error = max(max_bind_offset_error, offset_error)
            delta = matrix_error(rest_world_before[name], obj.matrix_world)
            moving_deltas[name] = max(moving_deltas[name], delta)
            if name not in animated_components:
                max_static_motion = max(max_static_motion, delta)
            if frame in frame_samples:
                frame_record["components"][name] = {
                    "bone": bone_name,
                    "worldDeltaFromRest": round(delta, 8),
                    "bindOffsetError": round(offset_error, 10),
                }
        if frame in frame_samples:
            sample_records.append(frame_record)

    scene.frame_set(scene.frame_start)
    bpy.context.view_layer.update()
    geometry_after = {name: mesh_digest(obj) for name, obj in typed_source_objects.items()}
    bound_parent_records = {
        name: {
            "parent": obj.parent.name if obj.parent else None,
            "parentType": obj.parent_type,
            "parentBone": obj.parent_bone,
            "restWorld": matrix_payload(obj.matrix_world),
        }
        for name, obj in typed_source_objects.items()
    }
    tolerance = float(mapping_document["qa"]["transformTolerance"])
    minimum_motion = float(mapping_document["qa"]["minimumAnimatedMotion"])
    checks = {
        "inputBlendHashUnchanged": source_hash_before == sha256_file(input_blend),
        "componentSetMatchesMapping": set(typed_source_objects) == set(mapping),
        "completeSourceSetMapped": not mapping_document.get("requireCompleteSourceSet", False)
        or set(mapping) == expected_source_names,
        "geometryUnchanged": geometry_before == geometry_after,
        "restWorldTransformsPreserved": max_rest_error <= tolerance,
        "allComponentsBoneParented": all(
            obj.parent == armature
            and obj.parent_type == "BONE"
            and obj.parent_bone == mapping[name]
            for name, obj in typed_source_objects.items()
        ),
        "noSkinWeights": all(not obj.vertex_groups for obj in typed_source_objects.values()),
        "noArmatureModifiers": all(
            modifier.type != "ARMATURE"
            for obj in typed_source_objects.values()
            for modifier in obj.modifiers
        ),
        "bindOffsetsStableAcrossAllFrames": max_bind_offset_error <= tolerance,
        "declaredAnimatedComponentsMove": all(
            moving_deltas[name] >= minimum_motion for name in animated_components
        ),
        "nonAnimatedComponentsRemainStill": max_static_motion <= tolerance,
        "actionPreserved": action.name == mapping_document["expectedActionName"],
    }
    passed = all(checks.values())
    armature["vf_model_binding"] = "rigid-segmented-bone-parent"
    armature["vf_model_binding_map"] = str(mapping_path)
    armature["vf_model_binding_map_sha256"] = sha256_file(mapping_path)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend), compress=True)
    qa = {
        "schemaVersion": 1,
        "kind": "rigid-segmented-model-binding",
        "state": "pendingUserSignoff" if passed else "failed",
        "passed": passed,
        "checks": checks,
        "contract": {
            "binding": "Bone Parent per segmented mesh component",
            "skinWeights": False,
            "armatureModifiers": False,
            "meshDeformation": False,
            "rigidComponentMotion": True,
        },
        "source": {
            "inputBlend": str(input_blend),
            "inputBlendSha256": source_hash_before,
            "skeleton": str(skeleton_path),
            "mapping": str(mapping_path),
            "mappingSha256": sha256_file(mapping_path),
            "originalParents": original_parents,
        },
        "binding": {
            "mapping": mapping,
            "animatedComponents": sorted(animated_components),
            "boundParents": bound_parent_records,
            "maxRestWorldTransformError": round(max_rest_error, 10),
            "maxBindOffsetError": round(max_bind_offset_error, 10),
            "maxNonAnimatedComponentMotion": round(max_static_motion, 10),
            "componentMotionDeltas": {
                name: round(value, 8) for name, value in sorted(moving_deltas.items())
            },
            "samples": sample_records,
        },
        "output": {
            "blend": str(output_blend),
            "blendSha256": sha256_file(output_blend),
        },
    }
    atomic_write_json(qa_path, qa)
    if not passed:
        raise RuntimeError(f"rigid binding QA failed: {qa_path}")
    print(
        "VF_RIGID_BIND_RESULT",
        json.dumps(
            {"passed": True, "blend": str(output_blend), "qa": str(qa_path)},
            sort_keys=True,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
