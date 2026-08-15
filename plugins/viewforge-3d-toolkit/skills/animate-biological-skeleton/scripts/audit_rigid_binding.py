#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

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
        description="Independently reopen and audit rigid segmented binding."
    )
    parser.add_argument("--skeleton", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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


def matrix_error(left: Matrix, right: Matrix) -> float:
    return max(
        abs(left[row][column] - right[row][column]) for row in range(4) for column in range(4)
    )


def main() -> int:
    arguments = parse_arguments()
    skeleton_path = arguments.skeleton.expanduser().resolve()
    mapping_path = arguments.mapping.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    for path in (skeleton_path, mapping_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    skeleton = load_json(skeleton_path)
    mapping_document = load_json(mapping_path)
    validate_binding_profile(mapping_document)
    mapping: dict[str, str] = mapping_document["components"]
    if mapping_document.get("requireCompleteSourceSet", False) and set(mapping) != set(
        skeleton["source"]["meshObjects"]
    ):
        raise ValueError("binding map no longer matches the skeleton source set")

    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"expected one Armature, found {len(armatures)}")
    armature = armatures[0]
    action = armature.animation_data.action if armature.animation_data else None
    objects = {name: bpy.data.objects.get(name) for name in mapping}
    if any(obj is None or obj.type != "MESH" for obj in objects.values()):
        raise RuntimeError("mapped source mesh is missing")
    source_objects: dict[str, bpy.types.Object] = {
        name: obj for name, obj in objects.items() if obj is not None
    }
    geometry_matches = {
        name: obj.get("vf_geometry_sha256") == mesh_digest(obj)
        for name, obj in source_objects.items()
    }
    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)
    bpy.context.view_layer.update()
    rest_world = {name: obj.matrix_world.copy() for name, obj in source_objects.items()}
    rest_offsets = {
        name: (armature.matrix_world @ armature.pose.bones[bone].matrix).inverted()
        @ rest_world[name]
        for name, bone in mapping.items()
    }
    motion = {name: 0.0 for name in mapping}
    max_offset_error = 0.0
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        for name, bone_name in mapping.items():
            obj = source_objects[name]
            bone_world = armature.matrix_world @ armature.pose.bones[bone_name].matrix
            relative = bone_world.inverted() @ obj.matrix_world
            max_offset_error = max(max_offset_error, matrix_error(rest_offsets[name], relative))
            motion[name] = max(motion[name], matrix_error(rest_world[name], obj.matrix_world))
    animated = set(mapping_document["animatedComponents"])
    max_static_motion = max(
        (value for name, value in motion.items() if name not in animated), default=0.0
    )
    tolerance = float(mapping_document["qa"]["transformTolerance"])
    minimum_motion = float(mapping_document["qa"]["minimumAnimatedMotion"])
    checks = {
        "componentSetMatchesMapping": set(source_objects) == set(mapping),
        "completeSourceSetMapped": not mapping_document.get("requireCompleteSourceSet", False)
        or set(mapping) == set(skeleton["source"]["meshObjects"]),
        "allComponentsBoneParented": all(
            obj.parent == armature
            and obj.parent_type == "BONE"
            and obj.parent_bone == mapping[name]
            for name, obj in source_objects.items()
        ),
        "geometryMatchesBindRecord": all(geometry_matches.values()),
        "noSkinWeights": all(not obj.vertex_groups for obj in source_objects.values()),
        "noArmatureModifiers": all(
            modifier.type != "ARMATURE"
            for obj in source_objects.values()
            for modifier in obj.modifiers
        ),
        "bindOffsetsStableAcrossAllFrames": max_offset_error <= tolerance,
        "declaredAnimatedComponentsMove": all(motion[name] >= minimum_motion for name in animated),
        "nonAnimatedComponentsRemainStill": max_static_motion <= tolerance,
        "expectedActionActive": bool(
            action and action.name == mapping_document["expectedActionName"]
        ),
        "bindingProfileHashMatches": armature.get("vf_model_binding_map_sha256")
        == sha256_file(mapping_path),
    }
    passed = all(checks.values())
    payload = {
        "schemaVersion": 1,
        "kind": "rigid-binding-reopen-audit",
        "state": "pendingUserSignoff" if passed else "failed",
        "passed": passed,
        "checks": checks,
        "blend": {
            "path": bpy.data.filepath,
            "sha256": sha256_file(Path(bpy.data.filepath)),
        },
        "binding": {
            "mapping": mapping,
            "animatedComponents": sorted(animated),
            "geometryMatches": geometry_matches,
            "maxBindOffsetError": round(max_offset_error, 10),
            "maxNonAnimatedComponentMotion": round(max_static_motion, 10),
            "componentMotionDeltas": {
                name: round(value, 8) for name, value in sorted(motion.items())
            },
            "skinWeights": False,
            "armatureModifiers": False,
        },
    }
    atomic_write_json(output, payload)
    if not passed:
        raise RuntimeError(f"rigid binding reopen audit failed: {output}")
    print(
        "VF_RIGID_BIND_AUDIT",
        json.dumps({"passed": True, "output": str(output)}, sort_keys=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
