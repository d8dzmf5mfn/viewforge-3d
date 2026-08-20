#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix, Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _animation_common import atomic_write_json, load_json, sha256_file  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently reopen and audit a bone-only biological Action."
    )
    parser.add_argument("--skeleton", required=True, type=Path)
    parser.add_argument("--coordinates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def iter_action_fcurves(action: bpy.types.Action) -> Iterator[Any]:
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        yield from legacy
        return
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for channelbag in getattr(strip, "channelbags", []):
                yield from channelbag.fcurves


def matrix_error_from_identity(matrix: Matrix) -> float:
    identity = Matrix.Identity(4)
    return max(
        abs(matrix[row][column] - identity[row][column]) for row in range(4) for column in range(4)
    )


def angle_degrees(left: Vector, right: Vector) -> float:
    dot = max(-1.0, min(1.0, left.normalized().dot(right.normalized())))
    return math.degrees(math.acos(dot))


def main() -> int:
    arguments = parse_arguments()
    skeleton_path = arguments.skeleton.expanduser().resolve()
    coordinates_path = arguments.coordinates.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    for path in (skeleton_path, coordinates_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    blend_path = Path(bpy.data.filepath).resolve()
    skeleton = load_json(skeleton_path)
    coordinates = load_json(coordinates_path)
    animation = coordinates["animation"]
    chain_names = list(coordinates["retarget"]["movingBones"])
    action_name = animation["actionName"]
    tolerance = float(animation["qa"]["transformTolerance"])

    source_names = set(skeleton["source"]["meshObjects"])
    source_meshes = [
        obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj.name in source_names
    ]
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"expected one Armature, found {len(armatures)}")
    armature = armatures[0]
    action = armature.animation_data.action if armature.animation_data else None
    if action is None or action.name != action_name:
        raise RuntimeError(f"expected active Action {action_name}")
    if any(name not in armature.pose.bones for name in chain_names):
        raise RuntimeError("animated chain is missing from reopened Armature")

    curves = list(iter_action_fcurves(action))
    curve_paths = sorted({curve.data_path for curve in curves})
    expected_fragments = {f'pose.bones["{name}"]' for name in chain_names}
    unexpected_curve_paths = [
        path for path in curve_paths if not any(fragment in path for fragment in expected_fragments)
    ]
    interpolation_mismatches = 0
    for curve in curves:
        for point in curve.keyframe_points:
            if (
                point.interpolation != animation["interpolation"]["type"]
                or point.handle_left_type != animation["interpolation"]["handleType"]
                or point.handle_right_type != animation["interpolation"]["handleType"]
            ):
                interpolation_mismatches += 1

    rest_lengths = {bone.name: float(bone.length) for bone in armature.data.bones}
    previous_directions: dict[str, Vector] = {}
    max_step = 0.0
    max_depth = 0.0
    max_length_error = 0.0
    max_location = 0.0
    max_scale_error = 0.0
    non_finite_values = 0
    samples: list[dict[str, Any]] = []
    scene = bpy.context.scene
    frame_start = int(scene.frame_start)
    frame_end = int(scene.frame_end)
    schedule_frames = {int(entry["frame"]) for entry in animation["schedule"]}
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        frame_angles: dict[str, float] = {}
        for name in chain_names:
            pose_bone = armature.pose.bones[name]
            direction = pose_bone.tail - pose_bone.head
            max_depth = max(max_depth, abs(direction.y))
            max_length_error = max(max_length_error, abs(direction.length - rest_lengths[name]))
            max_location = max(max_location, pose_bone.location.length)
            max_scale_error = max(
                max_scale_error, max(abs(value - 1.0) for value in pose_bone.scale)
            )
            for row in pose_bone.matrix:
                for value in row:
                    if not math.isfinite(value):
                        non_finite_values += 1
            if name in previous_directions:
                max_step = max(max_step, angle_degrees(previous_directions[name], direction))
            previous_directions[name] = direction.copy()
            frame_angles[name] = round(math.degrees(math.atan2(direction.z, direction.x)), 4)
        if frame in schedule_frames:
            samples.append({"frame": frame, "anglesBlenderXZ": frame_angles})

    scene.frame_set(frame_start)
    start_error = max(
        matrix_error_from_identity(armature.pose.bones[name].matrix_basis) for name in chain_names
    )
    scene.frame_set(frame_end)
    end_error = max(
        matrix_error_from_identity(armature.pose.bones[name].matrix_basis) for name in chain_names
    )
    expected_preview_count = len(armature.data.bones) * 2 + 1
    preview_objects = [
        obj
        for obj in scene.objects
        if obj.type == "MESH" and obj.get("vf_preview_kind") in {"animated_bone", "animated_joint"}
    ]
    checks = {
        "sourceMeshSetMatches": {obj.name for obj in source_meshes} == source_names,
        "noSourceWeights": all(not obj.vertex_groups for obj in source_meshes),
        "noSourceArmatureModifiers": all(
            modifier.type != "ARMATURE" for obj in source_meshes for modifier in obj.modifiers
        ),
        "noSourceArmatureParenting": all(obj.parent != armature for obj in source_meshes),
        "allBonesNonDeforming": all(not bone.use_deform for bone in armature.data.bones),
        "boneSetMatchesSkeleton": {bone.name for bone in armature.data.bones}
        == {bone["name"] for bone in skeleton["bones"]},
        "onlyExpectedBonesAnimated": not unexpected_curve_paths,
        "onlyQuaternionCurves": all(
            curve.data_path.endswith("rotation_quaternion") for curve in curves
        ),
        "interpolationMatchesProfile": interpolation_mismatches == 0,
        "frameRangeMatchesSchedule": int(action.frame_range[0]) == frame_start
        and int(action.frame_range[1]) == frame_end
        and frame_start == min(schedule_frames)
        and frame_end == max(schedule_frames),
        "finitePoseMatrices": non_finite_values == 0,
        "planarMotion": max_depth <= tolerance,
        "boneLengthsStable": max_length_error <= tolerance,
        "noAnimatedTranslations": max_location <= tolerance,
        "noAnimatedScale": max_scale_error <= tolerance,
        "startAtRest": start_error <= tolerance,
        "endAtRest": end_error <= tolerance,
        "noLargeInterframeJump": max_step
        <= float(animation["qa"]["maxInterframeAngularStepDegrees"]),
        "completeAnimatedSkeletonPreview": len(preview_objects) == expected_preview_count,
    }
    passed = all(checks.values())
    payload = {
        "schemaVersion": 1,
        "kind": "bone-only-biological-animation-reopen-audit",
        "state": "pendingUserSignoff" if passed else "failed",
        "passed": passed,
        "checks": checks,
        "blend": {"path": str(blend_path), "sha256": sha256_file(blend_path)},
        "action": {
            "name": action.name,
            "animatedBones": chain_names,
            "fCurveCount": len(curves),
            "curvePaths": curve_paths,
            "unexpectedCurvePaths": unexpected_curve_paths,
            "interpolationMismatches": interpolation_mismatches,
            "maxInterframeAngularStepDegrees": round(max_step, 8),
            "maxDepthComponent": round(max_depth, 10),
            "maxBoneLengthError": round(max_length_error, 10),
            "maxPoseLocation": round(max_location, 10),
            "maxScaleError": round(max_scale_error, 10),
            "startRestMatrixError": round(start_error, 10),
            "endRestMatrixError": round(end_error, 10),
            "keySamples": samples,
        },
    }
    atomic_write_json(output, payload)
    if not passed:
        raise RuntimeError(f"reopen audit failed: {output}")
    print(
        "VF_BONE_ANIMATION_AUDIT",
        json.dumps({"passed": True, "output": str(output)}, sort_keys=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
