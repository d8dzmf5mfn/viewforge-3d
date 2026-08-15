#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
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
        description="Build a three-bone, rotation-only biological Action."
    )
    parser.add_argument("--input-blend", required=True, type=Path)
    parser.add_argument("--skeleton", required=True, type=Path)
    parser.add_argument("--coordinates", required=True, type=Path)
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


def source_snapshot(objects: list[bpy.types.Object]) -> dict[str, Any]:
    return {
        obj.name: {
            "meshDigest": mesh_digest(obj),
            "matrixWorld": [[float(value) for value in row] for row in obj.matrix_world],
            "materials": [
                slot.material.name if slot.material else None for slot in obj.material_slots
            ],
            "vertexGroups": [group.name for group in obj.vertex_groups],
            "modifiers": [modifier.type for modifier in obj.modifiers],
            "parent": obj.parent.name if obj.parent else None,
            "parentType": obj.parent_type,
        }
        for obj in sorted(objects, key=lambda item: item.name)
    }


def set_pose_matrices(armature: bpy.types.Object, matrix_map: dict[str, Matrix]) -> None:
    def recurse(pose_bone: bpy.types.PoseBone, parent_matrix: Matrix | None) -> None:
        if pose_bone.name in matrix_map:
            matrix = matrix_map[pose_bone.name]
            if pose_bone.parent:
                pose_bone.matrix_basis = pose_bone.bone.convert_local_to_pose(
                    matrix,
                    pose_bone.bone.matrix_local,
                    parent_matrix=parent_matrix,
                    parent_matrix_local=pose_bone.parent.bone.matrix_local,
                    invert=True,
                )
            else:
                pose_bone.matrix_basis = pose_bone.bone.convert_local_to_pose(
                    matrix, pose_bone.bone.matrix_local, invert=True
                )
        elif pose_bone.parent:
            matrix = pose_bone.bone.convert_local_to_pose(
                pose_bone.matrix_basis,
                pose_bone.bone.matrix_local,
                parent_matrix=parent_matrix,
                parent_matrix_local=pose_bone.parent.bone.matrix_local,
            )
        else:
            matrix = pose_bone.bone.convert_local_to_pose(
                pose_bone.matrix_basis, pose_bone.bone.matrix_local
            )
        for child in pose_bone.children:
            recurse(child, matrix)

    for pose_bone in armature.pose.bones:
        if not pose_bone.parent:
            recurse(pose_bone, None)


def matrix_for_direction(bone: bpy.types.Bone, head: Vector, direction: Vector) -> Matrix:
    target = direction.normalized()
    rest = (bone.tail_local - bone.head_local).normalized()
    rotation = rest.rotation_difference(target) @ bone.matrix_local.to_quaternion()
    matrix = rotation.to_matrix().to_4x4()
    matrix.translation = head
    return matrix


def tail_from_matrix(matrix: Matrix, length: float) -> Vector:
    return matrix @ Vector((0.0, length, 0.0))


def reset_pose(armature: bpy.types.Object) -> None:
    for pose_bone in armature.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.matrix_basis.identity()


def apply_direction_pose(
    armature: bpy.types.Object,
    chain_names: list[str],
    proximal_direction: Vector,
    distal_direction: Vector,
) -> None:
    first, second, third = (armature.data.bones[name] for name in chain_names)
    first_matrix = matrix_for_direction(first, first.head_local.copy(), proximal_direction)
    second_head = tail_from_matrix(first_matrix, first.length)
    second_matrix = matrix_for_direction(second, second_head, distal_direction)
    third_head = tail_from_matrix(second_matrix, second.length)
    third_matrix = matrix_for_direction(third, third_head, distal_direction)
    set_pose_matrices(
        armature,
        {
            chain_names[0]: first_matrix,
            chain_names[1]: second_matrix,
            chain_names[2]: third_matrix,
        },
    )
    bpy.context.view_layer.update()


def direction_from_payload(payload: dict[str, Any], key: str) -> Vector:
    values = payload[key]["unitBlenderXYZ"]
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"invalid direction payload: {key}")
    direction = Vector(tuple(float(value) for value in values))
    if not all(math.isfinite(value) for value in direction) or direction.length <= 1e-8:
        raise ValueError(f"invalid finite direction: {key}")
    if abs(direction.y) > 1e-8:
        raise ValueError("front-image retarget directions must remain in Blender XZ plane")
    return direction.normalized()


def iter_action_fcurves(action: bpy.types.Action) -> Iterator[Any]:
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        yield from legacy
        return
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for channelbag in getattr(strip, "channelbags", []):
                yield from channelbag.fcurves


def configure_interpolation(
    action: bpy.types.Action, interpolation: dict[str, str]
) -> tuple[int, int]:
    curve_count = 0
    point_count = 0
    for curve in iter_action_fcurves(action):
        curve_count += 1
        for point in curve.keyframe_points:
            point.interpolation = interpolation["type"]
            point.handle_left_type = interpolation["handleType"]
            point.handle_right_type = interpolation["handleType"]
            point_count += 1
    return curve_count, point_count


def principled_emission_material(
    name: str, color: tuple[float, float, float, float]
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Metallic"].default_value = 0.0
    shader.inputs["Roughness"].default_value = 0.3
    emission = shader.inputs.get("Emission Color") or shader.inputs.get("Emission")
    if emission:
        emission.default_value = color
    strength = shader.inputs.get("Emission Strength")
    if strength:
        strength.default_value = 3.0
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def move_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)


def add_copy_transforms(obj: bpy.types.Object, armature: bpy.types.Object, bone_name: str) -> None:
    constraint = obj.constraints.new(type="COPY_TRANSFORMS")
    constraint.name = f"VF_Follow__{bone_name}"
    constraint.target = armature
    constraint.subtarget = bone_name
    constraint.target_space = "WORLD"
    constraint.owner_space = "WORLD"


def create_animated_preview(armature: bpy.types.Object, diagonal: float) -> list[bpy.types.Object]:
    collection_name = "VF_AnimatedBonePreview"
    if bpy.data.collections.get(collection_name):
        raise RuntimeError(f"derivative already contains {collection_name}")
    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)
    bone_radius = max(diagonal * 0.0075, 0.025)
    joint_radius = bone_radius * 1.7
    bone_material = principled_emission_material("VF_AnimatedBoneMaterial", (1.0, 0.22, 0.015, 1.0))
    joint_material = principled_emission_material("VF_AnimatedJointMaterial", (0.0, 0.85, 1.0, 1.0))
    created: list[bpy.types.Object] = []
    for bone in armature.data.bones:
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=12,
            radius=bone_radius,
            depth=bone.length,
            end_fill_type="NGON",
            location=(0.0, 0.0, 0.0),
        )
        cylinder = bpy.context.object
        cylinder.name = f"VF_AnimBone__{bone.name}"
        cylinder.data.transform(
            Matrix.Translation((0.0, bone.length * 0.5, 0.0))
            @ Matrix.Rotation(math.radians(-90.0), 4, "X")
        )
        cylinder.data.materials.append(bone_material)
        cylinder["vf_preview_kind"] = "animated_bone"
        cylinder["vf_bone_name"] = bone.name
        move_to_collection(cylinder, collection)
        add_copy_transforms(cylinder, armature, bone.name)
        created.append(cylinder)

        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=2, radius=joint_radius, location=(0.0, 0.0, 0.0)
        )
        joint = bpy.context.object
        joint.name = f"VF_AnimJointTail__{bone.name}"
        joint.data.transform(Matrix.Translation((0.0, bone.length, 0.0)))
        joint.data.materials.append(joint_material)
        joint["vf_preview_kind"] = "animated_joint"
        joint["vf_bone_name"] = bone.name
        move_to_collection(joint, collection)
        add_copy_transforms(joint, armature, bone.name)
        created.append(joint)

    root = next(bone for bone in armature.data.bones if bone.parent is None)
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=2, radius=joint_radius, location=(0.0, 0.0, 0.0)
    )
    root_joint = bpy.context.object
    root_joint.name = "VF_AnimJointHead__root"
    root_joint.data.materials.append(joint_material)
    root_joint["vf_preview_kind"] = "animated_joint"
    root_joint["vf_bone_name"] = root.name
    move_to_collection(root_joint, collection)
    add_copy_transforms(root_joint, armature, root.name)
    created.append(root_joint)
    return created


def bone_direction(pose_bone: bpy.types.PoseBone) -> Vector:
    return (pose_bone.tail - pose_bone.head).normalized()


def angle_error_degrees(actual: Vector, expected: Vector) -> float:
    dot = max(-1.0, min(1.0, actual.normalized().dot(expected.normalized())))
    return math.degrees(math.acos(dot))


def main() -> int:
    arguments = parse_arguments()
    input_blend = arguments.input_blend.expanduser().resolve()
    skeleton_path = arguments.skeleton.expanduser().resolve()
    coordinates_path = arguments.coordinates.expanduser().resolve()
    output_blend = arguments.output_blend.expanduser().resolve()
    qa_path = arguments.qa.expanduser().resolve()
    for path in (input_blend, skeleton_path, coordinates_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if Path(bpy.data.filepath).resolve() != input_blend:
        raise RuntimeError(f"Blender opened {bpy.data.filepath}, expected {input_blend}")
    for path in (output_blend, qa_path):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    skeleton = load_json(skeleton_path)
    coordinates = load_json(coordinates_path)
    if coordinates.get("kind") != "imagegen-relative-bone-pose":
        raise ValueError("unsupported coordinate document")
    if coordinates.get("source", {}).get("skeletonSha256") != sha256_file(skeleton_path):
        raise ValueError("coordinate document does not match skeleton.json")
    animation = coordinates.get("animation")
    retarget = coordinates.get("retarget")
    if not isinstance(animation, dict) or not isinstance(retarget, dict):
        raise ValueError("coordinate document lacks animation or retarget contract")
    chain_names = retarget.get("movingBones")
    if not isinstance(chain_names, list) or len(chain_names) != 3:
        raise ValueError("retarget movingBones must contain exactly three names")
    schedule = animation.get("schedule")
    if not isinstance(schedule, list) or len(schedule) < 2:
        raise ValueError("coordinate document lacks animation schedule")
    if schedule[0].get("pose") != "rest" or schedule[-1].get("pose") != "rest":
        raise ValueError("animation must start and end at rest")
    action_name = animation.get("actionName")
    if not isinstance(action_name, str) or not action_name:
        raise ValueError("animation requires actionName")

    source_hash_before = sha256_file(input_blend)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"expected one Armature, found {len(armatures)}")
    armature = armatures[0]
    if any(name not in armature.pose.bones for name in chain_names):
        raise RuntimeError("animation target bones are missing")
    if armature.data.bones[chain_names[1]].parent != armature.data.bones[chain_names[0]]:
        raise RuntimeError("second animation bone is not parented to first")
    if armature.data.bones[chain_names[2]].parent != armature.data.bones[chain_names[1]]:
        raise RuntimeError("third animation bone is not parented to second")
    if armature.animation_data and armature.animation_data.action:
        raise RuntimeError("input Armature already has an active Action; use an immutable baseline")
    if bpy.data.actions.get(action_name):
        raise RuntimeError(f"input already contains Action datablock {action_name}")

    expected_source_names = set(skeleton["source"]["meshObjects"])
    source_meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.name in expected_source_names
    ]
    if {obj.name for obj in source_meshes} != expected_source_names:
        raise RuntimeError("source mesh set differs from skeleton record")
    source_before = source_snapshot(source_meshes)
    rest_lengths = {bone.name: float(bone.length) for bone in armature.data.bones}
    if any(bone.use_deform for bone in armature.data.bones):
        raise RuntimeError("bone-only input contains deforming bones")
    if any(obj.vertex_groups for obj in source_meshes):
        raise RuntimeError("source meshes unexpectedly contain vertex groups")
    if any(modifier.type == "ARMATURE" for obj in source_meshes for modifier in obj.modifiers):
        raise RuntimeError("source meshes unexpectedly contain Armature modifiers")
    if any(obj.parent == armature for obj in source_meshes):
        raise RuntimeError("source meshes unexpectedly parent to the Armature")

    for obj in bpy.context.scene.objects:
        if obj.get("vf_preview_kind") in {"bone", "joint"}:
            obj.hide_viewport = True
            obj.hide_render = True
    armature.show_in_front = True
    armature.data.display_type = "OCTAHEDRAL"
    armature.hide_render = True
    armature["vf_animation_kind"] = "bone-only-three-bone-direction"
    armature["vf_animation_coordinate_source"] = str(coordinates_path)
    armature["vf_animated_bones"] = json.dumps(chain_names)

    reset_pose(armature)
    armature.animation_data_create()
    action = bpy.data.actions.new(action_name)
    armature.animation_data.action = action
    scene = bpy.context.scene
    schedule_start = min(int(entry["frame"]) for entry in schedule)
    schedule_end = max(int(entry["frame"]) for entry in schedule)
    scene.render.fps = int(animation["fps"])
    scene.frame_start = schedule_start
    scene.frame_end = schedule_end
    scene.frame_preview_start = schedule_start
    scene.frame_preview_end = schedule_end

    keyed_basis: dict[str, Any] = {}
    for entry in schedule:
        frame = int(entry["frame"])
        pose_name = entry["pose"]
        scene.frame_set(frame)
        reset_pose(armature)
        if pose_name != "rest":
            try:
                pose = coordinates["frames"][pose_name]
            except KeyError as error:
                raise ValueError(f"scheduled pose lacks coordinates: {pose_name}") from error
            apply_direction_pose(
                armature,
                chain_names,
                direction_from_payload(pose, "proximalDirection"),
                direction_from_payload(pose, "distalDirection"),
            )
        entry_basis: dict[str, Any] = {}
        for name in chain_names:
            pose_bone = armature.pose.bones[name]
            tolerance = float(animation["qa"]["transformTolerance"])
            if pose_bone.location.length > tolerance:
                raise RuntimeError(
                    f"retarget introduced translation on {name}: {pose_bone.location}"
                )
            scale_error = max(abs(value - 1.0) for value in pose_bone.scale)
            if scale_error > tolerance:
                raise RuntimeError(f"retarget introduced scale on {name}: {pose_bone.scale}")
            pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=name)
            entry_basis[name] = {
                "location": [round(float(value), 8) for value in pose_bone.location],
                "rotationQuaternion": [
                    round(float(value), 8) for value in pose_bone.rotation_quaternion
                ],
                "scale": [round(float(value), 8) for value in pose_bone.scale],
            }
        keyed_basis[str(frame)] = entry_basis
    curve_count, keyframe_point_count = configure_interpolation(action, animation["interpolation"])

    points = [
        obj.matrix_world @ Vector(corner) for obj in source_meshes for corner in obj.bound_box
    ]
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    preview_objects = create_animated_preview(armature, (maximum - minimum).length)
    for obj in source_meshes:
        obj.hide_render = True

    pose_checks: list[dict[str, Any]] = []
    max_direction_error = 0.0
    for entry in schedule:
        frame = int(entry["frame"])
        pose_name = entry["pose"]
        scene.frame_set(frame)
        if pose_name == "rest":
            expected = [
                (bone.tail_local - bone.head_local).normalized()
                for bone in (armature.data.bones[name] for name in chain_names)
            ]
        else:
            pose = coordinates["frames"][pose_name]
            proximal = direction_from_payload(pose, "proximalDirection")
            distal = direction_from_payload(pose, "distalDirection")
            expected = [proximal, distal, distal]
        errors = [
            angle_error_degrees(bone_direction(armature.pose.bones[name]), direction)
            for name, direction in zip(chain_names, expected, strict=True)
        ]
        max_direction_error = max(max_direction_error, *errors)
        pose_checks.append(
            {
                "frame": frame,
                "pose": pose_name,
                "boneErrorsDegrees": {
                    name: round(error, 6) for name, error in zip(chain_names, errors, strict=True)
                },
            }
        )

    source_after = source_snapshot(source_meshes)
    lengths_after = {bone.name: float(bone.length) for bone in armature.data.bones}
    tolerance = float(animation["qa"]["transformTolerance"])
    checks = {
        "inputBlendHashUnchanged": source_hash_before == sha256_file(input_blend),
        "sourceGeometryTransformsMaterialsUnchanged": source_before == source_after,
        "sourceMeshSetMatchesSkeleton": len(source_meshes) == len(expected_source_names),
        "oneArmature": len(armatures) == 1,
        "boneSetMatchesSkeleton": {bone.name for bone in armature.data.bones}
        == {bone["name"] for bone in skeleton["bones"]},
        "allBonesNonDeforming": all(not bone.use_deform for bone in armature.data.bones),
        "noSourceWeights": all(not obj.vertex_groups for obj in source_meshes),
        "noSourceArmatureModifiers": all(
            modifier.type != "ARMATURE" for obj in source_meshes for modifier in obj.modifiers
        ),
        "noSourceArmatureParenting": all(obj.parent != armature for obj in source_meshes),
        "boneLengthsUnchanged": rest_lengths == lengths_after,
        "animatedBonesOnlyUseRelativeRotation": all(
            all(abs(value) <= tolerance for value in basis[name]["location"])
            and all(abs(value - 1.0) <= tolerance for value in basis[name]["scale"])
            for basis in keyed_basis.values()
            for name in chain_names
        ),
        "poseDirectionsMatchImagegenReferences": max_direction_error
        <= float(animation["qa"]["maxDirectionErrorDegrees"]),
        "actionRangeMatchesSchedule": int(action.frame_range[0]) == schedule_start
        and int(action.frame_range[1]) == schedule_end,
        "previewUsesConstraintsNotSkin": all(
            obj.get("vf_preview_kind") in {"animated_bone", "animated_joint"}
            and len(obj.constraints) == 1
            and obj.constraints[0].type == "COPY_TRANSFORMS"
            for obj in preview_objects
        ),
    }
    passed = all(checks.values())
    scene.frame_set(schedule_start)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend), compress=True)
    qa = {
        "schemaVersion": 1,
        "kind": "bone-only-biological-animation",
        "state": "pendingUserSignoff" if passed else "failed",
        "passed": passed,
        "checks": checks,
        "source": {
            "inputBlend": str(input_blend),
            "inputBlendSha256": source_hash_before,
            "skeleton": str(skeleton_path),
            "coordinates": str(coordinates_path),
        },
        "animation": {
            "action": action_name,
            "fps": scene.render.fps,
            "frameStart": scene.frame_start,
            "frameEnd": scene.frame_end,
            "durationSeconds": (scene.frame_end - scene.frame_start + 1) / scene.render.fps,
            "animatedBones": chain_names,
            "interpolation": animation["interpolation"],
            "fCurveCount": curve_count,
            "keyframePointCount": keyframe_point_count,
            "schedule": schedule,
            "keyedBasis": keyed_basis,
            "poseDirectionChecks": pose_checks,
            "maxDirectionErrorDegrees": round(max_direction_error, 8),
        },
        "output": {
            "blend": str(output_blend),
            "blendSha256": sha256_file(output_blend),
            "animatedPreviewMeshCount": len(preview_objects),
            "skinWeights": False,
            "sourceMeshDeformation": False,
        },
    }
    atomic_write_json(qa_path, qa)
    if not passed:
        raise RuntimeError(f"animation QA failed: {qa_path}")
    print(
        "VF_BONE_ANIMATION_RESULT",
        json.dumps(
            {
                "passed": True,
                "blend": str(output_blend),
                "qa": str(qa_path),
                "frames": [scene.frame_start, scene.frame_end],
                "fps": scene.render.fps,
            },
            sort_keys=True,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
