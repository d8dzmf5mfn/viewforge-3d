#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import bpy  # type: ignore[import-not-found]
    from mathutils import Vector  # type: ignore[import-not-found]
except ModuleNotFoundError:  # Allows contract tests outside Blender.
    bpy = None
    Vector = None


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
ASSET_DIR = SKILL_DIR / "assets"
OUTPUT_NAMES = {
    "skeleton": "skeleton.json",
    "blend": "bone-only-armature.blend",
    "preview": "bone-preview.glb",
    "qa": "qa.json",
}


@dataclass(frozen=True)
class BoneRecord:
    name: str
    parent: str | None
    head_landmark: str
    tail_landmark: str
    head: tuple[float, float, float]
    tail: tuple[float, float, float]
    connect: bool


@dataclass
class MeshPart:
    role: str
    object_name: str
    center: Any
    points: list[Any]


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a bone-only biological Armature without weights or mesh deformation."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", default="humanoid-v1")
    parser.add_argument("--landmarks", type=Path)
    parser.add_argument("--component-map", type=Path)
    parser.add_argument("--front-annotation", type=Path)
    parser.add_argument("--side-annotation", type=Path)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def profile_path(profile: str) -> Path:
    candidate = Path(profile)
    if candidate.is_file():
        return candidate.resolve()
    bundled = ASSET_DIR / f"{profile}.profile.json"
    if not bundled.is_file():
        raise FileNotFoundError(f"unknown skeleton profile: {profile}")
    return bundled


def validate_profile(profile: dict[str, Any]) -> None:
    if profile.get("schemaVersion") != 1:
        raise ValueError("skeleton profile schemaVersion must be 1")
    profile_id = profile.get("profileId")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("skeleton profile requires profileId")
    required = profile.get("requiredLandmarks")
    bones = profile.get("bones")
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise ValueError("requiredLandmarks must be a string list")
    if len(required) != len(set(required)):
        raise ValueError("requiredLandmarks contains duplicates")
    if not isinstance(bones, list) or not bones:
        raise ValueError("profile requires at least one bone")
    names = [bone.get("name") for bone in bones if isinstance(bone, dict)]
    if len(names) != len(bones) or not all(isinstance(name, str) and name for name in names):
        raise ValueError("every bone requires a non-empty name")
    if len(names) != len(set(names)):
        raise ValueError("bone names must be unique")
    name_set = set(names)
    required_set = set(required)
    for bone in bones:
        parent = bone.get("parent")
        if parent is not None and parent not in name_set:
            raise ValueError(f"unknown parent {parent!r} for bone {bone['name']}")
        if bone.get("head") not in required_set or bone.get("tail") not in required_set:
            raise ValueError(f"bone {bone['name']} references an undeclared landmark")
        if not isinstance(bone.get("connect"), bool):
            raise ValueError(f"bone {bone['name']} requires boolean connect")


def landmarks_from_payload(
    payload: dict[str, Any], profile: dict[str, Any]
) -> dict[str, tuple[float, float, float]]:
    if payload.get("schemaVersion") != 1:
        raise ValueError("landmark schemaVersion must be 1")
    if payload.get("profileId") != profile["profileId"]:
        raise ValueError("landmark profileId does not match the selected profile")
    raw = payload.get("landmarks")
    if not isinstance(raw, dict):
        raise ValueError("landmarks must be an object")
    missing = sorted(set(profile["requiredLandmarks"]) - set(raw))
    if missing:
        raise ValueError(f"missing required landmarks: {', '.join(missing)}")
    result: dict[str, tuple[float, float, float]] = {}
    for name in profile["requiredLandmarks"]:
        value = raw[name]
        if (
            not isinstance(value, list)
            or len(value) != 3
            or not all(isinstance(item, int | float) and math.isfinite(item) for item in value)
        ):
            raise ValueError(f"landmark {name} must be three finite numbers")
        result[name] = tuple(float(item) for item in value)
    return result


def bone_records(
    profile: dict[str, Any], landmarks: dict[str, tuple[float, float, float]]
) -> list[BoneRecord]:
    missing = sorted(set(profile["requiredLandmarks"]) - set(landmarks))
    if missing:
        raise ValueError(f"missing required landmarks: {', '.join(missing)}")
    records: list[BoneRecord] = []
    for definition in profile["bones"]:
        head = landmarks[definition["head"]]
        tail = landmarks[definition["tail"]]
        length = math.sqrt(sum((tail[index] - head[index]) ** 2 for index in range(3)))
        if not math.isfinite(length) or length <= 1e-6:
            raise ValueError(f"bone {definition['name']} has zero or invalid length")
        records.append(
            BoneRecord(
                name=definition["name"],
                parent=definition["parent"],
                head_landmark=definition["head"],
                tail_landmark=definition["tail"],
                head=head,
                tail=tail,
                connect=definition["connect"],
            )
        )
    return records


def normalize_name(name: str) -> list[str]:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return [token for token in re.sub(r"[^A-Za-z0-9]+", " ", separated).lower().split() if token]


def object_matches_role(name: str, role: str) -> bool:
    tokens = set(normalize_name(name))
    if role in {"head", "neck", "torso", "pelvis"}:
        aliases = {
            "head": {"head", "skull"},
            "neck": {"neck"},
            "torso": {"torso", "chest", "body"},
            "pelvis": {"pelvis", "hips", "hip"},
        }
        return bool(tokens & aliases[role])
    semantic, side = role.split(".")
    side_tokens = {"L": {"l", "left"}, "R": {"r", "right"}}[side]
    if not tokens & side_tokens:
        return False
    if semantic == "upper_arm":
        return "arm" in tokens and bool(tokens & {"upper", "up"})
    if semantic == "lower_arm":
        return "forearm" in tokens or ("arm" in tokens and bool(tokens & {"lower", "low"}))
    if semantic == "hand":
        return bool(tokens & {"hand"})
    if semantic == "upper_leg":
        return "thigh" in tokens or ("leg" in tokens and bool(tokens & {"upper", "up"}))
    if semantic == "lower_leg":
        return bool(tokens & {"shin", "calf"}) or (
            "leg" in tokens and bool(tokens & {"lower", "low"})
        )
    if semantic == "foot":
        return bool(tokens & {"foot", "shoe"})
    return False


HUMANOID_COMPONENT_ROLES = [
    "head",
    "neck",
    "torso",
    "pelvis",
    "upper_arm.L",
    "lower_arm.L",
    "hand.L",
    "upper_arm.R",
    "lower_arm.R",
    "hand.R",
    "upper_leg.L",
    "lower_leg.L",
    "foot.L",
    "upper_leg.R",
    "lower_leg.R",
    "foot.R",
]

COARSE_HUMANOID_COMPONENT_ROLES = [
    "head",
    "torso",
    "arm.L",
    "arm.R",
    "leg.L",
    "leg.R",
]


def resolve_humanoid_components(
    mesh_objects: list[Any], component_map: dict[str, Any] | None
) -> dict[str, Any]:
    by_name = {obj.name: obj for obj in mesh_objects}
    resolved: dict[str, Any] = {}
    overrides = component_map or {}
    unknown_override_roles = sorted(set(overrides) - set(HUMANOID_COMPONENT_ROLES))
    if unknown_override_roles:
        raise ValueError(f"unknown component-map roles: {', '.join(unknown_override_roles)}")
    for role in HUMANOID_COMPONENT_ROLES:
        if role in overrides:
            object_name = overrides[role]
            if not isinstance(object_name, str) or object_name not in by_name:
                raise ValueError(f"component-map object not found for {role}: {object_name!r}")
            resolved[role] = by_name[object_name]
            continue
        matches = [obj for obj in mesh_objects if object_matches_role(obj.name, role)]
        if len(matches) != 1:
            names = ", ".join(sorted(obj.name for obj in matches)) or "none"
            raise ValueError(f"expected one mesh for {role}, found {names}")
        resolved[role] = matches[0]
    if len({obj.name for obj in resolved.values()}) != len(resolved):
        raise ValueError("one source mesh resolved to multiple humanoid component roles")
    return resolved


def object_matches_coarse_role(name: str, role: str) -> bool:
    tokens = set(normalize_name(name))
    if role == "head":
        return bool(tokens & {"head", "skull"})
    if role == "torso":
        return bool(tokens & {"torso", "body", "chest"})
    semantic, side = role.split(".")
    side_tokens = {"L": {"l", "left"}, "R": {"r", "right"}}[side]
    return semantic in tokens and bool(tokens & side_tokens)


def resolve_coarse_humanoid_components(mesh_objects: list[Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for role in COARSE_HUMANOID_COMPONENT_ROLES:
        matches = [obj for obj in mesh_objects if object_matches_coarse_role(obj.name, role)]
        if len(matches) != 1:
            names = ", ".join(sorted(obj.name for obj in matches)) or "none"
            raise ValueError(f"expected one coarse mesh for {role}, found {names}")
        resolved[role] = matches[0]
    if len({obj.name for obj in resolved.values()}) != len(resolved):
        raise ValueError("one source mesh resolved to multiple coarse humanoid roles")
    return resolved


def world_points(obj: Any) -> list[Any]:
    return [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]


def part_from_object(role: str, obj: Any) -> MeshPart:
    points = world_points(obj)
    if not points:
        raise ValueError(f"mesh object has no vertices: {obj.name}")
    center = sum(points, Vector((0.0, 0.0, 0.0))) / len(points)
    return MeshPart(role=role, object_name=obj.name, center=center, points=points)


def normalized(vector: Any, label: str) -> Any:
    if vector.length <= 1e-8:
        raise ValueError(f"cannot normalize zero vector for {label}")
    return vector.normalized()


def axis_radius(part: MeshPart, direction: Any) -> float:
    unit = normalized(direction, part.role)
    return max(abs((point - part.center).dot(unit)) for point in part.points)


def axis_endpoint(part: MeshPart, direction: Any, positive: bool) -> Any:
    unit = normalized(direction, part.role)
    sign = 1.0 if positive else -1.0
    return part.center + unit * axis_radius(part, unit) * sign


def chain_points(parts: list[MeshPart]) -> list[Any]:
    if len(parts) < 2:
        raise ValueError("a component chain requires at least two parts")
    directions = [
        normalized(parts[index + 1].center - parts[index].center, parts[index].role)
        for index in range(len(parts) - 1)
    ]
    points = [axis_endpoint(parts[0], directions[0], positive=False)]
    for index in range(len(parts) - 1):
        left = axis_endpoint(parts[index], directions[index], positive=True)
        right = axis_endpoint(parts[index + 1], directions[index], positive=False)
        points.append((left + right) * 0.5)
    points.append(axis_endpoint(parts[-1], directions[-1], positive=True))
    return points


def vec_tuple(value: Any) -> tuple[float, float, float]:
    return tuple(float(component) for component in value)


def derive_segmented_humanoid_landmarks(
    components: dict[str, Any]
) -> tuple[dict[str, tuple[float, float, float]], dict[str, list[str]]]:
    parts = {role: part_from_object(role, obj) for role, obj in components.items()}
    up = Vector((0.0, 0.0, 1.0))
    forward = Vector((0.0, 1.0, 0.0))
    pelvis = parts["pelvis"]
    torso = parts["torso"]
    neck = parts["neck"]
    head = parts["head"]
    pelvis_radius = axis_radius(pelvis, up)
    torso_top = axis_endpoint(torso, up, positive=True)
    neck_bottom = axis_endpoint(neck, up, positive=False)
    neck_top = axis_endpoint(neck, up, positive=True)
    head_bottom = axis_endpoint(head, up, positive=False)
    landmarks: dict[str, Any] = {
        "root_lower": pelvis.center - up * pelvis_radius * 0.5,
        "pelvis": pelvis.center + up * pelvis_radius * 0.5,
        "spine_mid": torso.center,
        "chest": (torso_top + neck_bottom) * 0.5,
        "neck_base": (neck_top + head_bottom) * 0.5,
        "head_top": head.center + up * axis_radius(head, up) * 0.72,
    }
    evidence: dict[str, list[str]] = {
        "root": [pelvis.object_name],
        "spine": [pelvis.object_name, torso.object_name],
        "chest": [torso.object_name, neck.object_name],
        "neck": [torso.object_name, neck.object_name, head.object_name],
        "head": [neck.object_name, head.object_name],
    }
    for side in ("L", "R"):
        arm_roles = [f"upper_arm.{side}", f"lower_arm.{side}", f"hand.{side}"]
        shoulder, elbow, wrist, hand_end = chain_points([parts[role] for role in arm_roles])
        landmarks.update(
            {
                f"shoulder.{side}": shoulder,
                f"elbow.{side}": elbow,
                f"wrist.{side}": wrist,
                f"hand_end.{side}": hand_end,
            }
        )
        evidence[f"clavicle.{side}"] = [torso.object_name, parts[arm_roles[0]].object_name]
        evidence[f"upper_arm.{side}"] = [
            parts[arm_roles[0]].object_name,
            parts[arm_roles[1]].object_name,
        ]
        evidence[f"forearm.{side}"] = [
            parts[arm_roles[0]].object_name,
            parts[arm_roles[1]].object_name,
            parts[arm_roles[2]].object_name,
        ]
        evidence[f"hand.{side}"] = [
            parts[arm_roles[1]].object_name,
            parts[arm_roles[2]].object_name,
        ]

        upper = parts[f"upper_leg.{side}"]
        lower = parts[f"lower_leg.{side}"]
        foot = parts[f"foot.{side}"]
        leg_direction = normalized(lower.center - upper.center, f"leg.{side}")
        hip = axis_endpoint(upper, leg_direction, positive=False)
        upper_distal = axis_endpoint(upper, leg_direction, positive=True)
        lower_proximal = axis_endpoint(lower, leg_direction, positive=False)
        knee = (upper_distal + lower_proximal) * 0.5
        lower_distal = axis_endpoint(lower, leg_direction, positive=True)
        foot_proximal = axis_endpoint(foot, leg_direction, positive=False)
        ankle = (lower_distal + foot_proximal) * 0.5
        toe = foot.center + forward * axis_radius(foot, forward)
        landmarks.update(
            {
                f"hip.{side}": hip,
                f"knee.{side}": knee,
                f"ankle.{side}": ankle,
                f"toe.{side}": toe,
            }
        )
        evidence[f"pelvis.{side}"] = [pelvis.object_name, upper.object_name]
        evidence[f"thigh.{side}"] = [upper.object_name, lower.object_name]
        evidence[f"shin.{side}"] = [upper.object_name, lower.object_name, foot.object_name]
        evidence[f"foot.{side}"] = [lower.object_name, foot.object_name]
    return {name: vec_tuple(value) for name, value in landmarks.items()}, evidence


def _lerp(left: Any, right: Any, factor: float) -> Any:
    return left + (right - left) * factor


def derive_coarse_humanoid_landmarks(
    components: dict[str, Any]
) -> tuple[dict[str, tuple[float, float, float]], dict[str, list[str]]]:
    """Derive a review-only skeleton from six named 3D proxy components.

    The route uses only measured component centers and extents. It is intended for block-character
    previews whose arms and legs are each one rigid mesh; it does not claim elbow, wrist, knee, or
    ankle segmentation in the source geometry.
    """
    parts = {role: part_from_object(role, obj) for role, obj in components.items()}
    up = Vector((0.0, 0.0, 1.0))
    forward = Vector((0.0, 1.0, 0.0))
    torso = parts["torso"]
    head = parts["head"]
    torso_bottom = axis_endpoint(torso, up, positive=False)
    torso_top = axis_endpoint(torso, up, positive=True)
    head_bottom = axis_endpoint(head, up, positive=False)
    head_top = axis_endpoint(head, up, positive=True)
    landmarks: dict[str, Any] = {
        "root_lower": _lerp(torso_bottom, torso_top, 0.05),
        "pelvis": _lerp(torso_bottom, torso_top, 0.22),
        "spine_mid": _lerp(torso_bottom, torso_top, 0.52),
        "chest": _lerp(torso_bottom, torso_top, 0.82),
        "neck_base": (torso_top + head_bottom) * 0.5,
        "head_top": head_top,
    }
    evidence: dict[str, list[str]] = {
        "root": [torso.object_name],
        "spine": [torso.object_name],
        "chest": [torso.object_name],
        "neck": [torso.object_name, head.object_name],
        "head": [head.object_name],
    }
    for side in ("L", "R"):
        arm = parts[f"arm.{side}"]
        arm_direction = normalized(arm.center - torso.center, f"arm.{side}")
        shoulder = axis_endpoint(arm, arm_direction, positive=False)
        hand_end = axis_endpoint(arm, arm_direction, positive=True)
        elbow = _lerp(shoulder, hand_end, 0.5)
        wrist = _lerp(shoulder, hand_end, 0.84)
        landmarks.update(
            {
                f"shoulder.{side}": shoulder,
                f"elbow.{side}": elbow,
                f"wrist.{side}": wrist,
                f"hand_end.{side}": hand_end,
            }
        )
        evidence[f"clavicle.{side}"] = [torso.object_name, arm.object_name]
        evidence[f"upper_arm.{side}"] = [arm.object_name]
        evidence[f"forearm.{side}"] = [arm.object_name]
        evidence[f"hand.{side}"] = [arm.object_name]

        leg = parts[f"leg.{side}"]
        hip = axis_endpoint(leg, up, positive=True)
        sole = axis_endpoint(leg, up, positive=False)
        knee = _lerp(hip, sole, 0.5)
        ankle = _lerp(hip, sole, 0.9)
        toe = ankle + forward * axis_radius(leg, forward)
        landmarks.update(
            {
                f"hip.{side}": hip,
                f"knee.{side}": knee,
                f"ankle.{side}": ankle,
                f"toe.{side}": toe,
            }
        )
        evidence[f"pelvis.{side}"] = [torso.object_name, leg.object_name]
        evidence[f"thigh.{side}"] = [leg.object_name]
        evidence[f"shin.{side}"] = [leg.object_name]
        evidence[f"foot.{side}"] = [leg.object_name]
    return {name: vec_tuple(value) for name, value in landmarks.items()}, evidence


def matrix_values(matrix: Any) -> list[float]:
    return [float(matrix[row][column]) for row in range(4) for column in range(4)]


def mesh_fingerprint(obj: Any) -> dict[str, Any]:
    digest = hashlib.sha256()
    digest.update(obj.name.encode("utf-8"))
    for value in matrix_values(obj.matrix_world):
        digest.update(struct.pack("<d", value))
    for vertex in obj.data.vertices:
        digest.update(struct.pack("<3d", *[float(value) for value in vertex.co]))
    for polygon in obj.data.polygons:
        digest.update(struct.pack("<I", len(polygon.vertices)))
        for index in polygon.vertices:
            digest.update(struct.pack("<I", int(index)))
    return {
        "geometryAndTransformSha256": digest.hexdigest(),
        "vertices": len(obj.data.vertices),
        "polygons": len(obj.data.polygons),
        "matrixWorld": [round(value, 9) for value in matrix_values(obj.matrix_world)],
        "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
        "parent": obj.parent.name if obj.parent else None,
        "vertexGroups": [group.name for group in obj.vertex_groups],
        "modifiers": [modifier.type for modifier in obj.modifiers],
    }


def source_snapshot(mesh_objects: list[Any]) -> dict[str, dict[str, Any]]:
    return {
        obj.name: mesh_fingerprint(obj)
        for obj in sorted(mesh_objects, key=lambda item: item.name)
    }


def create_armature(records: list[BoneRecord], profile: dict[str, Any], source_hash: str) -> Any:
    armature_data = bpy.data.armatures.new("VF_BiologicalSkeleton")
    armature_object = bpy.data.objects.new("VF_Armature", armature_data)
    bpy.context.scene.collection.objects.link(armature_object)
    armature_object.show_in_front = True
    armature_object.data.display_type = "OCTAHEDRAL"
    armature_object["vf_mode"] = "bone-only"
    armature_object["vf_profile"] = profile["profileId"]
    armature_object["vf_species_class"] = profile["speciesClass"]
    armature_object["vf_source_sha256"] = source_hash
    armature_object["vf_skin_weights"] = False
    bpy.ops.object.select_all(action="DESELECT")
    armature_object.select_set(True)
    bpy.context.view_layer.objects.active = armature_object
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones: dict[str, Any] = {}
    for record in records:
        bone = armature_data.edit_bones.new(record.name)
        bone.head = record.head
        bone.tail = record.tail
        edit_bones[record.name] = bone
    for record in records:
        bone = edit_bones[record.name]
        if record.parent:
            bone.parent = edit_bones[record.parent]
            if record.connect and (bone.head - bone.parent.tail).length <= 1e-5:
                bone.use_connect = True
    bpy.ops.object.mode_set(mode="OBJECT")
    for bone in armature_data.bones:
        bone.use_deform = False
    return armature_object


def material(name: str, color: tuple[float, float, float, float]) -> Any:
    created = bpy.data.materials.new(name)
    created.diffuse_color = color
    created.use_nodes = True
    node = created.node_tree.nodes.get("Principled BSDF")
    if node and "Base Color" in node.inputs:
        node.inputs["Base Color"].default_value = color
    if node and "Roughness" in node.inputs:
        node.inputs["Roughness"].default_value = 0.25
    return created


def move_to_collection(obj: Any, collection: Any) -> None:
    for existing in list(obj.users_collection):
        existing.objects.unlink(obj)
    collection.objects.link(obj)


def create_preview_geometry(
    records: list[BoneRecord], model_diagonal: float
) -> tuple[list[Any], Any]:
    collection = bpy.data.collections.new("VF_BonePreview")
    bpy.context.scene.collection.children.link(collection)
    bone_material = material("VF_Bone_Yellow", (1.0, 0.45, 0.02, 1.0))
    joint_material = material("VF_Joint_Cyan", (0.0, 0.82, 1.0, 1.0))
    bone_radius = max(model_diagonal * 0.007, 0.002)
    joint_radius = bone_radius * 1.65
    created: list[Any] = []
    for record in records:
        head = Vector(record.head)
        tail = Vector(record.tail)
        direction = tail - head
        length = direction.length
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=12,
            radius=bone_radius,
            depth=length,
            location=(head + tail) * 0.5,
        )
        cylinder = bpy.context.object
        cylinder.name = f"VF_Bone__{record.name}"
        cylinder.rotation_mode = "QUATERNION"
        cylinder.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(
            direction.normalized()
        )
        cylinder.data.materials.append(bone_material)
        cylinder["vf_preview_kind"] = "bone"
        cylinder["vf_bone_name"] = record.name
        move_to_collection(cylinder, collection)
        created.append(cylinder)
    unique_joints: dict[tuple[float, float, float], str] = {}
    for record in records:
        unique_joints.setdefault(
            tuple(round(value, 7) for value in record.head), record.head_landmark
        )
        unique_joints.setdefault(
            tuple(round(value, 7) for value in record.tail), record.tail_landmark
        )
    for coordinates, landmark_name in unique_joints.items():
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=2, radius=joint_radius, location=coordinates
        )
        sphere = bpy.context.object
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", landmark_name)
        sphere.name = f"VF_Joint__{safe_name}"
        sphere.data.materials.append(joint_material)
        sphere["vf_preview_kind"] = "joint"
        sphere["vf_landmark_name"] = landmark_name
        move_to_collection(sphere, collection)
        created.append(sphere)
    return created, collection


def scene_diagonal(mesh_objects: list[Any]) -> float:
    points = [point for obj in mesh_objects for point in world_points(obj)]
    minimum = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    maximum = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    return (maximum - minimum).length


def export_preview(
    path: Path, source_objects: list[Any], source_empties: list[Any], preview_objects: list[Any]
) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in [*source_empties, *source_objects, *preview_objects]:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = source_objects[0]
    result = bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_skins=False,
        export_animations=False,
        export_cameras=False,
        export_lights=False,
        export_extras=True,
        export_apply=False,
        export_yup=True,
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"glTF export failed: {result}")


def glb_json(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError(f"not a GLB file: {path}")
    offset = 12
    while offset + 8 <= len(data):
        length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        payload = data[offset : offset + length]
        offset += length
        if chunk_type == 0x4E4F534A:
            return json.loads(payload.decode("utf-8").rstrip(" \t\r\n\x00"))
    raise ValueError("GLB contains no JSON chunk")


def annotation_records(arguments: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for role, path in (
        ("front", arguments.front_annotation),
        ("side", arguments.side_annotation),
    ):
        if path is None:
            continue
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        records.append(
            {
                "role": role,
                "path": str(resolved),
                "sha256": sha256_file(resolved),
                "authority": "visual-hypothesis-only",
            }
        )
    return records


def record_payload(
    record: BoneRecord, evidence: dict[str, list[str]], route: str
) -> dict[str, Any]:
    return {
        "name": record.name,
        "parent": record.parent,
        "head": [round(value, 8) for value in record.head],
        "tail": [round(value, 8) for value in record.tail],
        "headLandmark": record.head_landmark,
        "tailLandmark": record.tail_landmark,
        "connectedToParent": record.connect,
        "useDeform": False,
        "derivation": route,
        "sourceComponents": evidence.get(record.name, []),
    }


def ensure_output_paths(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: output_dir / name for key, name in OUTPUT_NAMES.items()}
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(f"immutable skeleton outputs already exist: {collisions}")
    return paths


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    if bpy is None or Vector is None:
        raise RuntimeError("run this script with Blender Python")
    source = arguments.input.expanduser().resolve()
    if source.suffix.lower() not in {".blend", ".glb", ".gltf"} or not source.is_file():
        raise FileNotFoundError(f"input must be an existing Blend, GLB, or glTF file: {source}")
    output_dir = arguments.output_dir.expanduser().resolve()
    paths = ensure_output_paths(output_dir)
    selected_profile_path = profile_path(arguments.profile)
    profile = load_json(selected_profile_path)
    validate_profile(profile)
    annotations = annotation_records(arguments)
    source_hash_before = sha256_file(source)

    if source.suffix.lower() == ".blend":
        loaded = Path(bpy.data.filepath).resolve() if bpy.data.filepath else None
        if loaded != source:
            raise RuntimeError("the requested Blend source was not loaded before the worker script")
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        import_result = bpy.ops.import_scene.gltf(
            filepath=str(source),
            import_pack_images=False,
            import_scene_extras=True,
            import_select_created_objects=True,
        )
        if "FINISHED" not in import_result:
            raise RuntimeError(f"glTF import failed: {import_result}")
    scene_objects = list(bpy.context.scene.objects)
    mesh_objects = [obj for obj in scene_objects if obj.type == "MESH"]
    source_empties = [obj for obj in scene_objects if obj.type == "EMPTY"]
    existing_armatures = [obj.name for obj in scene_objects if obj.type == "ARMATURE"]
    if not mesh_objects:
        raise ValueError("input contains no mesh objects")
    if existing_armatures:
        raise ValueError(f"input already contains armatures: {', '.join(existing_armatures)}")
    baseline = source_snapshot(mesh_objects)

    evidence: dict[str, list[str]] = {}
    if arguments.landmarks:
        landmark_path = arguments.landmarks.expanduser().resolve()
        landmarks = landmarks_from_payload(load_json(landmark_path), profile)
        route = "explicit-3d-landmarks"
    else:
        if profile["profileId"] != "humanoid-v1":
            raise ValueError(
                "automatic component inference is available only for humanoid-v1; "
                "pass --landmarks for this profile"
            )
        component_map = (
            load_json(arguments.component_map.expanduser().resolve())
            if arguments.component_map
            else None
        )
        try:
            components = resolve_humanoid_components(mesh_objects, component_map)
            landmarks, evidence = derive_segmented_humanoid_landmarks(components)
            route = "named-segmented-components"
        except ValueError as detailed_error:
            if component_map is not None:
                raise
            try:
                coarse_components = resolve_coarse_humanoid_components(mesh_objects)
            except ValueError as coarse_error:
                raise ValueError(
                    "humanoid geometry does not satisfy the detailed or coarse component "
                    f"contracts; detailed: {detailed_error}; coarse: {coarse_error}"
                ) from coarse_error
            landmarks, evidence = derive_coarse_humanoid_landmarks(coarse_components)
            route = "coarse-segmented-humanoid-preview"
    records = bone_records(profile, landmarks)
    armature = create_armature(records, profile, source_hash_before)
    preview_objects, _ = create_preview_geometry(records, scene_diagonal(mesh_objects))
    after = source_snapshot(mesh_objects)

    export_preview(paths["preview"], mesh_objects, source_empties, preview_objects)
    bpy.ops.wm.save_as_mainfile(filepath=str(paths["blend"]), compress=True)
    source_hash_after = sha256_file(source)
    preview_document = glb_json(paths["preview"])

    geometry_unchanged = baseline == after
    no_armature_modifiers = all(
        "ARMATURE" not in snapshot["modifiers"] for snapshot in after.values()
    )
    no_added_weights = all(
        after[name]["vertexGroups"] == baseline[name]["vertexGroups"] for name in baseline
    )
    no_mesh_armature_parenting = all(obj.parent != armature for obj in mesh_objects)
    expected_names = [record.name for record in records]
    actual_names = [bone.name for bone in armature.data.bones]
    hierarchy_matches = actual_names == expected_names and all(
        (
            armature.data.bones[record.name].parent.name
            if armature.data.bones[record.name].parent
            else None
        )
        == record.parent
        for record in records
    )
    all_non_deforming = all(not bone.use_deform for bone in armature.data.bones)
    no_skins = not preview_document.get("skins")
    no_animations = not preview_document.get("animations")
    checks = {
        "sourceHashUnchanged": source_hash_before == source_hash_after,
        "sourceGeometryAndSceneStateUnchanged": geometry_unchanged,
        "noArmatureModifiersAdded": no_armature_modifiers,
        "noWeightsAdded": no_added_weights,
        "noMeshArmatureParenting": no_mesh_armature_parenting,
        "boneHierarchyMatchesProfile": hierarchy_matches,
        "allBonesNonDeforming": all_non_deforming,
        "previewContainsNoSkins": no_skins,
        "previewContainsNoAnimations": no_animations,
    }
    passed = all(checks.values())
    qa = {
        "schemaVersion": 1,
        "mode": "bone-only",
        "profileId": profile["profileId"],
        "route": route,
        "passed": passed,
        "checks": checks,
        "source": {
            "path": str(source),
            "sha256": source_hash_before,
            "meshCount": len(mesh_objects),
            "armatureCountBefore": 0,
            "snapshot": baseline,
        },
        "output": {
            "armatureCount": 1,
            "boneCount": len(records),
            "previewMeshCount": len(preview_objects),
            "blendSha256": sha256_file(paths["blend"]),
            "previewGlbSha256": sha256_file(paths["preview"]),
            "previewGlbSkins": len(preview_document.get("skins", [])),
            "previewGlbAnimations": len(preview_document.get("animations", [])),
        },
    }
    atomic_write_json(paths["qa"], qa)
    skeleton = {
        "schemaVersion": 1,
        "kind": "biological-skeleton",
        "mode": "bone-only",
        "state": "pendingUserSignoff" if passed else "failed",
        "profileId": profile["profileId"],
        "speciesClass": profile["speciesClass"],
        "route": route,
        "coordinateSystem": {
            "space": "Blender world",
            "right": "+X",
            "forward": "+Y",
            "up": "+Z",
            "unit": "source glTF unit",
        },
        "constraints": {
            "skinWeights": False,
            "armatureModifiers": False,
            "meshParentingToArmature": False,
            "geometryMutation": False,
            "animation": False,
        },
        "source": {
            "path": str(source),
            "sha256": source_hash_before,
            "meshObjects": sorted(obj.name for obj in mesh_objects),
        },
        "annotations": annotations,
        "landmarks": {
            name: [round(value, 8) for value in coordinates]
            for name, coordinates in sorted(landmarks.items())
        },
        "bones": [record_payload(record, evidence, route) for record in records],
        "artifacts": {
            "blend": paths["blend"].name,
            "previewGlb": paths["preview"].name,
            "qa": paths["qa"].name,
        },
    }
    atomic_write_json(paths["skeleton"], skeleton)
    if not passed:
        raise RuntimeError(f"skeleton QA failed; inspect {paths['qa']}")
    return {
        "passed": True,
        "profileId": profile["profileId"],
        "route": route,
        "boneCount": len(records),
        "outputs": {key: str(path) for key, path in paths.items()},
    }


def main() -> int:
    blender_args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    result = run(parse_arguments(blender_args))
    print("VF_SKELETON_RESULT", json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
