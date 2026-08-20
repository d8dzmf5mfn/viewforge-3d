#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


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


def angle_degrees(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("angle vectors must have the same non-zero dimension")
    left_length = math.sqrt(sum(value * value for value in left))
    right_length = math.sqrt(sum(value * value for value in right))
    if left_length <= 1e-12 or right_length <= 1e-12:
        raise ValueError("angle vectors must be non-zero")
    dot = sum(a * b for a, b in zip(left, right, strict=True)) / (left_length * right_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def validate_animation_profile(profile: dict[str, Any]) -> None:
    if profile.get("schemaVersion") != 1:
        raise ValueError("animation profile schemaVersion must be 1")
    if profile.get("kind") != "three-bone-direction-animation":
        raise ValueError("unsupported animation profile kind")
    if not isinstance(profile.get("profileId"), str) or not profile["profileId"]:
        raise ValueError("animation profile requires profileId")
    if not isinstance(profile.get("actionName"), str) or not profile["actionName"]:
        raise ValueError("animation profile requires actionName")
    fps = profile.get("fps")
    if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
        raise ValueError("animation profile fps must be a positive integer")

    chain = profile.get("chain")
    if not isinstance(chain, dict):
        raise ValueError("animation profile requires chain")
    bones = chain.get("bones")
    if (
        not isinstance(bones, list)
        or len(bones) != 3
        or len(set(bones)) != 3
        or not all(isinstance(name, str) and name for name in bones)
    ):
        raise ValueError("chain.bones must contain exactly three unique bone names")
    if chain.get("imageSide") not in {"viewer-left", "viewer-right"}:
        raise ValueError("chain.imageSide must be viewer-left or viewer-right")

    required = profile.get("requiredPoseImages")
    if (
        not isinstance(required, list)
        or not required
        or len(set(required)) != len(required)
        or not all(isinstance(name, str) and name and name != "rest" for name in required)
    ):
        raise ValueError("requiredPoseImages must contain unique non-rest names")
    rest_reference = profile.get("restReferencePose")
    if rest_reference not in required:
        raise ValueError("restReferencePose must name a required pose image")

    schedule = profile.get("schedule")
    if not isinstance(schedule, list) or len(schedule) < 2:
        raise ValueError("animation profile requires at least two schedule entries")
    frames: list[int] = []
    for entry in schedule:
        if not isinstance(entry, dict):
            raise ValueError("schedule entries must be objects")
        frame = entry.get("frame")
        pose = entry.get("pose")
        if not isinstance(frame, int) or isinstance(frame, bool) or frame <= 0:
            raise ValueError("schedule frames must be positive integers")
        if not isinstance(pose, str) or (pose != "rest" and pose not in required):
            raise ValueError(f"schedule references unknown pose: {pose!r}")
        frames.append(frame)
    if frames != sorted(frames) or len(set(frames)) != len(frames):
        raise ValueError("schedule frames must be strictly increasing and unique")
    if schedule[0]["pose"] != "rest" or schedule[-1]["pose"] != "rest":
        raise ValueError("animation schedule must start and end at rest")
    scheduled_poses = {entry["pose"] for entry in schedule if entry["pose"] != "rest"}
    missing_scheduled = scheduled_poses - set(required)
    if missing_scheduled:
        raise ValueError(f"schedule poses lack required images: {sorted(missing_scheduled)}")

    coordinates = profile.get("coordinateSystem")
    expected_coordinates = {
        "projection": "front-orthographic",
        "imageRight": "-Blender X",
        "imageUp": "+Blender Z",
        "depth": "Y=0",
    }
    if not isinstance(coordinates, dict) or any(
        coordinates.get(key) != value for key, value in expected_coordinates.items()
    ):
        raise ValueError("profile must use the supported fixed-front coordinate contract")

    marker = profile.get("markerDetection")
    if not isinstance(marker, dict):
        raise ValueError("animation profile requires markerDetection")
    y_range = marker.get("shoulderYRange")
    if (
        not isinstance(y_range, list)
        or len(y_range) != 2
        or not all(isinstance(value, (int, float)) for value in y_range)
        or not 0 <= float(y_range[0]) < float(y_range[1]) <= 1
    ):
        raise ValueError("markerDetection.shoulderYRange must be two increasing ratios")
    for key in ("minimumAreaPixels", "minimumMarkerCount"):
        value = marker.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"markerDetection.{key} must be a positive integer")
    for key in ("shoulderCenterOffset", "chainOutwardOffset"):
        value = marker.get(key)
        if not isinstance(value, (int, float)) or not 0 < float(value) < 0.5:
            raise ValueError(f"markerDetection.{key} must be between zero and 0.5")

    interpolation = profile.get("interpolation")
    if not isinstance(interpolation, dict) or interpolation.get("type") != "BEZIER":
        raise ValueError("only BEZIER interpolation is supported")
    if interpolation.get("handleType") != "AUTO_CLAMPED":
        raise ValueError("only AUTO_CLAMPED handles are supported")
    qa = profile.get("qa")
    if not isinstance(qa, dict):
        raise ValueError("animation profile requires qa tolerances")
    for key in (
        "maxDirectionErrorDegrees",
        "maxRestReferenceAngleDegrees",
        "maxInterframeAngularStepDegrees",
        "transformTolerance",
    ):
        value = qa.get(key)
        if not isinstance(value, (int, float)) or float(value) <= 0:
            raise ValueError(f"qa.{key} must be positive")


def validate_binding_profile(profile: dict[str, Any]) -> None:
    if profile.get("schemaVersion") != 1:
        raise ValueError("binding profile schemaVersion must be 1")
    if profile.get("mode") != "rigid-segmented-bone-parent":
        raise ValueError("unsupported binding mode")
    if profile.get("segmented") is not True:
        raise ValueError("rigid binding requires explicit segmented=true")
    if profile.get("skinWeights") is not False:
        raise ValueError("no-skin binding requires skinWeights=false")
    if profile.get("armatureModifiers") is not False:
        raise ValueError("no-skin binding requires armatureModifiers=false")
    components = profile.get("components")
    if not isinstance(components, dict) or not components:
        raise ValueError("binding profile requires a non-empty components map")
    if not all(
        isinstance(component, str) and component and isinstance(bone, str) and bone
        for component, bone in components.items()
    ):
        raise ValueError("component and bone names must be non-empty strings")
    animated = profile.get("animatedComponents")
    if (
        not isinstance(animated, list)
        or not animated
        or len(set(animated)) != len(animated)
        or not set(animated).issubset(components)
    ):
        raise ValueError("animatedComponents must be a non-empty subset of components")
    if not isinstance(profile.get("expectedActionName"), str) or not profile["expectedActionName"]:
        raise ValueError("binding profile requires expectedActionName")
    qa = profile.get("qa")
    if not isinstance(qa, dict):
        raise ValueError("binding profile requires qa tolerances")
    for key in ("transformTolerance", "minimumAnimatedMotion"):
        value = qa.get(key)
        if not isinstance(value, (int, float)) or float(value) <= 0:
            raise ValueError(f"qa.{key} must be positive")
