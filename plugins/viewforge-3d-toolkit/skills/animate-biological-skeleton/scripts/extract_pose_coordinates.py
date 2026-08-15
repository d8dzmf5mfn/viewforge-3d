#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from _animation_common import (
    angle_degrees,
    atomic_write_json,
    load_json,
    sha256_file,
    validate_animation_profile,
)
from PIL import Image


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract reviewed root-relative directions from Imagegen skeleton overlays."
    )
    parser.add_argument("--skeleton", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument(
        "--pose",
        required=True,
        action="append",
        metavar="NAME=IMAGE",
        help="Repeat once for every required pose image.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def parse_pose_assignments(values: list[str]) -> dict[str, Path]:
    assignments: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"invalid --pose assignment: {value!r}")
        if name in assignments:
            raise ValueError(f"duplicate pose assignment: {name}")
        assignments[name] = Path(raw_path).expanduser().resolve()
    return assignments


def cyan_components(
    path: Path, minimum_area: int, minimum_count: int
) -> tuple[tuple[int, int], list[dict[str, float]]]:
    image = np.asarray(Image.open(path).convert("RGB"))
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    mask = (
        (red <= 90)
        & (green >= 145)
        & (blue >= 145)
        & (np.abs(green.astype(np.int16) - blue.astype(np.int16)) <= 105)
    )
    ys, xs = np.nonzero(mask)
    remaining = {(int(x), int(y)) for x, y in zip(xs, ys, strict=True)}
    components: list[dict[str, float]] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        points = [seed]
        while stack:
            x, y = stack.pop()
            for dx, dy in (
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ):
                neighbor = (x + dx, y + dy)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    points.append(neighbor)
        if len(points) < minimum_area:
            continue
        components.append(
            {
                "x": sum(point[0] for point in points) / len(points),
                "y": sum(point[1] for point in points) / len(points),
                "area": float(len(points)),
            }
        )
    components.sort(key=lambda item: (item["y"], item["x"]))
    if len(components) < minimum_count:
        raise ValueError(
            f"expected at least {minimum_count} separate cyan markers in {path}, "
            f"found {len(components)}"
        )
    return (int(image.shape[1]), int(image.shape[0])), components


def vector_payload(start: dict[str, float], end: dict[str, float]) -> dict[str, Any]:
    dx = end["x"] - start["x"]
    dy_up = start["y"] - end["y"]
    length = math.hypot(dx, dy_up)
    if length <= 1e-6:
        raise ValueError("coincident joint markers")
    unit_x = dx / length
    unit_z = dy_up / length
    blender_x = -unit_x
    return {
        "pixels": [round(dx, 4), round(dy_up, 4)],
        "lengthPixels": round(length, 4),
        "unitImageXZ": [round(unit_x, 8), round(unit_z, 8)],
        "unitBlenderXYZ": [round(blender_x, 8), 0.0, round(unit_z, 8)],
        "angleDegreesFromPositiveX": round(math.degrees(math.atan2(unit_z, unit_x)), 4),
        "angleDegreesBlenderXZ": round(math.degrees(math.atan2(unit_z, blender_x)), 4),
    }


def extract_chain(
    path: Path,
    *,
    image_side: str,
    forearm_fraction: float,
    marker_config: dict[str, Any],
) -> dict[str, Any]:
    (width, height), markers = cyan_components(
        path,
        int(marker_config["minimumAreaPixels"]),
        int(marker_config["minimumMarkerCount"]),
    )
    center_x = width * 0.5
    y_min, y_max = (float(value) for value in marker_config["shoulderYRange"])
    center_offset = width * float(marker_config["shoulderCenterOffset"])
    outward_offset = width * float(marker_config["chainOutwardOffset"])
    if image_side == "viewer-left":
        shoulder_candidates = [
            marker
            for marker in markers
            if marker["x"] < center_x - center_offset
            and height * y_min <= marker["y"] <= height * y_max
        ]
        shoulder_selector = max
    else:
        shoulder_candidates = [
            marker
            for marker in markers
            if marker["x"] > center_x + center_offset
            and height * y_min <= marker["y"] <= height * y_max
        ]
        shoulder_selector = min
    if not shoulder_candidates:
        raise ValueError(f"could not locate {image_side} shoulder in {path}")
    shoulder = shoulder_selector(shoulder_candidates, key=lambda item: item["x"])
    if image_side == "viewer-left":
        chain_candidates = [
            marker for marker in markers if marker["x"] < shoulder["x"] - outward_offset
        ]
    else:
        chain_candidates = [
            marker for marker in markers if marker["x"] > shoulder["x"] + outward_offset
        ]
    if len(chain_candidates) < 2:
        raise ValueError(f"could not locate a three-joint moving chain in {path}")
    chain_candidates.sort(
        key=lambda item: math.hypot(item["x"] - shoulder["x"], item["y"] - shoulder["y"])
    )
    middle = chain_candidates[0]
    chain_end = chain_candidates[-1]
    distal_split = {
        "x": middle["x"] + (chain_end["x"] - middle["x"]) * forearm_fraction,
        "y": middle["y"] + (chain_end["y"] - middle["y"]) * forearm_fraction,
        "area": 0.0,
    }
    proximal_direction = vector_payload(shoulder, middle)
    distal_direction = vector_payload(middle, chain_end)
    proximal_scale = float(proximal_direction["lengthPixels"])

    def normalized_point(point: dict[str, float]) -> list[float]:
        return [
            round((point["x"] - shoulder["x"]) / proximal_scale, 8),
            round((shoulder["y"] - point["y"]) / proximal_scale, 8),
        ]

    return {
        "image": {
            "path": str(path),
            "sha256": sha256_file(path),
            "width": width,
            "height": height,
        },
        "detectedMarkerCount": len(markers),
        "detectedMarkersPixels": [
            [round(marker["x"], 4), round(marker["y"], 4)] for marker in markers
        ],
        "movingChainPixels": {
            "root": [round(shoulder["x"], 4), round(shoulder["y"], 4)],
            "middle": [round(middle["x"], 4), round(middle["y"], 4)],
            "distalSplitFromRestLengthRatio": [
                round(distal_split["x"], 4),
                round(distal_split["y"], 4),
            ],
            "end": [round(chain_end["x"], 4), round(chain_end["y"], 4)],
        },
        "relativeToRootProximalUnits": {
            "root": [0.0, 0.0],
            "middle": normalized_point(middle),
            "distalSplit": normalized_point(distal_split),
            "end": normalized_point(chain_end),
        },
        "proximalDirection": proximal_direction,
        "distalDirection": distal_direction,
    }


def bone_vector(bone: dict[str, Any]) -> tuple[float, float, float]:
    head = bone["head"]
    tail = bone["tail"]
    vector = tuple(float(tail[index]) - float(head[index]) for index in range(3))
    if (
        not all(math.isfinite(value) for value in vector)
        or math.sqrt(sum(value * value for value in vector)) <= 1e-8
    ):
        raise ValueError(f"invalid rest vector for bone {bone.get('name')}")
    return vector


def main() -> int:
    arguments = parse_arguments()
    skeleton_path = arguments.skeleton.expanduser().resolve()
    profile_path = arguments.profile.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    poses = parse_pose_assignments(arguments.pose)
    for path in (skeleton_path, profile_path, *poses.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)

    skeleton = load_json(skeleton_path)
    profile = load_json(profile_path)
    validate_animation_profile(profile)
    required = set(profile["requiredPoseImages"])
    if set(poses) != required:
        raise ValueError(
            f"pose assignments must exactly match {sorted(required)}, got {sorted(poses)}"
        )
    bones_by_name = {bone["name"]: bone for bone in skeleton.get("bones", [])}
    chain_names = list(profile["chain"]["bones"])
    missing_bones = [name for name in chain_names if name not in bones_by_name]
    if missing_bones:
        raise ValueError(f"skeleton lacks profile bones: {missing_bones}")
    first, second, third = (bones_by_name[name] for name in chain_names)
    if second.get("parent") != first["name"] or third.get("parent") != second["name"]:
        raise ValueError("animation profile bones must form a connected parent chain")
    lengths = [math.dist(bone["head"], bone["tail"]) for bone in (first, second, third)]
    if any(not math.isfinite(length) or length <= 1e-8 for length in lengths):
        raise ValueError("profile chain contains zero or invalid rest length")
    forearm_fraction = lengths[1] / (lengths[1] + lengths[2])
    frames = {
        name: extract_chain(
            path,
            image_side=profile["chain"]["imageSide"],
            forearm_fraction=forearm_fraction,
            marker_config=profile["markerDetection"],
        )
        for name, path in sorted(poses.items())
    }

    rest_reference = frames[profile["restReferencePose"]]
    rest_proximal = bone_vector(first)
    rest_distal = tuple(
        float(third["tail"][index]) - float(second["head"][index]) for index in range(3)
    )
    proximal_error = angle_degrees(
        tuple(rest_reference["proximalDirection"]["unitBlenderXYZ"]), rest_proximal
    )
    distal_error = angle_degrees(
        tuple(rest_reference["distalDirection"]["unitBlenderXYZ"]), rest_distal
    )
    max_rest_error = max(proximal_error, distal_error)
    rest_tolerance = float(profile["qa"]["maxRestReferenceAngleDegrees"])
    if max_rest_error > rest_tolerance:
        raise ValueError(
            f"ending pose does not match Blender rest projection: {max_rest_error:.4f} degrees"
        )

    payload = {
        "schemaVersion": 1,
        "kind": "imagegen-relative-bone-pose",
        "state": "derived-not-user-accepted",
        "authority": {
            "imagegen": "2D pose hypothesis",
            "skeleton": "bone hierarchy and segment lengths",
            "depth": "unobserved; directions constrained to Blender XZ plane",
        },
        "source": {
            "skeleton": str(skeleton_path),
            "skeletonSha256": sha256_file(skeleton_path),
            "profile": str(profile_path),
            "profileSha256": sha256_file(profile_path),
            "profileId": profile["profileId"],
        },
        "animation": {
            "actionName": profile["actionName"],
            "fps": profile["fps"],
            "interpolation": profile["interpolation"],
            "schedule": profile["schedule"],
            "qa": profile["qa"],
        },
        "coordinateSystem": profile["coordinateSystem"],
        "retarget": {
            "movingBones": chain_names,
            "secondBoneFractionOfDistalImageSegment": round(forearm_fraction, 8),
            "preserveBlenderRestLengths": True,
            "translations": False,
            "scales": False,
        },
        "frames": frames,
        "extractionQa": {
            "passed": True,
            "restReferencePose": profile["restReferencePose"],
            "proximalRestReferenceErrorDegrees": round(proximal_error, 8),
            "distalRestReferenceErrorDegrees": round(distal_error, 8),
            "maxRestReferenceErrorDegrees": round(max_rest_error, 8),
            "toleranceDegrees": rest_tolerance,
        },
    }
    atomic_write_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "profile": profile["profileId"],
                "frames": sorted(frames),
                "passed": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
