from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "plugins" / "viewforge-3d-toolkit" / "skills" / "build-biological-skeleton"
SCRIPT = SKILL / "scripts" / "build_biological_skeleton.py"
SPEC = importlib.util.spec_from_file_location("build_biological_skeleton", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("profile_id", ["humanoid-v1", "quadruped-v1"])
def test_bundled_profiles_are_valid(profile_id: str) -> None:
    profile = json.loads((SKILL / "assets" / f"{profile_id}.profile.json").read_text())

    MODULE.validate_profile(profile)

    assert profile["profileId"] == profile_id
    assert all(bone["name"] and bone["head"] != bone["tail"] for bone in profile["bones"])


def test_bone_records_reject_zero_length_bones() -> None:
    profile = {
        "schemaVersion": 1,
        "profileId": "fixture-v1",
        "speciesClass": "fixture",
        "requiredLandmarks": ["a", "b"],
        "bones": [
            {"name": "fixture", "parent": None, "head": "a", "tail": "b", "connect": False}
        ],
    }
    MODULE.validate_profile(profile)

    with pytest.raises(ValueError, match="zero or invalid length"):
        MODULE.bone_records(profile, {"a": (0.0, 0.0, 0.0), "b": (0.0, 0.0, 0.0)})


@pytest.mark.parametrize(
    ("name", "role"),
    [
        ("arm-left-upper", "upper_arm.L"),
        ("RightForearm", "lower_arm.R"),
        ("leg_left_lower", "lower_leg.L"),
        ("shoe-right", "foot.R"),
        ("Pelvis", "pelvis"),
    ],
)
def test_component_role_aliases(name: str, role: str) -> None:
    assert MODULE.object_matches_role(name, role)


@pytest.mark.parametrize(
    ("name", "role"),
    [
        ("Head", "head"),
        ("Torso", "torso"),
        ("Arm_L", "arm.L"),
        ("RightArm", "arm.R"),
        ("Leg-L", "leg.L"),
        ("right_leg", "leg.R"),
    ],
)
def test_coarse_component_role_aliases(name: str, role: str) -> None:
    assert MODULE.object_matches_coarse_role(name, role)


def test_coarse_humanoid_landmarks_cover_profile_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Vec:
        def __init__(self, values: tuple[float, float, float]) -> None:
            self.values = tuple(float(value) for value in values)

        def __iter__(self):
            return iter(self.values)

        def __getitem__(self, index: int) -> float:
            return self.values[index]

        def __add__(self, other):
            return Vec(tuple(self[index] + other[index] for index in range(3)))

        def __sub__(self, other):
            return Vec(tuple(self[index] - other[index] for index in range(3)))

        def __mul__(self, scalar: float):
            return Vec(tuple(value * scalar for value in self.values))

        __rmul__ = __mul__

        def __truediv__(self, scalar: float):
            return Vec(tuple(value / scalar for value in self.values))

        @property
        def length(self) -> float:
            return math.sqrt(sum(value * value for value in self.values))

        def normalized(self):
            return self / self.length

        def dot(self, other) -> float:
            return sum(self[index] * other[index] for index in range(3))

    def part(role: str, name: str, center: tuple[float, float, float], half) -> object:
        measured_center = Vec(center)
        points = [
            measured_center + Vec((x * half[0], y * half[1], z * half[2]))
            for x in (-1, 1)
            for y in (-1, 1)
            for z in (-1, 1)
        ]
        return MODULE.MeshPart(
            role=role,
            object_name=name,
            center=measured_center,
            points=points,
        )

    parts = {
        "head": part("head", "Head", (0, 0, 1.575), (0.225, 0.225, 0.225)),
        "torso": part("torso", "Torso", (0, 0, 1.075), (0.25, 0.15, 0.325)),
        "arm.L": part("arm.L", "Arm_L", (-0.575, 0, 1.25), (0.325, 0.11, 0.11)),
        "arm.R": part("arm.R", "Arm_R", (0.575, 0, 1.25), (0.325, 0.11, 0.11)),
        "leg.L": part("leg.L", "Leg_L", (-0.13, 0, 0.375), (0.12, 0.12, 0.375)),
        "leg.R": part("leg.R", "Leg_R", (0.13, 0, 0.375), (0.12, 0.12, 0.375)),
    }
    components = {role: object() for role in parts}
    monkeypatch.setattr(MODULE, "Vector", Vec)
    monkeypatch.setattr(
        MODULE,
        "part_from_object",
        lambda role, object_: parts[role],
    )

    landmarks, evidence = MODULE.derive_coarse_humanoid_landmarks(components)
    profile = json.loads((SKILL / "assets" / "humanoid-v1.profile.json").read_text())
    records = MODULE.bone_records(profile, landmarks)

    assert set(landmarks) == set(profile["requiredLandmarks"])
    assert len(records) == len(profile["bones"])
    assert landmarks["shoulder.L"][0] > landmarks["hand_end.L"][0]
    assert landmarks["shoulder.R"][0] < landmarks["hand_end.R"][0]
    assert evidence["forearm.L"] == ["Arm_L"]


def test_landmark_payload_requires_matching_profile() -> None:
    profile = json.loads((SKILL / "assets" / "humanoid-v1.profile.json").read_text())

    with pytest.raises(ValueError, match="does not match"):
        MODULE.landmarks_from_payload(
            {"schemaVersion": 1, "profileId": "quadruped-v1", "landmarks": {}}, profile
        )
