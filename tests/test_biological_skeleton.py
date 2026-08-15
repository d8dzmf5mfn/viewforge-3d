from __future__ import annotations

import importlib.util
import json
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


def test_landmark_payload_requires_matching_profile() -> None:
    profile = json.loads((SKILL / "assets" / "humanoid-v1.profile.json").read_text())

    with pytest.raises(ValueError, match="does not match"):
        MODULE.landmarks_from_payload(
            {"schemaVersion": 1, "profileId": "quadruped-v1", "landmarks": {}}, profile
        )
