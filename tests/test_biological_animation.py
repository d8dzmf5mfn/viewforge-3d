from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "plugins" / "viewforge-3d-toolkit" / "skills" / "animate-biological-skeleton"
SCRIPTS = SKILL / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMMON = load_module("vf_animation_common", SCRIPTS / "_animation_common.py")
EXTRACT = load_module("vf_extract_pose_coordinates", SCRIPTS / "extract_pose_coordinates.py")


def animation_profile() -> dict:
    return json.loads((SKILL / "assets" / "humanoid-wave-v1.animation.json").read_text())


def binding_profile() -> dict:
    return json.loads((SKILL / "assets" / "segmented-humanoid-v1.bind.json").read_text())


def test_bundled_animation_profile_is_valid() -> None:
    profile = animation_profile()

    COMMON.validate_animation_profile(profile)

    assert profile["schedule"][0]["pose"] == "rest"
    assert profile["schedule"][-1]["pose"] == "rest"
    assert profile["chain"]["bones"] == ["upper_arm.R", "forearm.R", "hand.R"]


def test_animation_profile_rejects_non_rest_endpoint() -> None:
    profile = animation_profile()
    profile["schedule"][-1]["pose"] = "outward"

    with pytest.raises(ValueError, match="start and end at rest"):
        COMMON.validate_animation_profile(profile)


def test_animation_profile_rejects_hidden_depth_contract() -> None:
    profile = animation_profile()
    profile["coordinateSystem"]["depth"] = "inferred"

    with pytest.raises(ValueError, match="fixed-front coordinate contract"):
        COMMON.validate_animation_profile(profile)


def test_bundled_binding_profile_is_valid_and_complete() -> None:
    profile = binding_profile()

    COMMON.validate_binding_profile(profile)

    assert profile["segmented"] is True
    assert profile["skinWeights"] is False
    assert set(profile["animatedComponents"]) <= set(profile["components"])


def test_binding_profile_rejects_continuous_mesh_claim() -> None:
    profile = binding_profile()
    profile["segmented"] = False

    with pytest.raises(ValueError, match="segmented=true"):
        COMMON.validate_binding_profile(profile)


def test_pose_assignments_are_exact_and_unique(tmp_path: Path) -> None:
    raised = tmp_path / "raised.png"
    end = tmp_path / "end.png"

    parsed = EXTRACT.parse_pose_assignments([f"raised={raised}", f"end={end}"])

    assert parsed == {"raised": raised.resolve(), "end": end.resolve()}
    with pytest.raises(ValueError, match="duplicate pose"):
        EXTRACT.parse_pose_assignments([f"raised={raised}", f"raised={end}"])


def test_image_to_blender_direction_flips_horizontal_axis() -> None:
    start = {"x": 100.0, "y": 100.0}
    end = {"x": 140.0, "y": 70.0}

    payload = EXTRACT.vector_payload(start, end)

    assert payload["unitImageXZ"][0] > 0
    assert payload["unitBlenderXYZ"][0] < 0
    assert payload["unitBlenderXYZ"][1] == 0
    assert payload["unitBlenderXYZ"][2] > 0


def test_animation_profile_rejects_duplicate_schedule_frames() -> None:
    profile = copy.deepcopy(animation_profile())
    profile["schedule"][2]["frame"] = profile["schedule"][1]["frame"]

    with pytest.raises(ValueError, match="strictly increasing"):
        COMMON.validate_animation_profile(profile)


def test_multi_chain_interaction_contract_is_bundled() -> None:
    contract = (SKILL / "references" / "multi-chain-interaction-contract.md").read_text()

    assert "Bone Action" in contract
    assert "Character trajectory Action" in contract
    assert "Prop Action" in contract
    assert "maximum per-frame local bone angle" in contract
    assert "pendingUserSignoff" in contract
