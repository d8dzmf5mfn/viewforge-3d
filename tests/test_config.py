from pathlib import Path

from face3d.config import load_config


def test_default_config_contract() -> None:
    config = load_config(Path("configs/face-v1.yaml"))
    assert config.profile == "face-v1"
    assert config.mode == "pixel-direct"
    assert config.pixel.grid_size == 256
    assert config.sdf.resolution == 256
    assert (
        config.mesh.minimum_triangles
        <= config.mesh.target_triangles
        <= config.mesh.maximum_triangles
    )
    assert config.resolve_asset(config.assets.flame_model).is_absolute()
    assert not config.output.geometry_only


def test_anime_pixel_direct_config_is_geometry_only() -> None:
    config = load_config(Path("configs/anime-pixel-direct-geometry.yaml"))
    assert config.profile == "face-v1"
    assert config.mode == "pixel-direct"
    assert config.output.geometry_only
    assert config.acceptance.peak_memory_gb_max == 12.0


def test_face_v2_config_contract() -> None:
    config = load_config(Path("configs/face-v2.yaml"))
    assert config.is_v2
    assert config.profile == "face-v2"
    assert config.mode == "pixel-flame-hybrid"
    assert config.sdf.resolution == 384
    assert config.pixel.maximum_cells == 200000
    assert config.skin.uv_method == "xatlas"
    assert config.assets.flame_masks is not None
    assert config.assets.flame_prepared is not None


def test_face_v3_template_contract() -> None:
    config = load_config(Path("configs/face-v3.yaml"))
    assert config.is_v3
    assert config.uses_refined_landmarks
    assert config.profile == "face-v3"
    assert config.mode == "template-head-v0"
    assert config.assets.template_head is not None
    assert config.assets.template_landmarks is not None
    assert config.assets.template_manifest is not None
    assert config.fit.inversion_barrier_weight > 0
    assert config.input.maximum_jaw_open_score == 0.20
    assert config.input.minimum_face_detection_confidence == 0.65


def test_face_v3_anime_preview_uses_geometric_mouth_gate() -> None:
    config = load_config(Path("configs/face-v3-anime-preview.yaml"))
    assert config.is_v3
    assert config.mode == "template-head-v0"
    assert config.input.maximum_mouth_gap_ratio == 0.045
    assert config.input.maximum_jaw_open_score == 1.0
    assert config.input.minimum_face_detection_confidence == 0.10
    assert config.input.minimum_face_presence_confidence == 0.10
    assert config.input.minimum_face_tracking_confidence == 0.10
    assert config.input.mask_method == "white-background"
