from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from face3d.unified_head import EyeballAsset, UnifiedHeadAsset, geometry_hash


def test_unified_head_roundtrip_and_glb_nodes(tmp_path: Path) -> None:
    skin = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    vertices = np.asarray(skin.vertices, dtype=np.float64)
    faces = np.asarray(skin.faces, dtype=np.int64)
    render_to_skin = np.arange(len(vertices), dtype=np.int64)
    uv = np.column_stack(
        (
            0.5 + np.arctan2(vertices[:, 0], vertices[:, 2]) / (2 * np.pi),
            0.5 - np.arcsin(vertices[:, 1]) / np.pi,
        )
    ).astype(np.float32)
    digest = geometry_hash(vertices, faces)
    eye = EyeballAsset(
        center=np.asarray([-0.3, 0.15, 0.82]),
        radius=0.12,
        gaze=np.asarray([0.0, 0.0, 1.0]),
    )
    asset = UnifiedHeadAsset(
        skin_vertices=vertices,
        skin_faces=faces,
        render_to_skin=render_to_skin,
        render_faces=faces,
        uv=uv,
        regions={"left_ear": np.asarray([0]), "right_ear": np.asarray([1])},
        left_eye=eye,
        right_eye=EyeballAsset(-eye.center * np.asarray([1, -1, -1]), 0.12, eye.gaze),
        geometry_sha256=digest,
        anatomy={
            "schemaVersion": "2.0.0",
            "route": {"topology": "template-head-v0-continuous-head-neck"},
        },
    )
    saved = tmp_path / "unified-head.npz"
    asset.save(saved)
    restored = UnifiedHeadAsset.load(saved)
    assert restored.geometry_sha256 == digest
    assert np.array_equal(restored.render_faces, faces)
    glb = tmp_path / "head.glb"
    restored.export_head_glb(glb, Image.new("RGB", (32, 32), (180, 130, 110)))
    scene = trimesh.load(glb, force="scene")
    node_names = set(scene.graph.nodes)
    assert {"HeadSkin", "Eyeball.L", "Eyeball.R"} <= node_names
    assert scene.geometry["HeadSkin"].metadata["topology"] == (
        "template-head-v0-continuous-head-neck"
    )
