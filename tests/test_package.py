import json
from pathlib import Path

from PIL import Image

from face3d.config import load_config
from face3d.io import sha256_file
from face3d.models import ViewRole
from face3d.package import package_run


def test_package_is_deterministic(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for directory in ("models", "pixels", "qa", "references", "overlays", "textures"):
        (run / directory).mkdir(parents=True, exist_ok=True)
    (run / "models" / "voxels.glb").write_bytes(b"voxel")
    (run / "models" / "smooth.glb").write_bytes(b"smooth")
    (run / "models" / "skin.glb").write_bytes(b"skin")
    (run / "textures" / "skin-atlas.jpg").write_bytes(b"atlas")
    (run / "textures" / "skin-confidence.png").write_bytes(b"confidence")
    (run / "pixels" / "pixels.bin").write_bytes(b"pixels")
    (run / "pixels" / "schema.json").write_text("{}")
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "schemaVersion": "1.0.0",
                "mode": "pixel-direct",
                "pixel": {
                    "binary": "pixels/pixels.bin",
                    "binarySha256": sha256_file(run / "pixels" / "pixels.bin"),
                    "schema": "pixels/schema.json",
                    "schemaSha256": sha256_file(run / "pixels" / "schema.json"),
                },
            }
        )
    )
    (run / "qa" / "report.json").write_text("{}")
    for role in ViewRole:
        for relative in (
            f"references/{role.value}.png",
            f"overlays/landmarks-{role.value}.png",
            f"overlays/silhouette-{role.value}.png",
            f"qa/fixed-view-{role.value}.png",
        ):
            Image.new("RGB", (8, 8), (120, 130, 140)).save(run / relative)
    first = tmp_path / "first.face3d"
    second = tmp_path / "second.face3d"
    config = load_config(Path("configs/face-v1.yaml"))
    package_run(run, first, config)
    package_run(run, second, config)
    assert first.read_bytes() == second.read_bytes()


def test_package_v2_uses_single_head_geometry(tmp_path: Path) -> None:
    run = tmp_path / "run-v2"
    for directory in ("models", "pixels", "qa", "references", "overlays", "textures"):
        (run / directory).mkdir(parents=True, exist_ok=True)
    (run / "models" / "voxels.glb").write_bytes(b"voxel-v2")
    (run / "models" / "head.glb").write_bytes(b"head-v2")
    (run / "textures" / "head-confidence.png").write_bytes(b"confidence-v2")
    (run / "textures" / "head-source.png").write_bytes(b"source-v2")
    (run / "pixels" / "pixels.bin").write_bytes(b"pixels-v2")
    (run / "pixels" / "schema.json").write_text("{}")
    geometry_hash = "01" * 32
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "schemaVersion": "2.0.0",
                "mode": "pixel-flame-hybrid",
                "mesh": {"geometryHash": geometry_hash},
                "skin": {
                    "geometryHash": geometry_hash,
                    "skinGeometryHash": geometry_hash,
                    "neutralGeometryHash": geometry_hash,
                    "maximumVertexDifference": 0.0,
                    "modelSha256": sha256_file(run / "models" / "head.glb"),
                },
                "pixel": {
                    "binary": "pixels/pixels.bin",
                    "binarySha256": sha256_file(run / "pixels" / "pixels.bin"),
                    "schema": "pixels/schema.json",
                    "schemaSha256": sha256_file(run / "pixels" / "schema.json"),
                },
            }
        )
    )
    (run / "qa" / "report.json").write_text("{}")
    (run / "qa" / "anatomy.json").write_text("{}")
    for role in ViewRole:
        for relative in (
            f"references/{role.value}.png",
            f"overlays/landmarks-{role.value}.png",
            f"overlays/silhouette-{role.value}.png",
            f"qa/fixed-view-{role.value}.png",
            f"qa/registration-{role.value}.png",
        ):
            Image.new("RGB", (8, 8), (120, 130, 140)).save(run / relative)
    output = tmp_path / "v2.face3d"
    package_run(run, output, load_config(Path("configs/face-v2.yaml")))
    assert output.is_file()
