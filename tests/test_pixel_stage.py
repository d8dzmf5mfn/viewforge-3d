import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from face3d.config import load_config
from face3d.io import atomic_write_json, sha256_file
from face3d.models import ViewRole
from face3d.package import package_run
from face3d.pixel_binary import decode_header
from face3d.report import build_report, write_notices
from face3d.stages.pixel_direct import run_pixel_direct


def test_pixel_direct_stage_emits_traceable_cells_and_closed_mesh(tmp_path: Path) -> None:
    config = load_config(Path("configs/face-v1.yaml"))
    material_sources = (
        config.resolve_asset(config.skin.uv_albedo_source),
        config.resolve_asset(config.skin.micro_albedo_source),
    )
    if any(not path.is_file() for path in material_sources):
        pytest.skip("optional local material reference assets are not installed")

    run = tmp_path / "run"
    for directory in (
        "working/landmarks",
        "working/masks",
        "references",
        "overlays",
        "models",
        "pixels",
    ):
        (run / directory).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    count = 468
    local_x = rng.uniform(-0.34, 0.34, count)
    local_y = rng.uniform(-0.46, 0.46, count)
    local_z = 0.04 + 0.20 * np.exp(-((local_x / 0.13) ** 2 + (local_y / 0.24) ** 2))
    local_x[234], local_x[454] = -0.34, 0.34
    local_y[10], local_y[152] = -0.5, 0.5
    local_x[1], local_y[1], local_z[1] = 0.0, 0.0, 0.30
    for index in (93, 132, 234, 323, 361, 454):
        local_z[index] = 0.02

    image_size = 512
    image_scale = 0.6
    views = []
    for role, yaw in (
        (ViewRole.FRONT, 0.0),
        (ViewRole.LEFT45, -45.0),
        (ViewRole.RIGHT45, 45.0),
    ):
        theta = -math.radians(yaw)
        projected_x = math.cos(theta) * local_x + math.sin(theta) * local_z
        landmarks = np.zeros((count, 3), dtype=np.float32)
        landmarks[:, 0] = 0.5 + projected_x * image_scale
        landmarks[:, 1] = 0.5 + local_y * image_scale
        landmark_path = run / "working" / "landmarks" / f"{role.value}.npz"
        np.savez_compressed(
            landmark_path,
            all=landmarks,
            ibug68=landmarks[:68, :2] * image_size,
            image_size=np.asarray([image_size, image_size], dtype=np.int32),
        )
        y, x = np.mgrid[:image_size, :image_size]
        normalized_x = (x - image_size * 0.5) / (image_size * image_scale)
        normalized_y = (y - image_size * 0.5) / (image_size * image_scale)
        if role == ViewRole.FRONT:
            mask = (normalized_x / 0.37) ** 2 + (normalized_y / 0.52) ** 2 <= 1
        else:
            dense_y, dense_x = np.mgrid[-0.52:0.52:300j, -0.37:0.37:260j]
            inside = (dense_x / 0.37) ** 2 + (dense_y / 0.52) ** 2 <= 1
            dense_z = 0.04 + 0.20 * np.exp(-((dense_x / 0.13) ** 2 + (dense_y / 0.24) ** 2))
            side_x = (math.cos(theta) * dense_x + math.sin(theta) * dense_z)[inside]
            side_y = dense_y[inside]
            points = np.rint(
                np.column_stack(
                    (
                        image_size * 0.5 + side_x * image_size * image_scale,
                        image_size * 0.5 + side_y * image_size * image_scale,
                    )
                )
            ).astype(np.int32)
            binary = np.zeros((image_size, image_size), dtype=np.uint8)
            cv2.fillConvexPoly(binary, cv2.convexHull(points), 255)
            mask = binary > 0
        binary_mask = mask.astype(np.uint8) * 255
        mask_path = run / "working" / "masks" / f"{role.value}.png"
        cv2.imwrite(str(mask_path), binary_mask)
        rgb = np.full((image_size, image_size, 3), 235, dtype=np.uint8)
        rgb[mask] = np.asarray([190, 155, 135], dtype=np.uint8)
        reference = run / "references" / f"{role.value}.png"
        Image.fromarray(rgb).save(reference)
        Image.fromarray(rgb).save(run / "overlays" / f"landmarks-{role.value}.png")
        Image.fromarray(rgb).save(run / "overlays" / f"silhouette-{role.value}.png")
        views.append(
            {
                "role": role.value,
                "source_path": str(reference),
                "normalized_path": str(reference),
                "width": image_size,
                "height": image_size,
                "sha256": sha256_file(reference),
                "normalized_sha256": sha256_file(reference),
                "landmarks_path": str(landmark_path),
                "mask_path": str(mask_path),
                "pose_deg": {"yaw": yaw, "pitch": 0.0, "roll": 0.0},
                "sharpness": 100.0,
                "mouth_gap_ratio": 0.0,
                "mask_coverage": 1.0,
                "mask_confirmed": True,
                "warnings": [],
            }
        )
    atomic_write_json(
        run / "working" / "intake.json",
        {"schemaVersion": 1, "profile": "face-v1", "views": views, "maskConfirmed": True},
    )
    config = config.model_copy(
        update={
            "pixel": config.pixel.model_copy(
                update={
                    "grid_size": 96,
                    "coarse_depth_grid": 18,
                    "complex_region_radius_pixels": 4.0,
                }
            ),
            "mesh": config.mesh.model_copy(
                update={
                    "target_triangles": 8_000,
                    "minimum_triangles": 5_000,
                    "maximum_triangles": 12_000,
                }
            ),
            "skin": config.skin.model_copy(
                update={
                    "atlas_resolution": 256,
                    "detail_resolution": 128,
                    # The randomized compact fixture only exposes a sparse
                    # subset of vertices to its synthetic cameras. Production
                    # keeps the 0.25 observation gate from face-v1.yaml.
                    "minimum_observed_vertex_fraction": 0.05,
                }
            ),
            # This compact topology fixture has no anatomical neck or cranium
            # ground truth; it verifies traceability and closed-mesh behavior.
            "acceptance": config.acceptance.model_copy(
                update={
                    "silhouette_iou_drop_max": 0.01,
                    "normal_variance_reduction_min": 0.20,
                    "hausdorff_voxels_max": 40.0,
                }
            ),
        }
    )
    result = run_pixel_direct(run, config)
    pixel_metrics = json.loads((run / "working" / "sdf-metrics.json").read_text())
    header = decode_header((run / "pixels" / "pixels.bin").read_bytes())
    assert result["mode"] == "pixel-direct"
    assert (
        pixel_metrics["earReconstruction"]["mode"]
        == "smooth-anatomical-support-with-direct-side-view-skin"
    )
    assert header["recordCount"] == result["instanceCount"]
    assert result["complexPixelCount"] > 0
    assert pixel_metrics["surfaceCellCoverage"] >= 0.95
    assert pixel_metrics["frontSurfaceMaxDistancePixels"] <= 2.0
    assert (run / "models" / "voxels.glb").stat().st_size > 1_000
    assert (run / "models" / "smooth.glb").stat().st_size > 1_000
    assert (run / "models" / "skin.glb").stat().st_size > 1_000
    assert (run / "textures" / "skin-atlas.jpg").stat().st_size > 1_000
    manifest, report = build_report(
        run,
        config,
        {"elapsedSeconds": 1.0, "peakRssBytes": 256 * 1024**2, "deterministic": True},
    )
    write_notices(run)
    package = tmp_path / "synthetic.viewforge3d"
    packaged = package_run(run, package, config)
    assert manifest["mode"] == "pixel-direct"
    assert manifest["pixel"]["binarySha256"]
    assert report["summary"]["automatedGatesPassed"]
    assert packaged["bytes"] < 25 * 1024**2
