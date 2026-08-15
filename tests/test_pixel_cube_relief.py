import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from face3d.pixel_cube import PixelCubeSpec, PixelCuboidSpec
from face3d.pixel_cube_relief import (
    FRONT_RELIEF_SOURCE_BITS,
    FrontFaceReliefSpec,
    _curvature_gate,
    _write_front_relief_asset,
    apply_front_relief,
    estimate_front_relief,
)
from face3d.stages.intake import FACE_OVAL
from face3d.stages.pixel_direct import EYE_INDICES, MOUTH_INDICES


def _synthetic_landmarks() -> np.ndarray:
    landmarks = np.zeros((468, 3), dtype=np.float64)
    golden = math.pi * (3 - math.sqrt(5))
    for index in range(len(landmarks)):
        radius = math.sqrt(((index * 71) % 467 + 1) / 468) * 0.92
        angle = index * golden
        landmarks[index] = (
            0.5 + 0.23 * radius * math.cos(angle),
            0.52 + 0.30 * radius * math.sin(angle),
            -0.11 + 0.34 * radius**1.7,
        )

    for position, index in enumerate(FACE_OVAL):
        angle = -math.pi / 2 + 2 * math.pi * position / len(FACE_OVAL)
        landmarks[index] = (
            0.5 + 0.25 * math.cos(angle),
            0.52 + 0.31 * math.sin(angle),
            0.23,
        )

    for center_x, indices in ((0.395, EYE_INDICES[:16]), (0.605, EYE_INDICES[16:])):
        for position, index in enumerate(indices):
            angle = math.pi + 2 * math.pi * position / len(indices)
            landmarks[index] = (
                center_x + 0.055 * math.cos(angle),
                0.42 + 0.017 * math.sin(angle),
                -0.005,
            )

    for position, index in enumerate(MOUTH_INDICES):
        angle = math.pi + 2 * math.pi * position / len(MOUTH_INDICES)
        landmarks[index] = (
            0.5 + 0.09 * math.cos(angle),
            0.68 + 0.026 * math.sin(angle),
            -0.005,
        )

    landmarks[1] = (0.5, 0.54, -0.13)
    landmarks[33] = (0.34, 0.42, 0.02)
    landmarks[133] = (0.45, 0.42, -0.01)
    landmarks[159] = (0.395, 0.405, -0.01)
    landmarks[145] = (0.395, 0.435, 0.00)
    landmarks[362] = (0.55, 0.42, -0.01)
    landmarks[263] = (0.66, 0.42, 0.02)
    landmarks[386] = (0.605, 0.405, -0.01)
    landmarks[374] = (0.605, 0.435, 0.00)
    landmarks[61] = (0.41, 0.68, 0.01)
    landmarks[291] = (0.59, 0.68, 0.01)
    landmarks[13] = (0.5, 0.67, -0.02)
    landmarks[14] = (0.5, 0.69, -0.01)
    landmarks[98] = (0.46, 0.58, -0.05)
    landmarks[327] = (0.54, 0.58, -0.05)
    landmarks[2] = (0.5, 0.59, -0.07)
    landmarks[205] = (0.39, 0.55, 0.00)
    landmarks[425] = (0.61, 0.55, 0.00)
    return landmarks


def _synthetic_rgb() -> np.ndarray:
    image = Image.new("RGB", (320, 320), (190, 151, 126))
    draw = ImageDraw.Draw(image)
    draw.ellipse((102, 126, 142, 143), fill=(58, 43, 39))
    draw.ellipse((178, 126, 218, 143), fill=(58, 43, 39))
    draw.ellipse((151, 166, 169, 182), fill=(78, 51, 43))
    draw.line((132, 217, 188, 217), fill=(70, 37, 39), width=6)
    return np.asarray(image)


@pytest.fixture
def relief_spec() -> FrontFaceReliefSpec:
    return FrontFaceReliefSpec(
        cube=PixelCubeSpec(side_length_m=0.2, cells_per_edge=20),
        front_cells_xy=(20, 20),
        border_rim_cells_xy=(1, 1),
        coarse_depth_grid=8,
        complex_region_radius_pixels=1.0,
        max_inset_m=0.02,
    )


def test_front_relief_is_inward_only_and_preserves_cube_edges(
    relief_spec: FrontFaceReliefSpec,
) -> None:
    scan = estimate_front_relief(_synthetic_rgb(), _synthetic_landmarks(), relief_spec)
    geometry = apply_front_relief(scan, relief_spec)
    indices = geometry.grid_xyz
    base = geometry.base_positions
    positions = geometry.positions

    assert scan.face_mask.shape == (20, 20)
    assert 120 < np.count_nonzero(scan.face_mask) < 280
    assert not scan.face_mask[0].any()
    assert not scan.face_mask[-1].any()
    assert not scan.face_mask[:, 0].any()
    assert not scan.face_mask[:, -1].any()
    assert np.all(scan.indentation_m[~scan.face_mask] == 0)
    assert np.all(scan.indentation_m[scan.face_mask] > 0)
    assert float(np.max(scan.indentation_m)) <= 0.02 + 1e-12

    front = indices[:, 2] == 19
    np.testing.assert_allclose(positions[~front], base[~front])
    np.testing.assert_allclose(positions[:, :2], base[:, :2])
    assert np.all(positions[front, 2] <= base[front, 2])
    assert np.all(positions[front, 2] >= base[front, 2] - 0.02)
    affected = geometry.source_bits == FRONT_RELIEF_SOURCE_BITS
    assert np.count_nonzero(affected) == np.count_nonzero(scan.face_mask)
    assert np.all(front[affected])
    np.testing.assert_allclose(geometry.confidence[affected], 0.42)

    face_edge = (indices[:, 0] == 0) | (indices[:, 0] == 19)
    face_edge |= (indices[:, 1] == 0) | (indices[:, 1] == 19)
    np.testing.assert_allclose(positions[front & face_edge], base[front & face_edge])


def test_default_front_scan_uses_head_proportioned_cuboid_and_stays_in_budget() -> None:
    spec = FrontFaceReliefSpec()

    assert isinstance(spec.cube, PixelCuboidSpec)
    assert spec.cube.cells_xyz == (86, 128, 107)
    assert spec.cube.dimensions_m == pytest.approx((0.16125, 0.24, 0.200625))
    assert spec.front_cells_xy == (344, 512)
    assert spec.coarse_depth_grid == 80
    assert spec.complex_region_radius_pixels == pytest.approx(20.0)
    assert spec.front_cell_pitch_m == pytest.approx(0.00046875)
    assert spec.hybrid_surface_cell_count == 225_296
    assert spec.hybrid_surface_cell_count <= spec.maximum_cells == 360_000


def test_hybrid_front_subdivision_preserves_coarse_side_and_back_cells() -> None:
    spec = FrontFaceReliefSpec()
    scan = estimate_front_relief(_synthetic_rgb(), _synthetic_landmarks(), spec)
    geometry = apply_front_relief(scan, spec)

    assert len(geometry.grid_xyz) == 225_296
    coarse_shape = np.asarray((86, 128, 107), dtype=np.uint16)
    fine_shape = np.asarray((344, 512, 1), dtype=np.uint16)
    assert {
        tuple(value) for value in np.unique(geometry.grid_shape_xyz, axis=0).tolist()
    } == {tuple(coarse_shape), tuple(fine_shape)}
    np.testing.assert_allclose(
        np.unique(geometry.cell_pitch_m),
        [0.00046875, 0.001875],
    )

    coarse = np.all(geometry.grid_shape_xyz == coarse_shape, axis=1)
    coarse_front = coarse & (geometry.grid_xyz[:, 2] == 106)
    assert np.all(
        (geometry.grid_xyz[coarse_front, 0] == 0)
        | (geometry.grid_xyz[coarse_front, 0] == 85)
        | (geometry.grid_xyz[coarse_front, 1] == 0)
        | (geometry.grid_xyz[coarse_front, 1] == 127)
    )
    np.testing.assert_allclose(
        geometry.positions[coarse],
        geometry.base_positions[coarse],
    )

    fine = np.all(geometry.grid_shape_xyz == fine_shape, axis=1)
    assert np.all(geometry.grid_xyz[fine, 2] == 0)
    assert geometry.grid_xyz[fine, 0].min() == 4
    assert geometry.grid_xyz[fine, 0].max() == 339
    assert geometry.grid_xyz[fine, 1].min() == 4
    assert geometry.grid_xyz[fine, 1].max() == 507


def test_hires_scan_has_distinct_eye_nose_and_mouth_relief() -> None:
    scan = estimate_front_relief(
        _synthetic_rgb(),
        _synthetic_landmarks(),
        FrontFaceReliefSpec(),
    )
    gate = _curvature_gate(scan)

    assert gate["passed"], gate
    assert scan.feature_metrics["meanEyeSocketContrastMeters"] >= 0.0012
    assert scan.feature_metrics["noseProjectionContrastMeters"] >= 0.0018
    assert scan.feature_metrics["nostrilContrastMeters"] >= 0.0010
    assert scan.feature_metrics["mouthSeamContrastMeters"] >= 0.0012
    assert float(np.min(scan.feature_relief_m)) <= -0.0020
    assert float(np.max(scan.feature_relief_m)) >= 0.0020


def test_nose_remains_shallower_than_surrounding_face(
    relief_spec: FrontFaceReliefSpec,
) -> None:
    landmarks = _synthetic_landmarks()
    scan = estimate_front_relief(_synthetic_rgb(), landmarks, relief_spec)
    source_bounds = np.asarray(scan.mapping["sourceFaceBoundsNormalized"])
    target_bounds = np.asarray(scan.mapping["targetFaceBoundsCells"])
    source_min = source_bounds[:2]
    source_span = source_bounds[2:] - source_bounds[:2]
    target_min = target_bounds[:2]
    target_span = target_bounds[2:] - target_bounds[:2]
    target_nose = target_min + (landmarks[1, :2] - source_min) / source_span * target_span
    nose_x = int(np.clip(round(target_nose[0] - 0.5), 0, 19))
    nose_y = int(np.clip(round(20 - target_nose[1] - 0.5), 0, 19))

    assert scan.face_mask[nose_y, nose_x]
    assert scan.indentation_m[nose_y, nose_x] < np.median(
        scan.indentation_m[scan.face_mask]
    )


def test_front_relief_asset_writes_traceability_and_preview(
    tmp_path: Path,
    relief_spec: FrontFaceReliefSpec,
) -> None:
    source = tmp_path / "front.png"
    Image.fromarray(_synthetic_rgb()).save(source)
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"test-model")
    scan = estimate_front_relief(_synthetic_rgb(), _synthetic_landmarks(), relief_spec)

    result = _write_front_relief_asset(
        source,
        model,
        tmp_path / "asset",
        _synthetic_rgb(),
        scan,
        relief_spec,
    )

    assert result["frontReliefCellCount"] == np.count_nonzero(scan.face_mask)
    manifest = json.loads((tmp_path / "asset" / "manifest.json").read_text())
    assert manifest["assetType"] == "3d-pixel-cuboid-front-face-relief"
    assert manifest["frontRelief"]["modifiedFace"] == "zMax"
    assert manifest["frontRelief"]["inwardOnly"] is True
    assert manifest["frontRelief"]["otherFacesUnchanged"] is True
    assert manifest["frontRelief"]["exactMetricDepthMeasured"] is False

    records = [
        json.loads(line)
        for line in (tmp_path / "asset" / "pixels" / "cells.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2168
    observed = [record for record in records if record["observedFrom2D"]]
    assert len(observed) == np.count_nonzero(scan.face_mask)
    assert all(record["gridXYZ"][2] == 19 for record in observed)
    assert all(record["sourceBits"] == FRONT_RELIEF_SOURCE_BITS for record in observed)
    assert all(record["indentationMeters"] > 0 for record in observed)

    with Image.open(tmp_path / "asset" / "preview" / "front-relief-16x9.png") as preview:
        assert preview.size == (1600, 900)
