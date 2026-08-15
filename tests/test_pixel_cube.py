import json
import struct
from pathlib import Path

import pytest

from face3d.pixel_cube import PixelCubeSpec, create_pixel_cube, surface_cell_indices


def _glb_document(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", payload, 0)
    assert (magic, version, total) == (b"glTF", 2, len(payload))
    json_length = struct.unpack_from("<I", payload, 12)[0]
    return json.loads(payload[20 : 20 + json_length])


def test_surface_cell_indices_form_six_face_shell_without_interior() -> None:
    indices = surface_cell_indices(20)
    assert len(indices) == 20**3 - 18**3 == 2168
    assert ((indices == 0) | (indices == 19)).any(axis=1).all()


def test_create_pixel_cube_writes_20_cm_traceable_asset(tmp_path: Path) -> None:
    result = create_pixel_cube(tmp_path, PixelCubeSpec(cells_per_edge=20))

    assert result["sideLengthCentimeters"] == pytest.approx(20.0)
    assert result["cellPitchCentimeters"] == pytest.approx(1.0)
    assert result["instanceCount"] == 2168

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["assetType"] == "3d-pixel-cube"
    assert manifest["primaryAsset"] is True
    assert manifest["geometry"]["nominalDimensionsMeters"] == [0.2, 0.2, 0.2]
    assert manifest["pixel"]["surfaceOnly"] is True
    assert manifest["pixel"]["instanceCount"] == 2168
    assert manifest["pixel"]["sourceBits"] == 8

    records = (tmp_path / "pixels" / "cells.jsonl").read_text().splitlines()
    assert len(records) == 2168
    first = json.loads(records[0])
    last = json.loads(records[-1])
    assert first == {
        "cellId": 0,
        "confidence": 0.25,
        "faceMask": 21,
        "gridXYZ": [0, 0, 0],
        "positionMeters": [-0.095, -0.095, -0.095],
        "sourceBits": 8,
    }
    assert last["gridXYZ"] == [19, 19, 19]
    assert last["faceMask"] == 42

    document = _glb_document(tmp_path / "models" / "voxels.glb")
    node = document["nodes"][0]
    assert node["name"] == "surface-3d-pixels"
    assert node["extras"]["instanceCount"] == 2168
    assert node["extras"]["voxelSize"] == pytest.approx(0.01)
    assert node["extras"]["fillRatio"] == pytest.approx(0.92)


@pytest.mark.parametrize(
    "spec",
    [
        PixelCubeSpec(side_length_m=0.2, cells_per_edge=2),
    ],
)
def test_minimum_valid_cube_spec(spec: PixelCubeSpec) -> None:
    assert spec.surface_cell_count == 8


def test_default_cube_uses_first_version_like_pixel_density() -> None:
    spec = PixelCubeSpec()
    assert spec.cells_per_edge == 128
    assert spec.cell_pitch_m == pytest.approx(0.0015625)
    assert spec.surface_cell_count == 96_776


def test_invalid_cube_specs_fail_closed() -> None:
    with pytest.raises(ValueError):
        PixelCubeSpec(side_length_m=0)
    with pytest.raises(ValueError):
        PixelCubeSpec(cells_per_edge=1)
