from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from face3d.coordinate_unit_surface import connect_depth_envelope_units
from face3d.glb import export_colored_coordinate_unit_surface


def _colors(shape: tuple[int, int, int]) -> np.ndarray:
    return np.full((*shape, 3), (220, 230, 238), dtype=np.uint8)


def test_depth_envelope_joins_unit_coordinates_without_exporting_cubes() -> None:
    occupancy = np.ones((2, 2, 2), dtype=bool)
    surface = connect_depth_envelope_units(
        occupancy,
        np.zeros(3, dtype=np.float64),
        0.2,
        _colors(occupancy.shape),
    )

    assert surface.vertices.shape == (8, 3)
    assert surface.faces.shape == (12, 3)
    assert surface.metrics["watertight"] is True
    assert surface.metrics["windingConsistent"] is True
    assert surface.metrics["positiveVolume"] is True
    assert surface.metrics["boundaryEdgeCount"] == 0
    assert surface.metrics["nonmanifoldEdgeCount"] == 0
    assert surface.metrics["unitCoordinateMaximumError"] <= 1e-12
    assert surface.metrics["sourceUnitPositionsMoved"] is False
    assert surface.metrics["surfaceCoordinateInterpolationApplied"] is False
    assert surface.metrics["visibleUnitCubesExported"] is False
    assert surface.metrics["geometricSmoothingApplied"] is False
    assert surface.metrics["marchingCubesApplied"] is False


def test_single_layer_columns_are_excluded_instead_of_becoming_degenerate_faces() -> None:
    occupancy = np.ones((2, 2, 2), dtype=bool)
    occupancy[0, 0, 1] = False
    surface = connect_depth_envelope_units(
        occupancy,
        np.zeros(3, dtype=np.float64),
        0.2,
        _colors(occupancy.shape),
    )

    assert surface.metrics["excludedSingleLayerColumnCount"] == 1
    assert surface.metrics["degenerateTriangleCount"] == 0
    assert surface.metrics["watertight"] is True


def test_coordinate_unit_glb_is_indexed_quartz_surface(tmp_path: Path) -> None:
    occupancy = np.ones((2, 2, 2), dtype=bool)
    surface = connect_depth_envelope_units(
        occupancy,
        np.zeros(3, dtype=np.float64),
        0.2,
        _colors(occupancy.shape),
    )
    destination = tmp_path / "coordinate-units.glb"
    export_colored_coordinate_unit_surface(
        surface.vertices,
        surface.faces,
        surface.vertex_normals,
        surface.vertex_colors.astype(np.float32) / 255.0,
        destination,
        mapping="test direct coordinate-unit connection",
        topology_sidecar="working/coordinate-unit-surface.npz",
    )

    payload = destination.read_bytes()
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20 : 20 + json_length])
    node = document["nodes"][0]
    primitive = document["meshes"][0]["primitives"][0]
    position = document["accessors"][primitive["attributes"]["POSITION"]]
    indices = document["accessors"][primitive["indices"]]
    assert position["count"] == len(surface.vertices)
    assert indices["count"] == len(surface.faces) * 3
    assert node["extras"]["visibleUnitCubesExported"] is False
    assert node["extras"]["geometrySmoothingApplied"] is False
    assert node["extras"]["normalInterpolationApplied"] is True
    assert node["extras"]["marchingCubesApplied"] is False
    assert document["materials"][0]["name"] == "polished-milky-quartz"


def test_coordinate_unit_glb_preserves_baseline_contrast_vertex_colors(
    tmp_path: Path,
) -> None:
    occupancy = np.ones((2, 2, 2), dtype=bool)
    colors = _colors(occupancy.shape)
    colors[0, 0, 1] = (9, 14, 26)
    surface = connect_depth_envelope_units(
        occupancy,
        np.zeros(3, dtype=np.float64),
        0.2,
        colors,
    )
    destination = tmp_path / "contrast-connected.glb"
    export_colored_coordinate_unit_surface(
        surface.vertices,
        surface.faces,
        surface.vertex_normals,
        surface.vertex_colors.astype(np.float32) / 255.0,
        destination,
        mapping="test contrast-preserving direct connection",
        topology_sidecar="working/connected-surface.npz",
        material_profile="quality-baseline-contrast",
    )

    payload = destination.read_bytes()
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20 : 20 + json_length])
    primitive = document["meshes"][0]["primitives"][0]
    material = document["materials"][0]
    assert "COLOR_0" in primitive["attributes"]
    assert document["extensionsUsed"] == []
    assert material["name"] == "quality-baseline-contrast-connected-surface"
    assert material["pbrMetallicRoughness"] == {
        "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
        "metallicFactor": 0.0,
        "roughnessFactor": 0.64,
    }
    assert material["extras"]["vertexColorRole"] == ("preserve facial-feature and ear contrast")
    assert material["extras"]["qaShadowColorFactor"] == [0.11, 0.15, 0.22]


def test_coordinate_unit_glb_can_reduce_contrast_surface_roughness(tmp_path: Path) -> None:
    occupancy = np.ones((2, 2, 2), dtype=bool)
    surface = connect_depth_envelope_units(
        occupancy,
        np.zeros(3, dtype=np.float64),
        0.2,
        _colors(occupancy.shape),
    )
    destination = tmp_path / "smooth-contrast-connected.glb"
    export_colored_coordinate_unit_surface(
        surface.vertices,
        surface.faces,
        surface.vertex_normals,
        surface.vertex_colors.astype(np.float32) / 255.0,
        destination,
        mapping="test lower-roughness direct connection",
        topology_sidecar="working/connected-surface.npz",
        material_profile="quality-baseline-contrast-smooth",
    )

    payload = destination.read_bytes()
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20 : 20 + json_length])
    material = document["materials"][0]
    assert material["name"] == "quality-baseline-contrast-smooth-connected-surface"
    assert material["pbrMetallicRoughness"]["roughnessFactor"] == 0.32
    assert material["extras"]["vertexColorRole"] == ("preserve facial-feature and ear contrast")
