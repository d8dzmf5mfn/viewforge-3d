from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from face3d.glb import export_colored_connected_surface
from face3d.voxel_surface import (
    connect_voxel_surface,
    repair_nonmanifold_voxel_contacts,
)


def _colors(shape: tuple[int, int, int]) -> np.ndarray:
    return np.full((*shape, 3), (190, 200, 215), dtype=np.uint8)


def test_single_voxel_connects_to_exact_watertight_cube() -> None:
    occupancy = np.ones((1, 1, 1), dtype=bool)
    surface = connect_voxel_surface(
        occupancy,
        np.zeros(3, dtype=np.float64),
        0.2,
        _colors(occupancy.shape),
    )

    assert surface.vertices.shape == (8, 3)
    assert surface.faces.shape == (12, 3)
    assert surface.metrics["exposedQuadCount"] == 6
    assert surface.metrics["watertight"] is True
    assert surface.metrics["windingConsistent"] is True
    assert surface.metrics["boundaryEdgeCount"] == 0
    assert surface.metrics["nonmanifoldEdgeCount"] == 0
    assert surface.metrics["gridCornerMaximumError"] <= 1e-12
    assert surface.metrics["vertexSmoothingApplied"] is False
    assert surface.metrics["marchingCubesApplied"] is False


def test_adjacent_voxels_remove_internal_faces_and_share_grid_vertices() -> None:
    occupancy = np.ones((2, 1, 1), dtype=bool)
    surface = connect_voxel_surface(
        occupancy,
        np.zeros(3, dtype=np.float64),
        0.2,
        _colors(occupancy.shape),
    )

    assert surface.vertices.shape == (12, 3)
    assert surface.metrics["exposedQuadCount"] == 10
    assert surface.metrics["triangleCount"] == 20
    assert surface.metrics["removedInternalCellFaces"] == 2
    assert surface.metrics["componentCount"] == 1
    assert surface.metrics["watertight"] is True
    assert surface.metrics["boundaryEdgeCount"] == 0


def test_diagonal_contact_repair_adds_grid_cell_without_moving_vertices() -> None:
    occupancy = np.zeros((2, 2, 1), dtype=bool)
    occupancy[0, 0, 0] = True
    occupancy[1, 1, 0] = True
    source_surface = connect_voxel_surface(
        occupancy,
        np.zeros(3, dtype=np.float64),
        0.2,
        _colors(occupancy.shape),
    )
    assert source_surface.metrics["nonmanifoldEdgeCount"] == 1

    repaired, colors, record = repair_nonmanifold_voxel_contacts(
        occupancy,
        np.zeros(3, dtype=np.float64),
        0.2,
        _colors(occupancy.shape),
    )
    repaired_surface = connect_voxel_surface(
        repaired,
        np.zeros(3, dtype=np.float64),
        0.2,
        colors,
    )

    assert np.all(repaired[occupancy])
    assert record["addedCellCount"] == 1
    assert record["vertexPositionsMoved"] == 0
    assert record["smoothingApplied"] is False
    assert repaired_surface.metrics["nonmanifoldEdgeCount"] == 0
    assert repaired_surface.metrics["watertight"] is True
    assert repaired_surface.metrics["gridCornerMaximumError"] <= 1e-12


def test_connected_surface_glb_keeps_flat_faces_without_smoothing(tmp_path: Path) -> None:
    occupancy = np.ones((2, 1, 1), dtype=bool)
    surface = connect_voxel_surface(
        occupancy,
        np.zeros(3, dtype=np.float64),
        0.2,
        _colors(occupancy.shape),
    )
    output = tmp_path / "connected.glb"
    export_colored_connected_surface(
        surface.vertices,
        surface.faces,
        surface.face_colors.astype(np.float32) / 255.0,
        output,
        mapping="test direct connection",
        topology_sidecar="working/connected-surface.npz",
    )

    payload = output.read_bytes()
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20 : 20 + json_length])
    node = document["nodes"][0]
    primitive = document["meshes"][0]["primitives"][0]
    position = document["accessors"][primitive["attributes"]["POSITION"]]
    assert position["count"] == len(surface.faces) * 3
    assert node["extras"]["canonicalVertexCount"] == len(surface.vertices)
    assert node["extras"]["renderVerticesDuplicatedForFlatNormals"] is True
    assert node["extras"]["vertexSmoothingApplied"] is False
    assert node["extras"]["marchingCubesApplied"] is False
    assert "KHR_materials_specular" in document["extensionsUsed"]


def test_connected_surface_glb_exports_physical_quartz_material(tmp_path: Path) -> None:
    occupancy = np.ones((1, 1, 1), dtype=bool)
    surface = connect_voxel_surface(
        occupancy,
        np.zeros(3, dtype=np.float64),
        0.2,
        _colors(occupancy.shape),
    )
    output = tmp_path / "quartz.glb"
    export_colored_connected_surface(
        surface.vertices,
        surface.faces,
        surface.face_colors.astype(np.float32) / 255.0,
        output,
        mapping="test quartz connection",
        topology_sidecar="working/connected-surface.npz",
        material_profile="polished-milky-quartz",
    )

    payload = output.read_bytes()
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20 : 20 + json_length])
    material = document["materials"][0]
    assert document["nodes"][0]["extras"]["materialProfile"] == "polished-milky-quartz"
    assert material["name"] == "polished-milky-quartz"
    assert material["pbrMetallicRoughness"]["metallicFactor"] == 0.0
    assert material["pbrMetallicRoughness"]["roughnessFactor"] == 0.22
    assert material["extensions"]["KHR_materials_ior"]["ior"] == 1.544
    assert material["extensions"]["KHR_materials_transmission"]["transmissionFactor"] == 0.10
    assert material["extensions"]["KHR_materials_volume"]["attenuationColor"] == [
        0.82,
        0.91,
        1.0,
    ]
