import json
import struct
from pathlib import Path

import numpy as np
import pytest

from face3d.glb import (
    export_colored_instanced_voxels,
    export_colored_voxel_mesh,
    export_instanced_voxels,
    export_pixel_instances,
)


def test_instanced_voxel_glb_declares_extension(tmp_path: Path) -> None:
    output = tmp_path / "voxels.glb"
    export_instanced_voxels(
        np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        0.1,
        np.asarray([1.0, 0.25]),
        np.asarray([7, 15], dtype=np.uint8),
        output,
    )
    payload = output.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", payload, 0)
    assert (magic, version, total) == (b"glTF", 2, len(payload))
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20 : 20 + json_length])
    assert document["extensionsRequired"] == ["EXT_mesh_gpu_instancing"]
    node = document["nodes"][0]
    assert node["extras"]["instanceCount"] == 2
    assert "_CONFIDENCE" in node["extensions"]["EXT_mesh_gpu_instancing"]["attributes"]


def test_instanced_voxel_glb_supports_variable_pixel_sizes(tmp_path: Path) -> None:
    output = tmp_path / "variable-voxels.glb"
    export_instanced_voxels(
        np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        np.asarray([0.1, 0.05], dtype=np.float32),
        np.asarray([1.0, 0.5]),
        np.asarray([8, 9], dtype=np.uint8),
        output,
    )

    payload = output.read_bytes()
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20 : 20 + json_length])
    voxel_size = document["nodes"][0]["extras"]["voxelSize"]
    assert voxel_size["variable"] is True
    assert voxel_size["min"] == pytest.approx(0.05)
    assert voxel_size["max"] == pytest.approx(0.1)


def test_direct_pixel_glb_keeps_runtime_attributes_and_traceability_sidecar(tmp_path: Path) -> None:
    output = tmp_path / "pixels.glb"
    export_pixel_instances(
        translations=np.asarray([[0, 0, 0.2], [0.1, 0, 0.3]], dtype=np.float32),
        scales=np.full((2, 3), 0.05, dtype=np.float32),
        rotations=np.asarray([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float32),
        pixel_codes=np.asarray([0x112233, 0xAABBCC], dtype=np.uint32),
        source_uv=np.asarray([[100, 200], [101, 200]], dtype=np.uint16),
        depth=np.asarray([0.2, 0.3], dtype=np.float32),
        feature_class=np.asarray([0, 2], dtype=np.uint8),
        confidence=np.asarray([0.8, 0.9], dtype=np.float32),
        source_bits=np.asarray([7, 7], dtype=np.uint8),
        destination=output,
    )
    payload = output.read_bytes()
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20 : 20 + json_length])
    attributes = document["nodes"][0]["extensions"]["EXT_mesh_gpu_instancing"]["attributes"]
    assert {"TRANSLATION", "SCALE", "ROTATION", "_CONFIDENCE", "_SOURCE"} == set(attributes)
    assert document["nodes"][0]["extras"]["traceabilitySidecar"] == "pixels/pixels.bin"


def test_direct_pixel_glb_can_export_flat_faced_cubes(tmp_path: Path) -> None:
    output = tmp_path / "pixel-cubes.glb"
    export_pixel_instances(
        translations=np.asarray([[0, 0, 0]], dtype=np.float32),
        scales=np.full((1, 3), 0.05, dtype=np.float32),
        rotations=np.asarray([[0, 0, 0, 1]], dtype=np.float32),
        pixel_codes=np.asarray([0x112233], dtype=np.uint32),
        source_uv=np.asarray([[100, 200]], dtype=np.uint16),
        depth=np.asarray([0.2], dtype=np.float32),
        feature_class=np.asarray([0], dtype=np.uint8),
        confidence=np.asarray([0.8], dtype=np.float32),
        source_bits=np.asarray([7], dtype=np.uint8),
        destination=output,
        cell_shape="cube",
    )

    payload = output.read_bytes()
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20 : 20 + json_length])
    primitive = document["meshes"][0]["primitives"][0]
    position = document["accessors"][primitive["attributes"]["POSITION"]]
    assert document["meshes"][0]["name"] == "source-pixel-flat-cube"
    assert document["nodes"][0]["extras"]["cellShape"] == "cube"
    assert position["count"] == 36


def test_direct_pixel_glb_can_export_legacy_blue_cube_material_without_sidecars(
    tmp_path: Path,
) -> None:
    output = tmp_path / "quality-baseline-unit.glb"
    export_pixel_instances(
        translations=np.asarray([[0, 0, 0]], dtype=np.float32),
        scales=np.ones((1, 3), dtype=np.float32),
        rotations=np.asarray([[0, 0, 0, 1]], dtype=np.float32),
        pixel_codes=np.asarray([0], dtype=np.uint32),
        source_uv=np.asarray([[0, 0]], dtype=np.uint16),
        depth=np.asarray([0], dtype=np.float32),
        feature_class=np.asarray([0], dtype=np.uint8),
        confidence=np.asarray([1], dtype=np.float32),
        source_bits=np.asarray([0], dtype=np.uint8),
        destination=output,
        base_color=(0.49, 0.56, 0.67, 1.0),
        cell_shape="cube",
        material_name="quality-baseline-blue-cube-v1",
        metallic_factor=0.0,
        roughness_factor=0.64,
        specular_factor=None,
        traceability_sidecar=None,
        traceability_schema=None,
    )

    payload = output.read_bytes()
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20 : 20 + json_length])
    extras = document["nodes"][0]["extras"]
    material = document["materials"][0]
    assert document["extensionsUsed"] == ["EXT_mesh_gpu_instancing"]
    assert extras["cellShape"] == "cube"
    assert extras["materialProfile"] == "quality-baseline-blue-cube-v1"
    assert "traceabilitySidecar" not in extras
    assert "traceabilitySchema" not in extras
    assert material["pbrMetallicRoughness"] == {
        "baseColorFactor": [0.49, 0.56, 0.67, 1.0],
        "metallicFactor": 0.0,
        "roughnessFactor": 0.64,
    }
    assert "extensions" not in material


def test_colored_voxel_glb_binds_rgba_to_expanded_cube_vertices(tmp_path: Path) -> None:
    output = tmp_path / "colored-voxels.glb"
    export_colored_voxel_mesh(
        translations=np.asarray([[0, 0, 0], [0.1, 0, 0]], dtype=np.float32),
        scales=np.full((2, 3), 0.08, dtype=np.float32),
        colors=np.asarray([[0.2, 0.3, 0.4], [0.8, 0.7, 0.6]], dtype=np.float32),
        destination=output,
        mapping="test coordinate mapping",
    )

    payload = output.read_bytes()
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20 : 20 + json_length])
    primitive = document["meshes"][0]["primitives"][0]
    assert "COLOR_0" in primitive["attributes"]
    assert document["nodes"][0]["extras"]["cellCount"] == 2
    assert document["nodes"][0]["extras"]["mapping"] == "test coordinate mapping"
    assert "EXT_mesh_gpu_instancing" not in document.get("extensionsRequired", [])
    assert "KHR_materials_specular" in document["extensionsUsed"]
    material = document["materials"][0]
    assert material["pbrMetallicRoughness"]["roughnessFactor"] == pytest.approx(0.30)
    assert material["extensions"]["KHR_materials_specular"]["specularFactor"] == pytest.approx(0.78)


def test_colored_instanced_voxels_fill_volume_and_group_palette(tmp_path: Path) -> None:
    output = tmp_path / "solid-colored-voxels.glb"
    export_colored_instanced_voxels(
        translations=np.asarray(
            [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]],
            dtype=np.float32,
        ),
        scales=np.full((3, 3), 0.096, dtype=np.float32),
        colors=np.asarray(
            [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.8, 0.8, 0.8]],
            dtype=np.float32,
        ),
        destination=output,
        mapping="solid coordinate test",
        surface_cell_count=2,
    )

    payload = output.read_bytes()
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20 : 20 + json_length])
    assert document["extensionsRequired"] == ["EXT_mesh_gpu_instancing"]
    assert "KHR_materials_specular" in document["extensionsUsed"]
    assert document["extras"]["instanceCount"] == 3
    assert document["extras"]["surfaceCellCount"] == 2
    assert document["extras"]["interiorCellCount"] == 1
    assert document["extras"]["paletteSize"] == 2
    assert document["extras"]["solidVolumeFilled"] is True
    assert len(document["materials"]) == 2
    for material in document["materials"]:
        assert material["pbrMetallicRoughness"]["roughnessFactor"] == pytest.approx(0.30)
        assert material["extensions"]["KHR_materials_specular"]["specularFactor"] == pytest.approx(
            0.78
        )
    assert sum(node["extras"]["instanceCount"] for node in document["nodes"]) == 3
    for mesh in document["meshes"]:
        primitive = mesh["primitives"][0]
        position = document["accessors"][primitive["attributes"]["POSITION"]]
        assert position["count"] == 36


def test_colored_instanced_voxels_support_feature_palette_and_source_indices(
    tmp_path: Path,
) -> None:
    output = tmp_path / "contrast-pixels.glb"
    export_colored_instanced_voxels(
        translations=np.asarray([[0, 0, 0], [0.1, 0, 0]], dtype=np.float32),
        scales=np.full((2, 3), 0.092, dtype=np.float32),
        colors=np.asarray([[0.49, 0.56, 0.67], [0.035, 0.055, 0.10]], dtype=np.float32),
        destination=output,
        mapping="contrast pixel test",
        surface_cell_count=2,
        source_indices=np.asarray([4, 9], dtype=np.uint32),
        material_name="quality-baseline-blue-cube-v1",
        metallic_factor=0.0,
        roughness_factor=0.64,
        specular_factor=None,
        shadow_color_factor=(0.11, 0.15, 0.22),
        solid_volume_filled=False,
    )

    payload = output.read_bytes()
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20 : 20 + json_length])
    assert document["extensionsUsed"] == ["EXT_mesh_gpu_instancing"]
    assert document["extras"]["paletteSize"] == 2
    assert document["extras"]["solidVolumeFilled"] is False
    assert document["extras"]["materialProfile"] == "quality-baseline-blue-cube-v1"
    for node in document["nodes"]:
        attributes = node["extensions"]["EXT_mesh_gpu_instancing"]["attributes"]
        assert "_SOURCE_INDEX" in attributes
    for material in document["materials"]:
        assert material["pbrMetallicRoughness"]["metallicFactor"] == 0.0
        assert material["pbrMetallicRoughness"]["roughnessFactor"] == pytest.approx(0.64)
        assert material["extras"]["qaShadowColorFactor"] == [0.11, 0.15, 0.22]
        assert "extensions" not in material


def test_v2_pixel_glb_identifies_unified_head_mapping(tmp_path: Path) -> None:
    output = tmp_path / "pixels-v2.glb"
    export_pixel_instances(
        translations=np.asarray([[0, 0, 0.2]], dtype=np.float32),
        scales=np.full((1, 3), 0.05, dtype=np.float32),
        rotations=np.asarray([[0, 0, 0, 1]], dtype=np.float32),
        pixel_codes=np.asarray([0x112233], dtype=np.uint32),
        source_uv=np.asarray([[100, 200]], dtype=np.uint16),
        depth=np.asarray([0.2], dtype=np.float32),
        feature_class=np.asarray([1], dtype=np.uint8),
        confidence=np.asarray([0.8], dtype=np.float32),
        source_bits=np.asarray([1], dtype=np.uint8),
        destination=output,
        contract="v2",
    )
    payload = output.read_bytes()
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20 : 20 + json_length])
    assert document["nodes"][0]["name"] == "traceable-unified-head-3d-pixels"
    assert "canonical head triangles" in document["nodes"][0]["extras"]["mapping"]
