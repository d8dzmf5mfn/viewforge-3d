from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import trimesh

from face3d.io import atomic_write_bytes

GLTF_COMPONENT_FLOAT = 5126
GLTF_COMPONENT_UNSIGNED_BYTE = 5121
GLTF_COMPONENT_UNSIGNED_SHORT = 5123
GLTF_COMPONENT_UNSIGNED_INT = 5125
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963


def export_neutral_mesh(mesh: trimesh.Trimesh, destination: Path) -> None:
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=(0.50, 0.52, 0.56, 1.0),
        metallicFactor=0.0,
        roughnessFactor=0.76,
    )
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    atomic_write_bytes(
        destination,
        trimesh.exchange.gltf.export_glb(mesh, include_normals=True),
    )


@dataclass(slots=True)
class _BufferBuilder:
    payload: bytearray
    views: list[dict[str, Any]]
    accessors: list[dict[str, Any]]

    @classmethod
    def create(cls) -> _BufferBuilder:
        return cls(bytearray(), [], [])

    def add(
        self,
        array: np.ndarray,
        *,
        component_type: int,
        accessor_type: str,
        target: int | None = None,
        include_bounds: bool = False,
    ) -> int:
        while len(self.payload) % 4:
            self.payload.append(0)
        contiguous = np.ascontiguousarray(array)
        offset = len(self.payload)
        encoded = contiguous.tobytes(order="C")
        self.payload.extend(encoded)
        view: dict[str, Any] = {"buffer": 0, "byteOffset": offset, "byteLength": len(encoded)}
        if target is not None:
            view["target"] = target
        view_index = len(self.views)
        self.views.append(view)
        count = int(contiguous.shape[0])
        accessor: dict[str, Any] = {
            "bufferView": view_index,
            "byteOffset": 0,
            "componentType": component_type,
            "count": count,
            "type": accessor_type,
        }
        if include_bounds:
            values = contiguous.reshape(count, -1)
            accessor["min"] = values.min(axis=0).astype(float).tolist()
            accessor["max"] = values.max(axis=0).astype(float).tolist()
        accessor_index = len(self.accessors)
        self.accessors.append(accessor)
        return accessor_index


def _glb(json_document: dict[str, Any], binary: bytes) -> bytes:
    json_bytes = json.dumps(
        json_document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    json_padding = (-len(json_bytes)) % 4
    json_bytes += b" " * json_padding
    binary_padding = (-len(binary)) % 4
    binary += b"\x00" * binary_padding
    total_length = 12 + 8 + len(json_bytes) + 8 + len(binary)
    return b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total_length),
            struct.pack("<I4s", len(json_bytes), b"JSON"),
            json_bytes,
            struct.pack("<I4s", len(binary), b"BIN\x00"),
            binary,
        )
    )


def export_instanced_voxels(
    translations: np.ndarray,
    voxel_size: float | np.ndarray,
    confidence: np.ndarray,
    source_bits: np.ndarray,
    destination: Path,
    *,
    fill_ratio: float = 0.92,
) -> None:
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError("translations must be [N,3]")
    if len(confidence) != len(translations) or len(source_bits) != len(translations):
        raise ValueError("instance attribute lengths must match")
    voxel_sizes = np.asarray(voxel_size, dtype=np.float32)
    if voxel_sizes.ndim == 0:
        voxel_sizes = np.full(len(translations), float(voxel_sizes), dtype=np.float32)
    elif voxel_sizes.shape != (len(translations),):
        raise ValueError("voxel_size must be scalar or [N]")
    if not np.all(np.isfinite(voxel_sizes)) or np.any(voxel_sizes <= 0):
        raise ValueError("voxel_size must be finite and positive")
    if not np.isfinite(fill_ratio) or not 0 < fill_ratio <= 1:
        raise ValueError("fill_ratio must be in (0, 1]")
    # A shallow hexagonal prism reads as one surface tile after its local Z
    # axis is aligned to the mesh normal. The previous icosahedron exposed
    # sloped facets and made an otherwise smooth head look pitted.
    cell = trimesh.creation.cylinder(radius=0.5, height=1.0, sections=6)
    positions = np.asarray(cell.vertices, dtype=np.float32)
    normals = np.asarray(cell.vertex_normals, dtype=np.float32)
    indices = np.asarray(cell.faces.reshape(-1), dtype=np.uint16)
    scales = np.repeat(voxel_sizes[:, None], 3, axis=1) * fill_ratio
    builder = _BufferBuilder.create()
    position_accessor = builder.add(
        positions,
        component_type=GLTF_COMPONENT_FLOAT,
        accessor_type="VEC3",
        target=ARRAY_BUFFER,
        include_bounds=True,
    )
    normal_accessor = builder.add(
        normals,
        component_type=GLTF_COMPONENT_FLOAT,
        accessor_type="VEC3",
        target=ARRAY_BUFFER,
    )
    index_accessor = builder.add(
        indices,
        component_type=GLTF_COMPONENT_UNSIGNED_SHORT,
        accessor_type="SCALAR",
        target=ELEMENT_ARRAY_BUFFER,
    )
    translation_accessor = builder.add(
        np.asarray(translations, dtype=np.float32),
        component_type=GLTF_COMPONENT_FLOAT,
        accessor_type="VEC3",
        target=ARRAY_BUFFER,
        include_bounds=True,
    )
    scale_accessor = builder.add(
        scales,
        component_type=GLTF_COMPONENT_FLOAT,
        accessor_type="VEC3",
        target=ARRAY_BUFFER,
    )
    confidence_accessor = builder.add(
        np.asarray(confidence, dtype=np.float32).reshape(-1),
        component_type=GLTF_COMPONENT_FLOAT,
        accessor_type="SCALAR",
        target=ARRAY_BUFFER,
    )
    source_accessor = builder.add(
        np.asarray(source_bits, dtype=np.uint8).reshape(-1),
        component_type=GLTF_COMPONENT_UNSIGNED_BYTE,
        accessor_type="SCALAR",
        target=ARRAY_BUFFER,
    )
    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "face3d 0.1.0"},
        "extensionsUsed": ["EXT_mesh_gpu_instancing"],
        "extensionsRequired": ["EXT_mesh_gpu_instancing"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {
                "name": "surface-3d-pixels",
                "mesh": 0,
                "extensions": {
                    "EXT_mesh_gpu_instancing": {
                        "attributes": {
                            "TRANSLATION": translation_accessor,
                            "SCALE": scale_accessor,
                            "_CONFIDENCE": confidence_accessor,
                            "_SOURCE": source_accessor,
                        }
                    }
                },
                "extras": {
                    "instanceCount": int(len(translations)),
                    "voxelSize": (
                        float(voxel_sizes[0])
                        if np.all(voxel_sizes == voxel_sizes[0])
                        else {
                            "variable": True,
                            "min": float(np.min(voxel_sizes)),
                            "max": float(np.max(voxel_sizes)),
                        }
                    ),
                    "fillRatio": float(fill_ratio),
                    "sourceBits": {
                        "frontMask": 1,
                        "left45Mask": 2,
                        "right45Mask": 4,
                        "templateInferred": 8,
                    },
                },
            }
        ],
        "meshes": [
            {
                "name": "low-poly-convex-cell",
                "primitives": [
                    {
                        "attributes": {"POSITION": position_accessor, "NORMAL": normal_accessor},
                        "indices": index_accessor,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "neutral-3d-pixel",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.49, 0.56, 0.67, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.64,
                },
            }
        ],
        "buffers": [{"byteLength": len(builder.payload)}],
        "bufferViews": builder.views,
        "accessors": builder.accessors,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(destination, _glb(document, bytes(builder.payload)))


def export_pixel_instances(
    translations: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    pixel_codes: np.ndarray,
    source_uv: np.ndarray,
    depth: np.ndarray,
    feature_class: np.ndarray,
    confidence: np.ndarray,
    source_bits: np.ndarray,
    destination: Path,
    base_color: tuple[float, float, float, float] = (0.25, 0.68, 0.88, 1.0),
    contract: str = "v1",
    cell_shape: Literal["convex-tile", "cube"] = "convex-tile",
    mapping: str | None = None,
    source_bits_legend: dict[str, int] | None = None,
    material_name: str = "neutral-3d-pixel",
    metallic_factor: float = 0.04,
    roughness_factor: float = 0.30,
    specular_factor: float | None = 0.78,
    specular_color: tuple[float, float, float] = (0.92, 0.96, 1.0),
    traceability_sidecar: str | None = "pixels/pixels.bin",
    traceability_schema: str | None = "pixels/schema.json",
) -> None:
    """Export traceable cubic cells for measured and inferred pixel-shell samples."""
    translations = np.asarray(translations, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    rotations = np.asarray(rotations, dtype=np.float32)
    count = len(translations)
    if (
        translations.shape != (count, 3)
        or scales.shape != (count, 3)
        or rotations.shape != (count, 4)
    ):
        raise ValueError("translations/scales/rotations must be [N,3]/[N,3]/[N,4]")
    attributes = {
        "pixel_codes": np.asarray(pixel_codes, dtype=np.uint32).reshape(-1),
        "source_uv": np.asarray(source_uv, dtype=np.uint16).reshape(-1, 2),
        "depth": np.asarray(depth, dtype=np.float32).reshape(-1),
        "feature_class": np.asarray(feature_class, dtype=np.uint8).reshape(-1),
        "confidence": np.asarray(confidence, dtype=np.float32).reshape(-1),
        "source_bits": np.asarray(source_bits, dtype=np.uint8).reshape(-1),
    }
    if any(len(value) != count for value in attributes.values()):
        raise ValueError("instance attribute lengths must match")
    if (
        not np.all(np.isfinite(translations))
        or not np.all(np.isfinite(scales))
        or not np.all(np.isfinite(rotations))
    ):
        raise ValueError("instance transforms must be finite")
    if np.any(scales <= 0):
        raise ValueError("instance scales must be positive")
    quaternion_length = np.linalg.norm(rotations, axis=1)
    if np.any(np.abs(quaternion_length - 1.0) > 1e-4):
        raise ValueError("instance rotations must be normalized quaternions")
    material_scalars = np.asarray(
        [*base_color, metallic_factor, roughness_factor, *specular_color],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(material_scalars)) or np.any(material_scalars < 0):
        raise ValueError("pixel material values must be finite and non-negative")
    if np.any(np.asarray(base_color) > 1) or metallic_factor > 1 or roughness_factor > 1:
        raise ValueError("pixel PBR material values must be in [0, 1]")
    if np.any(np.asarray(specular_color) > 1):
        raise ValueError("pixel specular color values must be in [0, 1]")
    if specular_factor is not None and (
        not np.isfinite(specular_factor) or not 0 <= specular_factor <= 1
    ):
        raise ValueError("pixel specular factor must be None or in [0, 1]")

    if cell_shape == "convex-tile":
        cell = trimesh.creation.icosphere(subdivisions=0, radius=0.5)
        mesh_name = "source-pixel-convex-tile"
    elif cell_shape == "cube":
        cell = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        # Duplicate vertices per triangle so every cube face keeps a hard
        # normal even when a viewer uses an ordinary smooth-shaded material.
        cell.unmerge_vertices()
        mesh_name = "source-pixel-flat-cube"
    else:
        raise ValueError(f"unsupported pixel cell shape: {cell_shape}")
    positions = np.asarray(cell.vertices, dtype=np.float32)
    normals = np.asarray(cell.vertex_normals, dtype=np.float32)
    indices = np.asarray(cell.faces.reshape(-1), dtype=np.uint16)
    builder = _BufferBuilder.create()
    position_accessor = builder.add(
        positions,
        component_type=GLTF_COMPONENT_FLOAT,
        accessor_type="VEC3",
        target=ARRAY_BUFFER,
        include_bounds=True,
    )
    normal_accessor = builder.add(
        normals,
        component_type=GLTF_COMPONENT_FLOAT,
        accessor_type="VEC3",
        target=ARRAY_BUFFER,
    )
    index_accessor = builder.add(
        indices,
        component_type=GLTF_COMPONENT_UNSIGNED_SHORT,
        accessor_type="SCALAR",
        target=ELEMENT_ARRAY_BUFFER,
    )
    instance_accessors = {
        "TRANSLATION": builder.add(
            translations,
            component_type=GLTF_COMPONENT_FLOAT,
            accessor_type="VEC3",
            target=ARRAY_BUFFER,
            include_bounds=True,
        ),
        "SCALE": builder.add(
            scales,
            component_type=GLTF_COMPONENT_FLOAT,
            accessor_type="VEC3",
            target=ARRAY_BUFFER,
        ),
        "ROTATION": builder.add(
            rotations,
            component_type=GLTF_COMPONENT_FLOAT,
            accessor_type="VEC4",
            target=ARRAY_BUFFER,
        ),
        "_CONFIDENCE": builder.add(
            attributes["confidence"],
            component_type=GLTF_COMPONENT_FLOAT,
            accessor_type="SCALAR",
            target=ARRAY_BUFFER,
        ),
        "_SOURCE": builder.add(
            attributes["source_bits"],
            component_type=GLTF_COMPONENT_UNSIGNED_BYTE,
            accessor_type="SCALAR",
            target=ARRAY_BUFFER,
        ),
    }
    is_v2 = contract == "v2"
    node_extras: dict[str, Any] = {
        "instanceCount": count,
        "mapping": mapping
        or (
            "multi-view pixels to canonical head triangles and eyeball nodes"
            if is_v2
            else "measured front pixels plus multi-view sampled continuous surface"
        ),
        "cellShape": cell_shape,
        "materialProfile": material_name,
        "featureClasses": {
            "0": "simpleInterpolated",
            "1": "eyes",
            "2": "nose",
            "3": "mouth",
            "4": "ears",
            "5": "jaw",
        },
        "sourceBits": source_bits_legend
        or {
            "front": 1,
            "left45": 2,
            "right45": 4,
            "templateInferred": 8,
        },
    }
    if traceability_sidecar is not None:
        node_extras["traceabilitySidecar"] = traceability_sidecar
    if traceability_schema is not None:
        node_extras["traceabilitySchema"] = traceability_schema
    material: dict[str, Any] = {
        "name": material_name,
        "pbrMetallicRoughness": {
            "baseColorFactor": [float(value) for value in base_color],
            "metallicFactor": float(metallic_factor),
            "roughnessFactor": float(roughness_factor),
        },
    }
    extensions_used = ["EXT_mesh_gpu_instancing"]
    if specular_factor is not None:
        extensions_used.append("KHR_materials_specular")
        material["extensions"] = {
            "KHR_materials_specular": {
                "specularFactor": float(specular_factor),
                "specularColorFactor": [float(value) for value in specular_color],
            }
        }
    document: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": (
                "face3d pixel-flame-hybrid 0.1.0" if is_v2 else "face3d pixel-direct 0.1.0"
            ),
        },
        "extensionsUsed": extensions_used,
        "extensionsRequired": ["EXT_mesh_gpu_instancing"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {
                "name": (
                    "traceable-unified-head-3d-pixels" if is_v2 else "direct-2d-pixel-to-3d-cells"
                ),
                "mesh": 0,
                "extensions": {"EXT_mesh_gpu_instancing": {"attributes": instance_accessors}},
                "extras": node_extras,
            }
        ],
        "meshes": [
            {
                "name": mesh_name,
                "primitives": [
                    {
                        "attributes": {"POSITION": position_accessor, "NORMAL": normal_accessor},
                        "indices": index_accessor,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [material],
        "buffers": [{"byteLength": len(builder.payload)}],
        "bufferViews": builder.views,
        "accessors": builder.accessors,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(destination, _glb(document, bytes(builder.payload)))


def export_colored_voxel_mesh(
    translations: np.ndarray,
    scales: np.ndarray,
    colors: np.ndarray,
    destination: Path,
    *,
    mapping: str,
    traceability_sidecar: str = "pixels/pixels.bin",
) -> None:
    """Export hard-edged cubic cells with one baked RGBA color per 3D coordinate."""
    translations = np.asarray(translations, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    count = len(translations)
    if translations.shape != (count, 3) or scales.shape != (count, 3):
        raise ValueError("translations and scales must be [N,3]")
    if colors.shape == (count, 3):
        colors = np.column_stack((colors, np.ones(count, dtype=np.float32)))
    if colors.shape != (count, 4):
        raise ValueError("colors must be [N,3] or [N,4]")
    if count == 0:
        raise ValueError("at least one colored voxel is required")
    if (
        not np.all(np.isfinite(translations))
        or not np.all(np.isfinite(scales))
        or not np.all(np.isfinite(colors))
    ):
        raise ValueError("colored voxel attributes must be finite")
    if np.any(scales <= 0):
        raise ValueError("colored voxel scales must be positive")
    if np.any((colors < 0) | (colors > 1)):
        raise ValueError("colored voxel colors must be normalized to [0,1]")

    cell = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    cell.unmerge_vertices()
    cell_vertices = np.asarray(cell.vertices, dtype=np.float32)
    cell_normals = np.asarray(cell.vertex_normals, dtype=np.float32)
    cell_faces = np.asarray(cell.faces, dtype=np.uint32)
    vertices_per_cell = len(cell_vertices)
    faces_per_cell = len(cell_faces)
    positions = (cell_vertices[None, :, :] * scales[:, None, :] + translations[:, None, :]).reshape(
        -1, 3
    )
    normals = np.broadcast_to(
        cell_normals[None, :, :],
        (count, vertices_per_cell, 3),
    ).reshape(-1, 3)
    vertex_colors = np.broadcast_to(
        colors[:, None, :],
        (count, vertices_per_cell, 4),
    ).reshape(-1, 4)
    offsets = np.arange(count, dtype=np.uint32)[:, None, None] * vertices_per_cell
    faces = (cell_faces[None, :, :] + offsets).reshape(faces_per_cell * count, 3)

    builder = _BufferBuilder.create()
    position_accessor = builder.add(
        positions,
        component_type=GLTF_COMPONENT_FLOAT,
        accessor_type="VEC3",
        target=ARRAY_BUFFER,
        include_bounds=True,
    )
    normal_accessor = builder.add(
        normals,
        component_type=GLTF_COMPONENT_FLOAT,
        accessor_type="VEC3",
        target=ARRAY_BUFFER,
    )
    color_accessor = builder.add(
        vertex_colors,
        component_type=GLTF_COMPONENT_FLOAT,
        accessor_type="VEC4",
        target=ARRAY_BUFFER,
    )
    index_accessor = builder.add(
        faces.reshape(-1),
        component_type=GLTF_COMPONENT_UNSIGNED_INT,
        accessor_type="SCALAR",
        target=ELEMENT_ARRAY_BUFFER,
    )
    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "face3d colored-coordinate-pixel 0.1.0"},
        "extensionsUsed": ["KHR_materials_specular"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {
                "name": "colored-3d-coordinate-pixels",
                "mesh": 0,
                "extras": {
                    "cellCount": count,
                    "mapping": mapping,
                    "traceabilitySidecar": traceability_sidecar,
                    "cellShape": "axis-aligned-flat-cube",
                    "colorBinding": "one fused source color per 3D coordinate",
                },
            }
        ],
        "meshes": [
            {
                "name": "expanded-colored-coordinate-cubes",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": position_accessor,
                            "NORMAL": normal_accessor,
                            "COLOR_0": color_accessor,
                        },
                        "indices": index_accessor,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "source-gray-pixel-colors",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.04,
                    "roughnessFactor": 0.30,
                },
                "extensions": {
                    "KHR_materials_specular": {
                        "specularFactor": 0.78,
                        "specularColorFactor": [0.92, 0.96, 1.0],
                    }
                },
            }
        ],
        "buffers": [{"byteLength": len(builder.payload)}],
        "bufferViews": builder.views,
        "accessors": builder.accessors,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(destination, _glb(document, bytes(builder.payload)))


ConnectedMaterialProfile = Literal[
    "reflective-pixel",
    "polished-milky-quartz",
    "quality-baseline-contrast",
    "quality-baseline-contrast-smooth",
]


def _connected_surface_material(
    material_profile: ConnectedMaterialProfile,
) -> tuple[list[str], dict[str, Any]]:
    if material_profile == "reflective-pixel":
        return ["KHR_materials_specular"], {
            "name": "reflective-connected-pixel-surface",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.04,
                "roughnessFactor": 0.30,
            },
            "extensions": {
                "KHR_materials_specular": {
                    "specularFactor": 0.78,
                    "specularColorFactor": [0.92, 0.96, 1.0],
                }
            },
        }
    if material_profile == "polished-milky-quartz":
        return [
            "KHR_materials_specular",
            "KHR_materials_ior",
            "KHR_materials_transmission",
            "KHR_materials_volume",
        ], {
            "name": "polished-milky-quartz",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.97, 0.985, 1.0, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.22,
            },
            "extensions": {
                "KHR_materials_specular": {
                    "specularFactor": 1.0,
                    "specularColorFactor": [1.0, 1.0, 1.0],
                },
                "KHR_materials_ior": {"ior": 1.544},
                "KHR_materials_transmission": {"transmissionFactor": 0.10},
                "KHR_materials_volume": {
                    "thicknessFactor": 0.08,
                    "attenuationDistance": 0.42,
                    "attenuationColor": [0.82, 0.91, 1.0],
                },
            },
            "extras": {
                "materialClass": "natural-stone/quartz",
                "finish": "polished-milky",
                "vertexColorRole": "three-dimensional quartz tone and mineral veining",
            },
        }
    if material_profile == "quality-baseline-contrast":
        return [], {
            "name": "quality-baseline-contrast-connected-surface",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.64,
            },
            "extras": {
                "materialClass": "quality-baseline-blue",
                "finish": "matte-connected",
                "vertexColorRole": "preserve facial-feature and ear contrast",
                "qaShadowColorFactor": [0.11, 0.15, 0.22],
            },
        }
    if material_profile == "quality-baseline-contrast-smooth":
        return [], {
            "name": "quality-baseline-contrast-smooth-connected-surface",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.32,
            },
            "extras": {
                "materialClass": "quality-baseline-blue",
                "finish": "smooth-connected",
                "vertexColorRole": "preserve facial-feature and ear contrast",
                "qaShadowColorFactor": [0.11, 0.15, 0.22],
            },
        }
    raise ValueError(f"unsupported connected surface material: {material_profile}")


def export_colored_connected_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
    destination: Path,
    *,
    mapping: str,
    topology_sidecar: str,
    material_profile: ConnectedMaterialProfile = "reflective-pixel",
) -> None:
    """Export touching Pixel boundary faces with flat render normals and no smoothing."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    face_colors = np.asarray(face_colors, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("vertices must be a non-empty [N,3] array")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("faces must be a non-empty [M,3] array")
    if int(np.max(faces)) >= len(vertices):
        raise ValueError("faces contain an out-of-range vertex index")
    if face_colors.shape == (len(faces), 3):
        face_colors = np.column_stack((face_colors, np.ones(len(face_colors), dtype=np.float32)))
    if face_colors.shape != (len(faces), 4):
        raise ValueError("face_colors must be [M,3] or [M,4]")
    if (
        not np.all(np.isfinite(vertices))
        or not np.all(np.isfinite(face_colors))
        or np.any((face_colors < 0) | (face_colors > 1))
    ):
        raise ValueError("connected surface attributes must be finite and normalized")
    extensions_used, material = _connected_surface_material(material_profile)

    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    if np.any(lengths <= 1e-12):
        raise ValueError("connected surface contains a degenerate triangle")
    face_normals /= lengths
    render_positions = triangles.reshape(-1, 3)
    render_normals = np.repeat(face_normals, 3, axis=0).astype(np.float32)
    render_colors = np.repeat(face_colors[:, None, :], 3, axis=1).reshape(-1, 4)

    builder = _BufferBuilder.create()
    position_accessor = builder.add(
        render_positions,
        component_type=GLTF_COMPONENT_FLOAT,
        accessor_type="VEC3",
        target=ARRAY_BUFFER,
        include_bounds=True,
    )
    normal_accessor = builder.add(
        render_normals,
        component_type=GLTF_COMPONENT_FLOAT,
        accessor_type="VEC3",
        target=ARRAY_BUFFER,
    )
    color_accessor = builder.add(
        render_colors,
        component_type=GLTF_COMPONENT_FLOAT,
        accessor_type="VEC4",
        target=ARRAY_BUFFER,
    )
    document: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": "face3d connected-pixel-surface 0.1.0",
        },
        "extensionsUsed": extensions_used,
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {
                "name": "direct-connected-pixel-surface",
                "mesh": 0,
                "extras": {
                    "canonicalVertexCount": int(len(vertices)),
                    "triangleCount": int(len(faces)),
                    "mapping": mapping,
                    "topologySidecar": topology_sidecar,
                    "materialProfile": material_profile,
                    "surfaceConnection": "exact shared voxel-face boundaries",
                    "renderVerticesDuplicatedForFlatNormals": True,
                    "vertexSmoothingApplied": False,
                    "normalSmoothingApplied": False,
                    "subdivisionApplied": False,
                    "marchingCubesApplied": False,
                },
            }
        ],
        "meshes": [
            {
                "name": "touching-flat-pixel-boundary-faces",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": position_accessor,
                            "NORMAL": normal_accessor,
                            "COLOR_0": color_accessor,
                        },
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [material],
        "buffers": [{"byteLength": len(builder.payload)}],
        "bufferViews": builder.views,
        "accessors": builder.accessors,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(destination, _glb(document, bytes(builder.payload)))


def export_colored_coordinate_unit_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_normals: np.ndarray,
    vertex_colors: np.ndarray,
    destination: Path,
    *,
    mapping: str,
    topology_sidecar: str,
    material_profile: ConnectedMaterialProfile = "polished-milky-quartz",
) -> None:
    """Export an indexed surface joining custom 3D unit coordinates, not voxel cubes."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    vertex_normals = np.asarray(vertex_normals, dtype=np.float32)
    vertex_colors = np.asarray(vertex_colors, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("vertices must be a non-empty [N,3] array")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("faces must be a non-empty [M,3] array")
    if int(np.max(faces)) >= len(vertices):
        raise ValueError("faces contain an out-of-range vertex index")
    if vertex_normals.shape != vertices.shape:
        raise ValueError("vertex_normals must match vertices")
    if vertex_colors.shape == (len(vertices), 3):
        vertex_colors = np.column_stack(
            (vertex_colors, np.ones(len(vertex_colors), dtype=np.float32))
        )
    if vertex_colors.shape != (len(vertices), 4):
        raise ValueError("vertex_colors must be [N,3] or [N,4]")
    normal_lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    if (
        not np.all(np.isfinite(vertices))
        or not np.all(np.isfinite(vertex_normals))
        or not np.all(np.isfinite(vertex_colors))
        or np.any(normal_lengths <= 1e-12)
        or np.any((vertex_colors < 0) | (vertex_colors > 1))
    ):
        raise ValueError("coordinate-unit surface attributes must be finite and normalized")
    vertex_normals = vertex_normals / normal_lengths
    extensions_used, material = _connected_surface_material(material_profile)

    builder = _BufferBuilder.create()
    position_accessor = builder.add(
        vertices,
        component_type=GLTF_COMPONENT_FLOAT,
        accessor_type="VEC3",
        target=ARRAY_BUFFER,
        include_bounds=True,
    )
    normal_accessor = builder.add(
        vertex_normals,
        component_type=GLTF_COMPONENT_FLOAT,
        accessor_type="VEC3",
        target=ARRAY_BUFFER,
    )
    color_accessor = builder.add(
        vertex_colors,
        component_type=GLTF_COMPONENT_FLOAT,
        accessor_type="VEC4",
        target=ARRAY_BUFFER,
    )
    index_accessor = builder.add(
        faces.reshape(-1),
        component_type=GLTF_COMPONENT_UNSIGNED_INT,
        accessor_type="SCALAR",
        target=ELEMENT_ARRAY_BUFFER,
    )
    document: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": "face3d connected-coordinate-unit-surface 0.1.0",
        },
        "extensionsUsed": extensions_used,
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {
                "name": "connected-coordinate-unit-surface",
                "mesh": 0,
                "extras": {
                    "vertexCount": int(len(vertices)),
                    "triangleCount": int(len(faces)),
                    "mapping": mapping,
                    "topologySidecar": topology_sidecar,
                    "materialProfile": material_profile,
                    "surfaceConnection": "triangles join neighboring custom 3D unit coordinates",
                    "visibleUnitCubesExported": False,
                    "geometrySmoothingApplied": False,
                    "normalInterpolationApplied": True,
                    "subdivisionApplied": False,
                    "marchingCubesApplied": False,
                },
            }
        ],
        "meshes": [
            {
                "name": "continuous-coordinate-unit-triangle-surface",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": position_accessor,
                            "NORMAL": normal_accessor,
                            "COLOR_0": color_accessor,
                        },
                        "indices": index_accessor,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [material],
        "buffers": [{"byteLength": len(builder.payload)}],
        "bufferViews": builder.views,
        "accessors": builder.accessors,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(destination, _glb(document, bytes(builder.payload)))


def export_colored_instanced_voxels(
    translations: np.ndarray,
    scales: np.ndarray,
    colors: np.ndarray,
    destination: Path,
    *,
    mapping: str,
    surface_cell_count: int,
    traceability_sidecar: str | None = "pixels/pixels.bin",
    source_indices: np.ndarray | None = None,
    material_name: str = "coordinate-gray",
    metallic_factor: float = 0.04,
    roughness_factor: float = 0.30,
    specular_factor: float | None = 0.78,
    specular_color: tuple[float, float, float] = (0.92, 0.96, 1.0),
    shadow_color_factor: tuple[float, float, float] | None = None,
    solid_volume_filled: bool = True,
) -> None:
    """Export axis-aligned cubes grouped by material color."""
    translations = np.asarray(translations, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    count = len(translations)
    if translations.shape != (count, 3) or scales.shape != (count, 3):
        raise ValueError("translations and scales must be [N,3]")
    if colors.shape == (count, 3):
        colors = np.column_stack((colors, np.ones(count, dtype=np.float32)))
    if colors.shape != (count, 4):
        raise ValueError("colors must be [N,3] or [N,4]")
    if count == 0:
        raise ValueError("at least one colored voxel is required")
    if not 0 <= surface_cell_count <= count:
        raise ValueError("surface_cell_count must be between zero and the instance count")
    if (
        not np.all(np.isfinite(translations))
        or not np.all(np.isfinite(scales))
        or not np.all(np.isfinite(colors))
    ):
        raise ValueError("colored voxel attributes must be finite")
    if np.any(scales <= 0):
        raise ValueError("colored voxel scales must be positive")
    if np.any((colors < 0) | (colors > 1)):
        raise ValueError("colored voxel colors must be normalized to [0,1]")
    resolved_source_indices = None
    if source_indices is not None:
        resolved_source_indices = np.asarray(source_indices, dtype=np.uint32).reshape(-1)
        if len(resolved_source_indices) != count:
            raise ValueError("source_indices must contain one value per colored voxel")
    material_scalars = np.asarray(
        [metallic_factor, roughness_factor, *specular_color], dtype=np.float64
    )
    if not np.all(np.isfinite(material_scalars)) or np.any(material_scalars < 0):
        raise ValueError("colored voxel material values must be finite and non-negative")
    if metallic_factor > 1 or roughness_factor > 1 or np.any(np.asarray(specular_color) > 1):
        raise ValueError("colored voxel PBR material values must be in [0,1]")
    if specular_factor is not None and (
        not np.isfinite(specular_factor) or not 0 <= specular_factor <= 1
    ):
        raise ValueError("colored voxel specular factor must be None or in [0,1]")
    if shadow_color_factor is not None:
        shadow_values = np.asarray(shadow_color_factor, dtype=np.float64)
        if (
            shadow_values.shape != (3,)
            or not np.all(np.isfinite(shadow_values))
            or np.any((shadow_values < 0) | (shadow_values > 1))
        ):
            raise ValueError("shadow_color_factor must contain three values in [0,1]")

    # Material grouping keeps per-cell color in ordinary glTF viewers without
    # expanding hundreds of thousands of identical cubes into unique vertices.
    rgba8 = np.rint(colors * 255.0).astype(np.uint8)
    palette, palette_index = np.unique(rgba8, axis=0, return_inverse=True)
    cell = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    cell.unmerge_vertices()
    builder = _BufferBuilder.create()
    position_accessor = builder.add(
        np.asarray(cell.vertices, dtype=np.float32),
        component_type=GLTF_COMPONENT_FLOAT,
        accessor_type="VEC3",
        target=ARRAY_BUFFER,
        include_bounds=True,
    )
    normal_accessor = builder.add(
        np.asarray(cell.vertex_normals, dtype=np.float32),
        component_type=GLTF_COMPONENT_FLOAT,
        accessor_type="VEC3",
        target=ARRAY_BUFFER,
    )
    index_accessor = builder.add(
        np.asarray(cell.faces.reshape(-1), dtype=np.uint16),
        component_type=GLTF_COMPONENT_UNSIGNED_SHORT,
        accessor_type="SCALAR",
        target=ELEMENT_ARRAY_BUFFER,
    )

    nodes: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    for color_index, rgba in enumerate(palette):
        selected = np.flatnonzero(palette_index == color_index)
        translation_accessor = builder.add(
            translations[selected],
            component_type=GLTF_COMPONENT_FLOAT,
            accessor_type="VEC3",
            target=ARRAY_BUFFER,
            include_bounds=True,
        )
        scale_accessor = builder.add(
            scales[selected],
            component_type=GLTF_COMPONENT_FLOAT,
            accessor_type="VEC3",
            target=ARRAY_BUFFER,
        )
        instance_attributes = {
            "TRANSLATION": translation_accessor,
            "SCALE": scale_accessor,
        }
        if resolved_source_indices is not None:
            instance_attributes["_SOURCE_INDEX"] = builder.add(
                resolved_source_indices[selected],
                component_type=GLTF_COMPONENT_UNSIGNED_INT,
                accessor_type="SCALAR",
                target=ARRAY_BUFFER,
            )
        nodes.append(
            {
                "name": f"solid-cube-color-{color_index:02d}",
                "mesh": color_index,
                "extensions": {"EXT_mesh_gpu_instancing": {"attributes": instance_attributes}},
                "extras": {
                    "instanceCount": int(len(selected)),
                    "paletteIndex": color_index,
                    "rgba8": rgba.astype(int).tolist(),
                },
            }
        )
        meshes.append(
            {
                "name": f"solid-axis-aligned-cubes-{color_index:02d}",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": position_accessor,
                            "NORMAL": normal_accessor,
                        },
                        "indices": index_accessor,
                        "material": color_index,
                        "mode": 4,
                    }
                ],
            }
        )
        material: dict[str, Any] = {
            "name": f"{material_name}-{color_index:02d}",
            "pbrMetallicRoughness": {
                "baseColorFactor": (rgba.astype(np.float32) / 255.0).tolist(),
                "metallicFactor": float(metallic_factor),
                "roughnessFactor": float(roughness_factor),
            },
        }
        if specular_factor is not None:
            material["extensions"] = {
                "KHR_materials_specular": {
                    "specularFactor": float(specular_factor),
                    "specularColorFactor": [float(value) for value in specular_color],
                }
            }
        if shadow_color_factor is not None:
            material["extras"] = {
                "qaShadowColorFactor": [float(value) for value in shadow_color_factor]
            }
        materials.append(material)

    extensions_used = ["EXT_mesh_gpu_instancing"]
    if specular_factor is not None:
        extensions_used.append("KHR_materials_specular")
    root_extras: dict[str, Any] = {
        "instanceCount": count,
        "surfaceCellCount": int(surface_cell_count),
        "interiorCellCount": int(count - surface_cell_count),
        "paletteSize": int(len(palette)),
        "mapping": mapping,
        "cellShape": "hard-flat-axis-aligned-cube",
        "solidVolumeFilled": bool(solid_volume_filled),
        "materialProfile": material_name,
        "materialResponse": {
            "roughnessFactor": float(roughness_factor),
            "metallicFactor": float(metallic_factor),
            "specularFactor": None if specular_factor is None else float(specular_factor),
        },
    }
    if traceability_sidecar is not None:
        root_extras["traceabilitySidecar"] = traceability_sidecar

    document: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": "face3d solid-colored-coordinate-cubes 0.1.0",
        },
        "extensionsUsed": extensions_used,
        "extensionsRequired": ["EXT_mesh_gpu_instancing"],
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "buffers": [{"byteLength": len(builder.payload)}],
        "bufferViews": builder.views,
        "accessors": builder.accessors,
        "extras": root_extras,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(destination, _glb(document, bytes(builder.payload)))
