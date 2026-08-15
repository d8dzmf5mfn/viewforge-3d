from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from scipy.spatial import Delaunay

from face3d.io import atomic_write_bytes
from face3d.template_head_anatomy import _self_intersection_pairs
from face3d.template_head_v0 import _edge_and_component_metrics
from face3d.unified_head import geometry_hash


@dataclass(frozen=True, slots=True)
class PhoneDimensions:
    width_mm: float
    height_mm: float
    depth_mm: float
    corner_radius_mm: float
    edge_inset_mm: float
    front_glass_width_mm: float
    front_glass_height_mm: float
    camera_plateau_width_mm: float
    camera_plateau_height_mm: float
    camera_plateau_raise_mm: float
    camera_glass_raise_mm: float
    rear_camera_x_mm: float
    rear_camera_main_y_mm: float
    rear_camera_ultrawide_y_mm: float
    rear_camera_outer_diameter_mm: float
    flash_x_mm: float
    flash_y_mm: float
    flash_diameter_mm: float
    dynamic_island_width_mm: float
    dynamic_island_height_mm: float
    dynamic_island_center_y_mm: float

    @classmethod
    def template_phone_v0(cls) -> PhoneDimensions:
        return cls(
            width_mm=72.0,
            height_mm=150.0,
            depth_mm=8.0,
            corner_radius_mm=11.2,
            edge_inset_mm=0.45,
            front_glass_width_mm=69.8,
            front_glass_height_mm=148.0,
            camera_plateau_width_mm=24.0,
            camera_plateau_height_mm=46.0,
            camera_plateau_raise_mm=1.8,
            camera_glass_raise_mm=3.5,
            rear_camera_x_mm=-22.8,
            rear_camera_main_y_mm=52.4,
            rear_camera_ultrawide_y_mm=33.9,
            rear_camera_outer_diameter_mm=16.0,
            flash_x_mm=-4.4,
            flash_y_mm=44.4,
            flash_diameter_mm=6.28,
            dynamic_island_width_mm=20.75,
            dynamic_island_height_mm=5.12,
            dynamic_island_center_y_mm=61.5,
        )

    @classmethod
    def iphone17(cls) -> PhoneDimensions:
        # Width, height, depth, glass, camera and sensor values come from Apple's
        # published 2D specifications and dimensional drawing. The body corner
        # radius and camera plateau outline are image-fit parameters and stay
        # explicitly marked as inferred in the generated manifest.
        return cls(
            width_mm=71.5,
            height_mm=149.6,
            depth_mm=7.95,
            corner_radius_mm=11.1,
            edge_inset_mm=0.45,
            front_glass_width_mm=69.45,
            front_glass_height_mm=147.61,
            camera_plateau_width_mm=24.0,
            camera_plateau_height_mm=46.0,
            camera_plateau_raise_mm=1.78,
            camera_glass_raise_mm=3.45,
            rear_camera_x_mm=13.02 - 35.72,
            rear_camera_main_y_mm=149.6 / 2.0 - 22.48,
            rear_camera_ultrawide_y_mm=149.6 / 2.0 - 40.99,
            rear_camera_outer_diameter_mm=16.0,
            flash_x_mm=31.34 - 35.72,
            flash_y_mm=149.6 / 2.0 - 30.41,
            flash_diameter_mm=6.28,
            dynamic_island_width_mm=20.75,
            dynamic_island_height_mm=5.12,
            dynamic_island_center_y_mm=149.6 / 2.0 - 13.29,
        )

    def as_json(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def uv_hash(uv: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(uv, dtype="<f4").tobytes()).hexdigest()


@dataclass(slots=True)
class TemplatePhoneAsset:
    compute_vertices: np.ndarray
    compute_faces: np.ndarray
    render_to_compute: np.ndarray
    render_faces: np.ndarray
    uv: np.ndarray
    regions: dict[str, np.ndarray]
    dimensions: PhoneDimensions
    geometry_sha256: str
    uv_sha256: str
    metadata: dict[str, Any]

    @property
    def render_vertices(self) -> np.ndarray:
        return self.compute_vertices[self.render_to_compute]

    @property
    def compute_mesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(
            vertices=self.compute_vertices,
            faces=self.compute_faces,
            process=False,
            validate=False,
        )

    @property
    def render_mesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(
            vertices=self.render_vertices,
            faces=self.render_faces,
            process=False,
            validate=False,
        )

    def save(self, destination: Path) -> None:
        output = io.BytesIO()
        np.savez_compressed(
            output,
            compute_vertices=np.asarray(self.compute_vertices, dtype=np.float32),
            compute_faces=np.asarray(self.compute_faces, dtype=np.int32),
            render_to_compute=np.asarray(self.render_to_compute, dtype=np.int32),
            render_faces=np.asarray(self.render_faces, dtype=np.int32),
            uv=np.asarray(self.uv, dtype=np.float32),
            dimensions_json=np.asarray(json.dumps(self.dimensions.as_json(), sort_keys=True)),
            geometry_sha256=np.asarray(self.geometry_sha256),
            uv_sha256=np.asarray(self.uv_sha256),
            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
            **{
                f"region_{name}": np.asarray(indices, dtype=np.int32)
                for name, indices in self.regions.items()
            },
        )
        atomic_write_bytes(destination, output.getvalue())

    @classmethod
    def load(cls, source: Path) -> TemplatePhoneAsset:
        with np.load(source, allow_pickle=False) as payload:
            dimensions = PhoneDimensions(**json.loads(str(payload["dimensions_json"])))
            asset = cls(
                compute_vertices=np.asarray(payload["compute_vertices"], dtype=np.float64),
                compute_faces=np.asarray(payload["compute_faces"], dtype=np.int64),
                render_to_compute=np.asarray(payload["render_to_compute"], dtype=np.int64),
                render_faces=np.asarray(payload["render_faces"], dtype=np.int64),
                uv=np.asarray(payload["uv"], dtype=np.float32),
                regions={
                    name.removeprefix("region_"): np.asarray(payload[name], dtype=np.int64)
                    for name in payload.files
                    if name.startswith("region_")
                },
                dimensions=dimensions,
                geometry_sha256=str(payload["geometry_sha256"]),
                uv_sha256=str(payload["uv_sha256"]),
                metadata=json.loads(str(payload["metadata_json"])),
            )
        if geometry_hash(asset.render_vertices, asset.render_faces) != asset.geometry_sha256:
            raise ValueError("TemplatePhoneV0 geometry hash mismatch")
        if uv_hash(asset.uv) != asset.uv_sha256:
            raise ValueError("TemplatePhoneV0 UV hash mismatch")
        return asset

    def fit_dimensions(self, target: PhoneDimensions) -> TemplatePhoneAsset:
        source = self.dimensions
        scale = np.asarray(
            [
                target.width_mm / source.width_mm,
                target.height_mm / source.height_mm,
                target.depth_mm / source.depth_mm,
            ],
            dtype=np.float64,
        )
        fitted_vertices = np.asarray(self.compute_vertices, dtype=np.float64) * scale
        fitted_hash = geometry_hash(fitted_vertices[self.render_to_compute], self.render_faces)
        return TemplatePhoneAsset(
            compute_vertices=fitted_vertices,
            compute_faces=self.compute_faces.copy(),
            render_to_compute=self.render_to_compute.copy(),
            render_faces=self.render_faces.copy(),
            uv=self.uv.copy(),
            regions={name: indices.copy() for name, indices in self.regions.items()},
            dimensions=target,
            geometry_sha256=fitted_hash,
            uv_sha256=self.uv_sha256,
            metadata={
                **self.metadata,
                "state": "fitted-preview",
                "sourceGeometrySha256": self.geometry_sha256,
                "fit": {
                    "method": "bounded-axis-scale-from-2d-dimensional-evidence",
                    "scale": scale.astype(float).tolist(),
                    "topologyChanged": False,
                    "uvChanged": False,
                },
            },
        )


def _rounded_rectangle_perimeter(
    width: float,
    height: float,
    radius: float,
    *,
    corner_samples: int,
    long_edge_samples: int,
    short_edge_samples: int,
) -> np.ndarray:
    half_width = width / 2.0
    half_height = height / 2.0
    radius = float(np.clip(radius, 1e-3, min(half_width, half_height) - 1e-3))
    points: list[tuple[float, float]] = []

    def arc(center: tuple[float, float], start: float, end: float) -> None:
        for angle in np.linspace(start, end, corner_samples, endpoint=False):
            points.append(
                (
                    center[0] + radius * float(np.cos(angle)),
                    center[1] + radius * float(np.sin(angle)),
                )
            )

    arc((half_width - radius, half_height - radius), 0.0, np.pi / 2.0)
    for x in np.linspace(
        half_width - radius,
        -half_width + radius,
        short_edge_samples,
        endpoint=False,
    ):
        points.append((float(x), half_height))
    arc((-half_width + radius, half_height - radius), np.pi / 2.0, np.pi)
    for y in np.linspace(
        half_height - radius,
        -half_height + radius,
        long_edge_samples,
        endpoint=False,
    ):
        points.append((-half_width, float(y)))
    arc((-half_width + radius, -half_height + radius), np.pi, 3.0 * np.pi / 2.0)
    for x in np.linspace(
        -half_width + radius,
        half_width - radius,
        short_edge_samples,
        endpoint=False,
    ):
        points.append((float(x), -half_height))
    arc((half_width - radius, -half_height + radius), 3.0 * np.pi / 2.0, 2.0 * np.pi)
    for y in np.linspace(
        -half_height + radius,
        half_height - radius,
        long_edge_samples,
        endpoint=False,
    ):
        points.append((half_width, float(y)))
    return np.asarray(points, dtype=np.float64)


def _inside_rounded_rectangle(
    x: float,
    y: float,
    width: float,
    height: float,
    radius: float,
    *,
    margin: float,
) -> bool:
    half_width = width / 2.0 - margin
    half_height = height / 2.0 - margin
    radius = max(radius - margin, 1e-3)
    if half_width <= 0.0 or half_height <= 0.0:
        return False
    qx = abs(x) - (half_width - radius)
    qy = abs(y) - (half_height - radius)
    outside = np.hypot(max(qx, 0.0), max(qy, 0.0))
    return bool(max(qx, qy) <= 0.0 or outside <= radius)


def _cap_interior_points(
    width: float,
    height: float,
    radius: float,
    spacing: float,
) -> np.ndarray:
    points: list[tuple[float, float]] = []
    margin = spacing * 0.32
    for y in np.arange(-height / 2.0 + spacing, height / 2.0, spacing):
        for x in np.arange(-width / 2.0 + spacing, width / 2.0, spacing):
            if _inside_rounded_rectangle(
                float(x),
                float(y),
                width,
                height,
                radius,
                margin=margin,
            ):
                points.append((float(x), float(y)))
    if not points:
        points.append((0.0, 0.0))
    return np.asarray(points, dtype=np.float64)


def _triangulate_cap(perimeter: np.ndarray, interior: np.ndarray, *, front: bool) -> np.ndarray:
    points = np.concatenate((perimeter, interior), axis=0)
    faces = np.asarray(Delaunay(points).simplices, dtype=np.int64)
    triangles = points[faces]
    signed = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    want_positive = front
    flip = signed < 0.0 if want_positive else signed > 0.0
    faces[flip, 1], faces[flip, 2] = faces[flip, 2].copy(), faces[flip, 1].copy()
    return faces


def _build_compute_geometry(
    dimensions: PhoneDimensions,
    *,
    corner_samples: int = 22,
    long_edge_samples: int = 42,
    short_edge_samples: int = 20,
    cap_spacing_mm: float = 4.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[np.ndarray],
]:
    depth = dimensions.depth_mm
    first_bevel_depth = min(0.25, depth * 0.20)
    second_bevel_depth = min(0.70, depth * 0.42)
    z_levels = np.asarray(
        [
            -depth / 2.0,
            -depth / 2.0 + first_bevel_depth,
            -depth / 2.0 + second_bevel_depth,
            0.0,
            depth / 2.0 - second_bevel_depth,
            depth / 2.0 - first_bevel_depth,
            depth / 2.0,
        ],
        dtype=np.float64,
    )
    insets = np.asarray(
        [
            dimensions.edge_inset_mm,
            dimensions.edge_inset_mm * 0.42,
            0.0,
            0.0,
            0.0,
            dimensions.edge_inset_mm * 0.42,
            dimensions.edge_inset_mm,
        ],
        dtype=np.float64,
    )
    rings = [
        _rounded_rectangle_perimeter(
            dimensions.width_mm - 2.0 * inset,
            dimensions.height_mm - 2.0 * inset,
            dimensions.corner_radius_mm - inset,
            corner_samples=corner_samples,
            long_edge_samples=long_edge_samples,
            short_edge_samples=short_edge_samples,
        )
        for inset in insets
    ]
    perimeter_count = len(rings[0])
    if any(len(ring) != perimeter_count for ring in rings):
        raise AssertionError("rounded rectangle ring topology changed")

    vertex_blocks = [
        np.column_stack((ring, np.full(perimeter_count, z, dtype=np.float64)))
        for ring, z in zip(rings, z_levels, strict=True)
    ]
    back_interior_xy = _cap_interior_points(
        dimensions.width_mm - 2.0 * insets[0],
        dimensions.height_mm - 2.0 * insets[0],
        dimensions.corner_radius_mm - insets[0],
        cap_spacing_mm,
    )
    front_interior_xy = back_interior_xy.copy()
    back_interior_start = len(z_levels) * perimeter_count
    front_interior_start = back_interior_start + len(back_interior_xy)
    vertex_blocks.extend(
        (
            np.column_stack(
                (
                    back_interior_xy,
                    np.full(len(back_interior_xy), z_levels[0], dtype=np.float64),
                )
            ),
            np.column_stack(
                (
                    front_interior_xy,
                    np.full(len(front_interior_xy), z_levels[-1], dtype=np.float64),
                )
            ),
        )
    )
    vertices = np.concatenate(vertex_blocks, axis=0)

    back_local_faces = _triangulate_cap(rings[0], back_interior_xy, front=False)
    back_map = np.concatenate(
        (
            np.arange(perimeter_count, dtype=np.int64),
            np.arange(
                back_interior_start,
                back_interior_start + len(back_interior_xy),
                dtype=np.int64,
            ),
        )
    )
    back_faces = back_map[back_local_faces]

    side_faces: list[tuple[int, int, int]] = []
    for layer in range(len(z_levels) - 1):
        for index in range(perimeter_count):
            following = (index + 1) % perimeter_count
            back_a = layer * perimeter_count + index
            back_b = layer * perimeter_count + following
            front_b = (layer + 1) * perimeter_count + following
            front_a = (layer + 1) * perimeter_count + index
            side_faces.extend(((back_a, back_b, front_b), (back_a, front_b, front_a)))
    side_faces_array = np.asarray(side_faces, dtype=np.int64)

    front_local_faces = _triangulate_cap(rings[-1], front_interior_xy, front=True)
    front_boundary_start = (len(z_levels) - 1) * perimeter_count
    front_map = np.concatenate(
        (
            np.arange(
                front_boundary_start,
                front_boundary_start + perimeter_count,
                dtype=np.int64,
            ),
            np.arange(
                front_interior_start,
                front_interior_start + len(front_interior_xy),
                dtype=np.int64,
            ),
        )
    )
    front_faces = front_map[front_local_faces]
    faces = np.concatenate((back_faces, side_faces_array, front_faces), axis=0)
    face_groups = {
        "rear": np.arange(len(back_faces), dtype=np.int64),
        "frame": np.arange(
            len(back_faces),
            len(back_faces) + len(side_faces_array),
            dtype=np.int64,
        ),
        "front": np.arange(
            len(back_faces) + len(side_faces_array),
            len(faces),
            dtype=np.int64,
        ),
    }

    frame_vertices = np.unique(side_faces_array)
    tolerance = 0.8
    regions = {
        "rear_surface": np.unique(back_faces),
        "frame": frame_vertices,
        "front_surface": np.unique(front_faces),
        "left_edge": frame_vertices[
            vertices[frame_vertices, 0] <= vertices[frame_vertices, 0].min() + tolerance
        ],
        "right_edge": frame_vertices[
            vertices[frame_vertices, 0] >= vertices[frame_vertices, 0].max() - tolerance
        ],
        "top_edge": frame_vertices[
            vertices[frame_vertices, 1] >= vertices[frame_vertices, 1].max() - tolerance
        ],
        "bottom_edge": frame_vertices[
            vertices[frame_vertices, 1] <= vertices[frame_vertices, 1].min() + tolerance
        ],
    }
    rear_vertices = regions["rear_surface"]
    rear_points = vertices[rear_vertices]
    camera_zone = (
        (rear_points[:, 0] >= -dimensions.width_mm * 0.49)
        & (rear_points[:, 0] <= -dimensions.width_mm * 0.12)
        & (rear_points[:, 1] >= dimensions.height_mm * 0.10)
        & (rear_points[:, 1] <= dimensions.height_mm * 0.49)
    )
    regions["camera_attachment"] = rear_vertices[camera_zone]
    return vertices, faces, regions, face_groups, rings


def _append_planar_render_group(
    vertices: np.ndarray,
    group_faces: np.ndarray,
    *,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    width: float,
    height: float,
    render_to_compute: list[int],
    render_faces: list[tuple[int, int, int]],
    render_uv: list[tuple[float, float]],
) -> None:
    compute_indices = np.unique(group_faces)
    base = len(render_to_compute)
    local = {int(value): base + index for index, value in enumerate(compute_indices)}
    render_to_compute.extend(int(value) for value in compute_indices)
    render_uv.extend(
        (
            float(u_min + (vertices[index, 0] / width + 0.5) * (u_max - u_min)),
            float(v_min + (vertices[index, 1] / height + 0.5) * (v_max - v_min)),
        )
        for index in compute_indices
    )
    render_faces.extend(tuple(local[int(value)] for value in face) for face in group_faces)


def _render_topology(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_groups: dict[str, np.ndarray],
    rings: list[np.ndarray],
    dimensions: PhoneDimensions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    render_to_compute: list[int] = []
    render_faces: list[tuple[int, int, int]] = []
    render_uv: list[tuple[float, float]] = []
    _append_planar_render_group(
        vertices,
        faces[face_groups["rear"]],
        u_min=0.40,
        u_max=0.74,
        v_min=0.02,
        v_max=0.76,
        width=dimensions.width_mm,
        height=dimensions.height_mm,
        render_to_compute=render_to_compute,
        render_faces=render_faces,
        render_uv=render_uv,
    )

    perimeter_count = len(rings[0])
    layer_count = len(rings)
    ring_distances = np.linalg.norm(
        np.roll(rings[3], -1, axis=0) - rings[3],
        axis=1,
    )
    perimeter_u = np.concatenate(
        (
            np.asarray([0.0]),
            np.cumsum(ring_distances[:-1]),
        )
    )
    perimeter_u /= float(np.sum(ring_distances))
    side_map = np.empty((layer_count, perimeter_count + 1), dtype=np.int64)
    for layer in range(layer_count):
        v = 0.80 + 0.18 * layer / (layer_count - 1)
        for index in range(perimeter_count + 1):
            compute_index = layer * perimeter_count + (index % perimeter_count)
            side_map[layer, index] = len(render_to_compute)
            render_to_compute.append(compute_index)
            render_uv.append((1.0 if index == perimeter_count else float(perimeter_u[index]), v))
    for layer in range(layer_count - 1):
        for index in range(perimeter_count):
            a = int(side_map[layer, index])
            b = int(side_map[layer, index + 1])
            c = int(side_map[layer + 1, index + 1])
            d = int(side_map[layer + 1, index])
            render_faces.extend(((a, b, c), (a, c, d)))

    _append_planar_render_group(
        vertices,
        faces[face_groups["front"]],
        u_min=0.02,
        u_max=0.36,
        v_min=0.02,
        v_max=0.76,
        width=dimensions.width_mm,
        height=dimensions.height_mm,
        render_to_compute=render_to_compute,
        render_faces=render_faces,
        render_uv=render_uv,
    )
    mapping = np.asarray(render_to_compute, dtype=np.int64)
    render_faces_array = np.asarray(render_faces, dtype=np.int64)
    uv = np.asarray(render_uv, dtype=np.float32)
    if not np.array_equal(mapping[render_faces_array], faces):
        raise AssertionError("render-to-compute mapping changed primary topology")
    return mapping, render_faces_array, uv


def build_template_phone_v0() -> TemplatePhoneAsset:
    dimensions = PhoneDimensions.template_phone_v0()
    vertices, faces, regions, face_groups, rings = _build_compute_geometry(dimensions)
    render_to_compute, render_faces, uv = _render_topology(
        vertices,
        faces,
        face_groups,
        rings,
        dimensions,
    )
    digest = geometry_hash(vertices[render_to_compute], render_faces)
    return TemplatePhoneAsset(
        compute_vertices=vertices,
        compute_faces=faces,
        render_to_compute=render_to_compute,
        render_faces=render_faces,
        uv=uv,
        regions=regions,
        dimensions=dimensions,
        geometry_sha256=digest,
        uv_sha256=uv_hash(uv),
        metadata={
            "schemaVersion": 1,
            "templateId": "TemplatePhoneV0",
            "targetClass": "smartphone",
            "state": "template-candidate",
            "origin": "project-authored-no-external-3d-input",
            "surfaceLineage": "TemplatePhoneV0 -> fitted -> attached features -> QA",
            "sdf": {"role": "qa-only", "surfaceGenerated": False},
            "uv": {"method": "fixed-semantic-atlas", "seamMappingRequired": True},
        },
    )


def _component_mesh(
    width: float,
    height: float,
    depth: float,
    radius: float,
) -> trimesh.Trimesh:
    component_dimensions = replace(
        PhoneDimensions.template_phone_v0(),
        width_mm=width,
        height_mm=height,
        depth_mm=depth,
        corner_radius_mm=min(radius, width / 2.0 - 1e-3, height / 2.0 - 1e-3),
        edge_inset_mm=min(0.08, depth * 0.1),
    )
    vertices, faces, _, _, _ = _build_compute_geometry(
        component_dimensions,
        corner_samples=10,
        long_edge_samples=12,
        short_edge_samples=8,
        cap_spacing_mm=max(min(width, height) / 3.0, 0.8),
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)


def _material(
    mesh: trimesh.Trimesh,
    *,
    name: str,
    color: tuple[float, float, float, float],
    metallic: float,
    roughness: float,
    uv: np.ndarray | None = None,
) -> trimesh.Trimesh:
    material = trimesh.visual.material.PBRMaterial(
        name=name,
        baseColorFactor=color,
        metallicFactor=metallic,
        roughnessFactor=roughness,
        doubleSided=False,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    mesh.metadata["qaColor"] = list(color)
    return mesh


def _translated(mesh: trimesh.Trimesh, offset: tuple[float, float, float]) -> trimesh.Trimesh:
    result = mesh.copy()
    result.apply_translation(offset)
    return result


def _side_button(
    *,
    side: str,
    center_y: float,
    length: float,
    width: float,
    dimensions: PhoneDimensions,
) -> trimesh.Trimesh:
    mesh = _component_mesh(width, length, 0.46, min(width / 2.0, 1.0))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, (0, 1, 0)))
    sign = -1.0 if side == "left" else 1.0
    mesh.apply_translation((sign * (dimensions.width_mm / 2.0 + 0.23), center_y, 0.0))
    return mesh


def build_phone_scene(
    asset: TemplatePhoneAsset,
    dimensions: PhoneDimensions,
) -> trimesh.Scene:
    lavender = (0.72, 0.64, 0.82, 1.0)
    lavender_glass = (0.82, 0.76, 0.90, 1.0)
    black_glass = (0.012, 0.016, 0.024, 1.0)
    camera_glass = (0.018, 0.024, 0.034, 1.0)
    body = _material(
        asset.render_mesh,
        name="PhoneV1.LavenderFrame",
        color=lavender,
        metallic=0.58,
        roughness=0.30,
        uv=asset.uv,
    )
    body.metadata.update(
        {
            "name": "PhonePrimarySurface",
            "geometryHash": asset.geometry_sha256,
            "uvHash": asset.uv_sha256,
            "surfaceSource": "deformed-template",
        }
    )

    display = _component_mesh(
        dimensions.front_glass_width_mm,
        dimensions.front_glass_height_mm,
        0.16,
        max(dimensions.corner_radius_mm - 0.65, 1.0),
    )
    display = _translated(display, (0.0, 0.0, dimensions.depth_mm / 2.0 + 0.08))
    _material(
        display,
        name="PhoneV1.DisplayGlass",
        color=(0.035, 0.045, 0.075, 1.0),
        metallic=0.02,
        roughness=0.12,
    )

    island = _component_mesh(
        dimensions.dynamic_island_width_mm,
        dimensions.dynamic_island_height_mm,
        0.08,
        dimensions.dynamic_island_height_mm / 2.0,
    )
    island = _translated(
        island,
        (
            0.0,
            dimensions.dynamic_island_center_y_mm,
            dimensions.depth_mm / 2.0 + 0.20,
        ),
    )
    _material(
        island,
        name="PhoneV1.DynamicIsland",
        color=black_glass,
        metallic=0.0,
        roughness=0.08,
    )

    plateau = _component_mesh(
        dimensions.camera_plateau_width_mm,
        dimensions.camera_plateau_height_mm,
        dimensions.camera_plateau_raise_mm,
        dimensions.camera_plateau_width_mm / 2.0,
    )
    plateau_center_y = (
        dimensions.rear_camera_main_y_mm + dimensions.rear_camera_ultrawide_y_mm
    ) / 2.0
    plateau = _translated(
        plateau,
        (
            dimensions.rear_camera_x_mm,
            plateau_center_y,
            -dimensions.depth_mm / 2.0 - dimensions.camera_plateau_raise_mm / 2.0,
        ),
    )
    _material(
        plateau,
        name="PhoneV1.CameraPlateau",
        color=lavender_glass,
        metallic=0.12,
        roughness=0.22,
    )

    scene = trimesh.Scene()
    scene.metadata.update(
        {
            "geometryHash": asset.geometry_sha256,
            "uvHash": asset.uv_sha256,
            "templateId": "TemplatePhoneV0",
            "state": asset.metadata.get("state", "preview"),
        }
    )
    scene.add_geometry(body, node_name="PhonePrimarySurface", geom_name="PhonePrimarySurface")
    scene.add_geometry(display, node_name="DisplayGlass", geom_name="DisplayGlass")
    scene.add_geometry(island, node_name="DynamicIsland", geom_name="DynamicIsland")
    scene.add_geometry(plateau, node_name="CameraPlateau", geom_name="CameraPlateau")

    lens_extra = max(
        dimensions.camera_glass_raise_mm - dimensions.camera_plateau_raise_mm,
        0.2,
    )
    lens_z = -dimensions.depth_mm / 2.0 - dimensions.camera_plateau_raise_mm - lens_extra / 2.0
    for name, center_y in (
        ("Main", dimensions.rear_camera_main_y_mm),
        ("UltraWide", dimensions.rear_camera_ultrawide_y_mm),
    ):
        ring = trimesh.creation.cylinder(
            radius=dimensions.rear_camera_outer_diameter_mm / 2.0,
            height=lens_extra,
            sections=64,
        )
        ring.apply_translation((dimensions.rear_camera_x_mm, center_y, lens_z))
        _material(
            ring,
            name=f"PhoneV1.Camera.{name}.Ring",
            color=lavender,
            metallic=0.75,
            roughness=0.22,
        )
        glass = trimesh.creation.cylinder(
            radius=dimensions.rear_camera_outer_diameter_mm * 0.39,
            height=0.10,
            sections=64,
        )
        glass.apply_translation(
            (
                dimensions.rear_camera_x_mm,
                center_y,
                -dimensions.depth_mm / 2.0 - dimensions.camera_glass_raise_mm - 0.05,
            )
        )
        _material(
            glass,
            name=f"PhoneV1.Camera.{name}.Glass",
            color=camera_glass,
            metallic=0.0,
            roughness=0.08,
        )
        scene.add_geometry(
            ring,
            node_name=f"Camera.{name}.Ring",
            geom_name=f"Camera.{name}.Ring",
        )
        scene.add_geometry(
            glass,
            node_name=f"Camera.{name}.Glass",
            geom_name=f"Camera.{name}.Glass",
        )

    flash = trimesh.creation.cylinder(
        radius=dimensions.flash_diameter_mm / 2.0,
        height=0.18,
        sections=48,
    )
    flash.apply_translation(
        (
            dimensions.flash_x_mm,
            dimensions.flash_y_mm,
            -dimensions.depth_mm / 2.0 - 0.09,
        )
    )
    _material(
        flash,
        name="PhoneV1.Flash",
        color=(0.92, 0.88, 0.70, 1.0),
        metallic=0.0,
        roughness=0.25,
    )
    scene.add_geometry(flash, node_name="Flash", geom_name="Flash")

    microphone = trimesh.creation.cylinder(radius=0.50, height=0.12, sections=32)
    microphone.apply_translation(
        (
            dimensions.rear_camera_x_mm + 7.1,
            dimensions.rear_camera_main_y_mm - 9.4,
            -dimensions.depth_mm / 2.0 - 0.06,
        )
    )
    _material(
        microphone,
        name="PhoneV1.RearMic",
        color=black_glass,
        metallic=0.0,
        roughness=0.35,
    )
    scene.add_geometry(microphone, node_name="RearMic", geom_name="RearMic")

    buttons = (
        ("ActionButton", "left", 31.5, 8.4, 2.2),
        ("VolumeUp", "left", 14.0, 12.2, 2.2),
        ("VolumeDown", "left", -4.0, 12.2, 2.2),
        ("SideButton", "right", 22.0, 18.0, 2.2),
        ("CameraControl", "right", -37.0, 17.5, 2.2),
    )
    for name, side, center_y, length, width in buttons:
        button = _side_button(
            side=side,
            center_y=center_y,
            length=length,
            width=width,
            dimensions=dimensions,
        )
        _material(
            button,
            name=f"PhoneV1.{name}",
            color=lavender,
            metallic=0.70,
            roughness=0.28,
        )
        scene.add_geometry(button, node_name=name, geom_name=name)

    port = _component_mesh(12.45, 2.2, 0.12, 1.1)
    port.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, (1, 0, 0)))
    port.apply_translation((0.0, -dimensions.height_mm / 2.0 - 0.06, 0.0))
    _material(
        port,
        name="PhoneV1.UsbCReferenceInset",
        color=black_glass,
        metallic=0.0,
        roughness=0.38,
    )
    port.metadata["approximation"] = "appearance-only-recess-no-primary-boolean"
    scene.add_geometry(port, node_name="UsbCReferenceInset", geom_name="UsbCReferenceInset")
    return scene


def export_phone_glb(
    asset: TemplatePhoneAsset,
    dimensions: PhoneDimensions,
    destination: Path,
) -> None:
    scene = build_phone_scene(asset, dimensions)
    atomic_write_bytes(
        destination,
        trimesh.exchange.gltf.export_glb(scene, include_normals=True),
    )


FIXED_VIEWS: dict[str, tuple[float, float]] = {
    "front": (0.0, 0.0),
    "front-left45": (-45.0, 0.0),
    "front-right45": (45.0, 0.0),
    "back": (180.0, 0.0),
    "left": (-90.0, 0.0),
    "right": (90.0, 0.0),
    "top": (0.0, 90.0),
    "bottom": (0.0, -90.0),
    "orbit": (35.0, 18.0),
    "back-orbit": (215.0, 18.0),
}


def _rotation(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_degrees)
    pitch = np.deg2rad(pitch_degrees)
    rotate_y = np.asarray(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float64,
    )
    rotate_x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    return rotate_x @ rotate_y


def _rasterize_triangle(
    canvas: np.ndarray,
    depth_buffer: np.ndarray,
    points: np.ndarray,
    depths: np.ndarray,
    color: np.ndarray,
) -> None:
    height, width = depth_buffer.shape
    x_min = max(int(np.floor(points[:, 0].min())), 0)
    x_max = min(int(np.ceil(points[:, 0].max())), width - 1)
    y_min = max(int(np.floor(points[:, 1].min())), 0)
    y_max = min(int(np.ceil(points[:, 1].max())), height - 1)
    if x_min > x_max or y_min > y_max:
        return

    x0, y0 = points[0]
    x1, y1 = points[1]
    x2, y2 = points[2]
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denominator) <= 1e-12:
        return
    x_coordinates = np.arange(x_min, x_max + 1, dtype=np.float64) + 0.5
    y_coordinates = np.arange(y_min, y_max + 1, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates)
    weight_0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denominator
    weight_1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denominator
    weight_2 = 1.0 - weight_0 - weight_1
    inside = (weight_0 >= -1e-8) & (weight_1 >= -1e-8) & (weight_2 >= -1e-8)
    if not np.any(inside):
        return
    interpolated_depth = weight_0 * depths[0] + weight_1 * depths[1] + weight_2 * depths[2]
    region_depth = depth_buffer[y_min : y_max + 1, x_min : x_max + 1]
    nearer = inside & (interpolated_depth > region_depth)
    if not np.any(nearer):
        return
    region_depth[nearer] = interpolated_depth[nearer]
    region_canvas = canvas[y_min : y_max + 1, x_min : x_max + 1]
    region_canvas[nearer] = color


def render_phone_scene(
    scene: trimesh.Scene,
    destination: Path,
    *,
    view: str,
    width: int = 1024,
    height: int = 1024,
) -> None:
    if view not in FIXED_VIEWS:
        raise ValueError(f"unknown fixed phone view: {view}")
    rotation = _rotation(*FIXED_VIEWS[view])
    geometries = list(scene.geometry.values())
    transformed_vertices = [np.asarray(mesh.vertices) @ rotation.T for mesh in geometries]
    all_vertices = np.concatenate(transformed_vertices, axis=0)
    minimum = all_vertices[:, :2].min(axis=0)
    maximum = all_vertices[:, :2].max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    scale = min(width * 0.86 / span[0], height * 0.86 / span[1])
    center = (minimum + maximum) / 2.0
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (22, 24, 29)
    depth_buffer = np.full((height, width), -np.inf, dtype=np.float64)
    light = np.asarray([-0.35, 0.55, 0.76], dtype=np.float64)
    light /= np.linalg.norm(light)
    for mesh, vertices in zip(geometries, transformed_vertices, strict=True):
        faces = np.asarray(mesh.faces, dtype=np.int64)
        triangles = vertices[faces]
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        normal_length = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        normals /= normal_length
        visible = normals[:, 2] > 1e-9
        faces = faces[visible]
        triangles = triangles[visible]
        normals = normals[visible]
        if len(faces) == 0:
            continue
        base_rgba = np.asarray(mesh.metadata.get("qaColor", [0.62, 0.64, 0.70, 1.0]))
        base_bgr = base_rgba[:3][::-1] * 255.0
        projected = np.empty((len(vertices), 2), dtype=np.float64)
        projected[:, 0] = (vertices[:, 0] - center[0]) * scale + width / 2.0
        projected[:, 1] = -(vertices[:, 1] - center[1]) * scale + height / 2.0
        pixel_triangles = projected[faces]
        intensity = np.clip(0.32 + 0.68 * np.abs(normals @ light), 0.0, 1.0)
        for index, triangle in enumerate(pixel_triangles):
            color = np.clip(base_bgr * intensity[index], 0.0, 255.0).astype(np.uint8)
            _rasterize_triangle(
                canvas,
                depth_buffer,
                triangle,
                triangles[index, :, 2],
                color,
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise OSError(f"failed to write render: {destination}")


def primary_surface_metrics(asset: TemplatePhoneAsset) -> dict[str, Any]:
    vertices = np.asarray(asset.compute_vertices, dtype=np.float64)
    faces = np.asarray(asset.compute_faces, dtype=np.int64)
    mesh = asset.compute_mesh
    topology = _edge_and_component_metrics(vertices, faces)
    intersections = _self_intersection_pairs(mesh)
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    centers = triangles.mean(axis=1)
    outward = np.einsum("ij,ij->i", normals, centers - vertices.mean(axis=0))
    inverted = int(np.count_nonzero(outward <= 0.0))
    render_difference = np.max(
        np.abs(asset.render_vertices - vertices[asset.render_to_compute]),
        initial=0.0,
    )
    return {
        "schemaVersion": 1,
        "templateId": "TemplatePhoneV0",
        "geometryHash": asset.geometry_sha256,
        "uvHash": asset.uv_sha256,
        "vertices": int(len(vertices)),
        "triangles": int(len(faces)),
        "topology": {
            "connectedComponents": int(topology["componentCount"]),
            "boundaryEdges": int(topology["boundaryEdgeCount"]),
            "nonManifoldEdges": int(topology["nonManifoldEdgeCount"]),
            "degenerateFaces": int(topology["degenerateFaceCount"]),
            "duplicateFaces": int(topology["duplicateFaceCount"]),
            "duplicateVertices": int(topology["duplicateVertexCount"]),
            "watertight": bool(topology["watertight"]),
            "windingConsistent": bool(topology["windingConsistent"]),
            "invertedFaces": inverted,
            "selfIntersectionPairs": int(len(intersections)),
        },
        "renderToComputeMaximumDifference": float(render_difference),
        "uvMinimum": np.min(asset.uv, axis=0).astype(float).tolist(),
        "uvMaximum": np.max(asset.uv, axis=0).astype(float).tolist(),
        "sdf": {"role": "qa-only", "surfaceGenerated": False},
        "passed": bool(
            topology["componentCount"] == 1
            and topology["boundaryEdgeCount"] == 0
            and topology["nonManifoldEdgeCount"] == 0
            and topology["degenerateFaceCount"] == 0
            and topology["duplicateFaceCount"] == 0
            and topology["duplicateVertexCount"] == 0
            and topology["watertight"]
            and topology["windingConsistent"]
            and inverted == 0
            and len(intersections) == 0
            and render_difference == 0.0
        ),
    }
