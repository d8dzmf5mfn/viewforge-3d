from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh


@dataclass(frozen=True, slots=True)
class ConnectedCoordinateUnitSurface:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_normals: np.ndarray
    vertex_colors: np.ndarray
    source_unit_indices: np.ndarray
    metrics: dict[str, Any]


def _edge_metrics(faces: np.ndarray) -> tuple[int, int]:
    edges = np.sort(
        np.concatenate(
            (
                faces[:, (0, 1)],
                faces[:, (1, 2)],
                faces[:, (2, 0)],
            ),
            axis=0,
        ),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts == 1)), int(np.count_nonzero(counts > 2))


def _front_grid_faces(
    vertex_map: np.ndarray,
    front_indices: np.ndarray,
    back_indices: np.ndarray,
) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    for x_index in range(vertex_map.shape[0] - 1):
        for y_index in range(vertex_map.shape[1] - 1):
            coordinates = (
                (x_index, y_index),
                (x_index + 1, y_index),
                (x_index + 1, y_index + 1),
                (x_index, y_index + 1),
            )
            corner_ids = [int(vertex_map[coordinate]) for coordinate in coordinates]
            present = [index for index, value in enumerate(corner_ids) if value >= 0]
            if len(present) == 4:
                first_diagonal = abs(
                    int(front_indices[coordinates[0]]) - int(front_indices[coordinates[2]])
                ) + abs(int(back_indices[coordinates[0]]) - int(back_indices[coordinates[2]]))
                second_diagonal = abs(
                    int(front_indices[coordinates[1]]) - int(front_indices[coordinates[3]])
                ) + abs(int(back_indices[coordinates[1]]) - int(back_indices[coordinates[3]]))
                if first_diagonal <= second_diagonal:
                    faces.extend(
                        (
                            (corner_ids[0], corner_ids[1], corner_ids[2]),
                            (corner_ids[0], corner_ids[2], corner_ids[3]),
                        )
                    )
                else:
                    faces.extend(
                        (
                            (corner_ids[0], corner_ids[1], corner_ids[3]),
                            (corner_ids[1], corner_ids[2], corner_ids[3]),
                        )
                    )
            elif len(present) == 3:
                faces.append(tuple(corner_ids[index] for index in present))
    if not faces:
        raise ValueError("supported coordinate units do not form any surface triangles")
    return np.asarray(faces, dtype=np.uint32)


def _oriented_boundary_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    records: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for first, second, third in faces:
        for start, end in ((first, second), (second, third), (third, first)):
            oriented = (int(start), int(end))
            key = tuple(sorted(oriented))
            count, original = records.get(key, (0, oriented))
            records[key] = (count + 1, original)
    nonmanifold = [key for key, (count, _) in records.items() if count > 2]
    if nonmanifold:
        raise ValueError(f"front unit grid has {len(nonmanifold)} nonmanifold edges")
    return [oriented for count, oriented in records.values() if count == 1]


def connect_depth_envelope_units(
    occupancy: np.ndarray,
    origin: np.ndarray,
    unit_size: float,
    unit_colors: np.ndarray,
) -> ConnectedCoordinateUnitSurface:
    """Join front/back 3D unit coordinates directly; never render the units as cubes."""
    occupancy = np.asarray(occupancy, dtype=bool)
    origin = np.asarray(origin, dtype=np.float64)
    unit_colors = np.asarray(unit_colors, dtype=np.uint8)
    if occupancy.ndim != 3 or not np.any(occupancy):
        raise ValueError("occupancy must be a non-empty 3D boolean array")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin must be a finite three-vector")
    if not np.isfinite(unit_size) or unit_size <= 0:
        raise ValueError("unit_size must be finite and positive")
    if unit_colors.shape != (*occupancy.shape, 3):
        raise ValueError("unit_colors must contain one RGB value per 3D unit coordinate")

    source_support = np.any(occupancy, axis=2)
    back_indices = np.argmax(occupancy, axis=2).astype(np.int32)
    front_indices = (occupancy.shape[2] - 1 - np.argmax(occupancy[..., ::-1], axis=2)).astype(
        np.int32
    )
    depth_span = front_indices - back_indices
    support = source_support & (depth_span >= 1)
    if np.count_nonzero(support) < 3:
        raise ValueError("at least three depth-spanning coordinate columns are required")

    coordinates = np.argwhere(support).astype(np.int32)
    vertex_map = np.full(support.shape, -1, dtype=np.int32)
    vertex_map[tuple(coordinates.T)] = np.arange(len(coordinates), dtype=np.int32)
    front_faces = _front_grid_faces(vertex_map, front_indices, back_indices)
    boundary_edges = _oriented_boundary_edges(front_faces)

    front_unit_indices = np.column_stack((coordinates, front_indices[tuple(coordinates.T)])).astype(
        np.int32
    )
    back_unit_indices = np.column_stack((coordinates, back_indices[tuple(coordinates.T)])).astype(
        np.int32
    )
    source_unit_indices = np.vstack((front_unit_indices, back_unit_indices))
    vertices = origin[None, :] + source_unit_indices.astype(np.float64) * unit_size
    vertex_colors = unit_colors[tuple(source_unit_indices.T)]
    back_offset = len(coordinates)
    back_faces = front_faces[:, ::-1] + back_offset
    wall_faces: list[tuple[int, int, int]] = []
    for start, end in boundary_edges:
        wall_faces.extend(
            (
                (start, start + back_offset, end + back_offset),
                (start, end + back_offset, end),
            )
        )
    faces = np.vstack(
        (
            front_faces,
            back_faces,
            np.asarray(wall_faces, dtype=np.uint32),
        )
    )

    referenced, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.uint32)
    vertices = vertices[referenced]
    vertex_colors = vertex_colors[referenced]
    source_unit_indices = source_unit_indices[referenced]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    if float(mesh.volume) < 0:
        faces = faces[:, (0, 2, 1)]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)

    boundary_edge_count, nonmanifold_edge_count = _edge_metrics(faces)
    triangle_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    grid_coordinates = (vertices - origin[None, :]) / unit_size
    grid_error = float(np.max(np.abs(grid_coordinates - np.rint(grid_coordinates)), initial=0.0))
    occupied_per_column = np.count_nonzero(occupancy, axis=2)
    discontiguous_columns = source_support & (occupied_per_column != depth_span + 1)
    metrics: dict[str, Any] = {
        "operation": "triangulate neighboring front/back custom 3D unit coordinates",
        "inputInterpretation": "custom 3D coordinate units, not visible 2D pixels or cubes",
        "sourceOccupiedUnitCount": int(np.count_nonzero(occupancy)),
        "sourceDepthColumnCount": int(np.count_nonzero(source_support)),
        "connectedDepthColumnCount": int(np.count_nonzero(support)),
        "excludedSingleLayerColumnCount": int(np.count_nonzero(source_support & ~support)),
        "discontiguousSourceColumnCount": int(np.count_nonzero(discontiguous_columns)),
        "vertexCount": int(len(vertices)),
        "triangleCount": int(len(faces)),
        "frontTriangleCount": int(len(front_faces)),
        "backTriangleCount": int(len(back_faces)),
        "sideWallTriangleCount": int(len(wall_faces)),
        "componentCount": int(len(mesh.split(only_watertight=False))),
        "boundaryEdgeCount": boundary_edge_count,
        "nonmanifoldEdgeCount": nonmanifold_edge_count,
        "degenerateTriangleCount": int(np.count_nonzero(triangle_areas <= unit_size**2 * 1e-10)),
        "watertight": bool(mesh.is_watertight),
        "windingConsistent": bool(mesh.is_winding_consistent),
        "positiveVolume": bool(mesh.volume > 0),
        "eulerNumber": int(mesh.euler_number),
        "volume": float(mesh.volume),
        "unitCoordinateMaximumError": grid_error,
        "sourceUnitPositionsMoved": False,
        "surfaceCoordinateInterpolationApplied": False,
        "visibleUnitCubesExported": False,
        "geometricSmoothingApplied": False,
        "laplacianSmoothingApplied": False,
        "taubinSmoothingApplied": False,
        "normalInterpolationApplied": True,
        "subdivisionApplied": False,
        "marchingCubesApplied": False,
    }
    return ConnectedCoordinateUnitSurface(
        vertices=vertices.astype(np.float32),
        faces=faces,
        vertex_normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
        vertex_colors=vertex_colors,
        source_unit_indices=source_unit_indices.astype(np.uint16),
        metrics=metrics,
    )
