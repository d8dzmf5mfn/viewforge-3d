from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

_CUBE_FACES = (
    ((1, 0, 0), ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1))),
    ((-1, 0, 0), ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1))),
    ((0, 1, 0), ((-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1))),
    ((0, -1, 0), ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1))),
    ((0, 0, 1), ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))),
    ((0, 0, -1), ((-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1))),
)


@dataclass(frozen=True, slots=True)
class ConnectedVoxelSurface:
    vertices: np.ndarray
    faces: np.ndarray
    face_colors: np.ndarray
    vertex_colors: np.ndarray
    face_cell_indices: np.ndarray
    metrics: dict[str, Any]


def _nonmanifold_contacts(surface: ConnectedVoxelSurface) -> list[np.ndarray]:
    faces = surface.faces
    face_ids = np.tile(np.arange(len(faces), dtype=np.int32), 3)
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
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    contacts: list[np.ndarray] = []
    for edge_id in np.flatnonzero(counts > 2):
        incident_faces = face_ids[inverse == edge_id]
        contacts.append(np.unique(surface.face_cell_indices[incident_faces], axis=0))
    return contacts


def _neighbor_score(occupancy: np.ndarray, coordinate: np.ndarray) -> int:
    score = 0
    for axis in range(3):
        for offset in (-1, 1):
            neighbor = coordinate.copy()
            neighbor[axis] += offset
            if np.all((neighbor >= 0) & (neighbor < occupancy.shape)):
                score += int(occupancy[tuple(neighbor)])
    return score


def repair_nonmanifold_voxel_contacts(
    occupancy: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    cell_colors: np.ndarray,
    *,
    maximum_added_cells: int = 64,
    maximum_passes: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fill the minimum integer-grid bridge cells needed to remove diagonal edge contacts."""
    repaired = np.asarray(occupancy, dtype=bool).copy()
    colors = np.asarray(cell_colors, dtype=np.uint8).copy()
    added: list[tuple[int, int, int]] = []
    pass_records: list[dict[str, Any]] = []
    initial_surface = connect_voxel_surface(repaired, origin, voxel_size, colors)
    initial_nonmanifold = initial_surface.metrics["nonmanifoldEdgeCount"]
    for pass_index in range(maximum_passes):
        surface = connect_voxel_surface(repaired, origin, voxel_size, colors)
        contacts = _nonmanifold_contacts(surface)
        if not contacts:
            break
        proposals: dict[tuple[int, int, int], list[np.ndarray]] = {}
        for cells in contacts:
            candidates: list[tuple[int, tuple[int, int, int], np.ndarray]] = []
            for first_index in range(len(cells)):
                for second_index in range(first_index + 1, len(cells)):
                    first = cells[first_index].astype(np.int32)
                    second = cells[second_index].astype(np.int32)
                    difference = second - first
                    axes = np.flatnonzero(difference)
                    if len(axes) != 2 or np.any(np.abs(difference[axes]) != 1):
                        continue
                    for axis in axes:
                        candidate = first.copy()
                        candidate[axis] += difference[axis]
                        if (
                            np.all((candidate >= 0) & (candidate < repaired.shape))
                            and not repaired[tuple(candidate)]
                        ):
                            candidates.append(
                                (
                                    _neighbor_score(repaired, candidate),
                                    tuple(int(value) for value in candidate),
                                    np.rint(
                                        (
                                            colors[tuple(first)].astype(np.float64)
                                            + colors[tuple(second)].astype(np.float64)
                                        )
                                        * 0.5
                                    ).astype(np.uint8),
                                )
                            )
            if not candidates:
                continue
            _, coordinate, color = max(candidates, key=lambda value: (value[0], value[1]))
            proposals.setdefault(coordinate, []).append(color)
        if not proposals:
            break
        if len(added) + len(proposals) > maximum_added_cells:
            raise ValueError("nonmanifold contact repair exceeds the added-cell budget")
        applied: list[list[int]] = []
        for coordinate, candidate_colors in proposals.items():
            repaired[coordinate] = True
            colors[coordinate] = np.rint(
                np.mean(np.asarray(candidate_colors, dtype=np.float64), axis=0)
            ).astype(np.uint8)
            if coordinate not in added:
                added.append(coordinate)
                applied.append(list(coordinate))
        pass_records.append(
            {
                "pass": pass_index + 1,
                "nonmanifoldEdgesBefore": int(len(contacts)),
                "addedCells": applied,
            }
        )
    final_surface = connect_voxel_surface(repaired, origin, voxel_size, colors)
    return (
        repaired,
        colors,
        {
            "method": "integer-grid bridge fill for diagonal voxel edge contacts",
            "initialNonmanifoldEdgeCount": int(initial_nonmanifold),
            "finalNonmanifoldEdgeCount": int(final_surface.metrics["nonmanifoldEdgeCount"]),
            "addedCellCount": int(len(added)),
            "addedCells": [list(value) for value in added],
            "maximumAddedCells": int(maximum_added_cells),
            "passes": pass_records,
            "vertexPositionsMoved": 0,
            "smoothingApplied": False,
        },
    )


def _exposed_cells(occupancy: np.ndarray, direction: tuple[int, int, int]) -> np.ndarray:
    axis = next(index for index, value in enumerate(direction) if value)
    neighbor = np.zeros_like(occupancy)
    source = [slice(None)] * 3
    target = [slice(None)] * 3
    if direction[axis] > 0:
        source[axis] = slice(1, None)
        target[axis] = slice(None, -1)
    else:
        source[axis] = slice(None, -1)
        target[axis] = slice(1, None)
    neighbor[tuple(target)] = occupancy[tuple(source)]
    return np.argwhere(occupancy & ~neighbor)


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


def connect_voxel_surface(
    occupancy: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    cell_colors: np.ndarray,
) -> ConnectedVoxelSurface:
    """Connect exposed voxel faces exactly, without filtering or moving a grid corner."""
    occupancy = np.asarray(occupancy, dtype=bool)
    origin = np.asarray(origin, dtype=np.float64)
    cell_colors = np.asarray(cell_colors, dtype=np.uint8)
    if occupancy.ndim != 3 or not np.any(occupancy):
        raise ValueError("occupancy must be a non-empty 3D boolean array")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin must be a finite three-vector")
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and positive")
    if cell_colors.shape != (*occupancy.shape, 3):
        raise ValueError("cell_colors must contain one RGB value per occupancy coordinate")

    quad_keys: list[np.ndarray] = []
    quad_colors: list[np.ndarray] = []
    quad_cells: list[np.ndarray] = []
    exposed_by_direction: dict[str, int] = {}
    for direction, corners in _CUBE_FACES:
        indices = _exposed_cells(occupancy, direction)
        exposed_by_direction[",".join(str(value) for value in direction)] = int(len(indices))
        if not len(indices):
            continue
        integer_corners = np.asarray(corners, dtype=np.int32)
        quad_keys.append(indices[:, None, :].astype(np.int32) * 2 + integer_corners[None, :, :])
        quad_colors.append(cell_colors[tuple(indices.T)])
        quad_cells.append(indices.astype(np.uint16))
    if not quad_keys:
        raise ValueError("occupancy has no exposed faces")

    keys = np.concatenate(quad_keys, axis=0)
    colors = np.concatenate(quad_colors, axis=0)
    cells = np.concatenate(quad_cells, axis=0)
    unique_keys, inverse = np.unique(keys.reshape(-1, 3), axis=0, return_inverse=True)
    quads = inverse.reshape(-1, 4).astype(np.uint32)
    faces = np.empty((len(quads) * 2, 3), dtype=np.uint32)
    faces[0::2] = quads[:, (0, 1, 2)]
    faces[1::2] = quads[:, (0, 2, 3)]
    face_colors = np.repeat(colors, 2, axis=0)
    face_cell_indices = np.repeat(cells, 2, axis=0)
    vertices = origin[None, :] + unique_keys.astype(np.float64) * (voxel_size * 0.5)

    color_sum = np.zeros((len(vertices), 3), dtype=np.float64)
    color_count = np.zeros(len(vertices), dtype=np.int32)
    flat_inverse = inverse.reshape(-1)
    np.add.at(color_sum, flat_inverse, np.repeat(colors, 4, axis=0))
    np.add.at(color_count, flat_inverse, 1)
    vertex_colors = np.rint(color_sum / np.maximum(color_count[:, None], 1)).astype(np.uint8)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    boundary_edges, nonmanifold_edges = _edge_metrics(faces)
    triangle_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    grid_coordinates = (vertices - origin[None, :]) / (voxel_size * 0.5)
    grid_error = float(np.max(np.abs(grid_coordinates - np.rint(grid_coordinates)), initial=0.0))
    component_count = len(mesh.split(only_watertight=False))
    metrics: dict[str, Any] = {
        "operation": "connect exposed voxel faces on their exact shared grid corners",
        "occupiedCellCount": int(np.count_nonzero(occupancy)),
        "exposedQuadCount": int(len(quads)),
        "triangleCount": int(len(faces)),
        "vertexCount": int(len(vertices)),
        "removedInternalCellFaces": int(np.count_nonzero(occupancy) * 6 - len(quads)),
        "exposedQuadsByDirection": exposed_by_direction,
        "componentCount": int(component_count),
        "boundaryEdgeCount": boundary_edges,
        "nonmanifoldEdgeCount": nonmanifold_edges,
        "degenerateTriangleCount": int(np.count_nonzero(triangle_areas <= voxel_size**2 * 1e-10)),
        "watertight": bool(mesh.is_watertight),
        "windingConsistent": bool(mesh.is_winding_consistent),
        "eulerNumber": int(mesh.euler_number),
        "volume": float(mesh.volume),
        "gridCornerMaximumError": grid_error,
        "vertexSmoothingApplied": False,
        "normalSmoothingApplied": False,
        "subdivisionApplied": False,
        "marchingCubesApplied": False,
        "vertexDisplacementApplied": False,
    }
    return ConnectedVoxelSurface(
        vertices=vertices.astype(np.float32),
        faces=faces,
        face_colors=face_colors,
        vertex_colors=vertex_colors,
        face_cell_indices=face_cell_indices,
        metrics=metrics,
    )
