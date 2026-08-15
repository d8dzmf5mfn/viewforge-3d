from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from face3d.multiview_pixels import _surface_mask, _surface_normals


@dataclass(frozen=True, slots=True)
class CoarsenedPixelGrid:
    occupancy: np.ndarray
    surface_mask: np.ndarray
    surface_indices: np.ndarray
    solid_indices: np.ndarray
    origin: np.ndarray
    unit_size: float
    positions: np.ndarray
    normals: np.ndarray
    representative_source_indices: np.ndarray
    metrics: dict[str, Any]


def _pooled_occupancy(occupancy: np.ndarray, factor: int) -> np.ndarray:
    padding = tuple((0, (-length) % factor) for length in occupancy.shape)
    padded = np.pad(occupancy, padding, mode="constant", constant_values=False)
    shape = tuple(length // factor for length in padded.shape)
    return padded.reshape(
        shape[0],
        factor,
        shape[1],
        factor,
        shape[2],
        factor,
    ).any(axis=(1, 3, 5))


def _darkest_representatives(
    source_surface_indices: np.ndarray,
    source_rgb: np.ndarray,
    target_surface_indices: np.ndarray,
    target_shape: tuple[int, int, int],
    factor: int,
) -> np.ndarray:
    mapped = source_surface_indices // factor
    source_keys = np.ravel_multi_index(mapped.T, target_shape)
    luminance = source_rgb.astype(np.float64) @ np.asarray(
        [0.2126, 0.7152, 0.0722],
        dtype=np.float64,
    )
    source_order = np.arange(len(source_surface_indices), dtype=np.int64)
    order = np.lexsort((source_order, luminance, source_keys))
    sorted_keys = source_keys[order]
    unique_keys, first = np.unique(sorted_keys, return_index=True)
    representatives = order[first]

    target_keys = np.ravel_multi_index(target_surface_indices.T, target_shape)
    locations = np.searchsorted(unique_keys, target_keys)
    valid = locations < len(unique_keys)
    valid[valid] &= unique_keys[locations[valid]] == target_keys[valid]
    if not np.all(valid):
        raise ValueError(
            f"{int(np.count_nonzero(~valid))} coarse surface units lack a source representative"
        )
    return representatives[locations].astype(np.uint32)


def coarsen_pixel_grid(
    occupancy: np.ndarray,
    origin: np.ndarray,
    unit_size: float,
    source_surface_indices: np.ndarray,
    source_rgb: np.ndarray,
    *,
    factor: int = 2,
) -> CoarsenedPixelGrid:
    """Merge integer grid blocks while retaining discrete cubes and dark feature samples."""
    occupancy = np.asarray(occupancy, dtype=bool)
    origin = np.asarray(origin, dtype=np.float64)
    source_surface_indices = np.asarray(source_surface_indices, dtype=np.int64)
    source_rgb = np.asarray(source_rgb, dtype=np.uint8)
    if occupancy.ndim != 3 or not np.any(occupancy):
        raise ValueError("occupancy must be a non-empty 3D boolean array")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin must be a finite three-vector")
    if not np.isfinite(unit_size) or unit_size <= 0:
        raise ValueError("unit_size must be finite and positive")
    if not isinstance(factor, int) or factor < 2:
        raise ValueError("factor must be an integer of at least two")
    if source_surface_indices.ndim != 2 or source_surface_indices.shape[1] != 3:
        raise ValueError("source_surface_indices must be [N,3]")
    if source_rgb.shape != (len(source_surface_indices), 3):
        raise ValueError("source_rgb must contain one RGB value per source surface unit")
    if np.any(source_surface_indices < 0) or np.any(
        source_surface_indices >= np.asarray(occupancy.shape)[None, :]
    ):
        raise ValueError("source surface indices exceed occupancy bounds")
    if not np.all(occupancy[tuple(source_surface_indices.T)]):
        raise ValueError("source surface indices must be occupied")

    coarse_occupancy = _pooled_occupancy(occupancy, factor)
    surface_mask = _surface_mask(coarse_occupancy)
    surface_indices = np.argwhere(surface_mask).astype(np.uint16)
    solid_indices = np.argwhere(coarse_occupancy).astype(np.uint16)
    if not len(surface_indices):
        raise ValueError("coarsened occupancy has no surface units")
    coarse_origin = origin + (factor - 1) * unit_size * 0.5
    coarse_unit_size = float(unit_size * factor)
    positions = (
        coarse_origin[None, :] + surface_indices.astype(np.float64) * coarse_unit_size
    ).astype(np.float32)
    normals = _surface_normals(coarse_occupancy, surface_indices.astype(np.int64))
    representative_source_indices = _darkest_representatives(
        source_surface_indices,
        source_rgb,
        surface_indices.astype(np.int64),
        coarse_occupancy.shape,
        factor,
    )
    source_count = len(source_surface_indices)
    count_ratio = len(surface_indices) / source_count
    metrics: dict[str, Any] = {
        "operation": "integer block max-pool occupancy and retain discrete surface cubes",
        "factor": factor,
        "sourceGridShape": list(occupancy.shape),
        "coarseGridShape": list(coarse_occupancy.shape),
        "sourceUnitSize": float(unit_size),
        "coarseUnitSize": coarse_unit_size,
        "unitEdgeLengthFactor": float(factor),
        "unitVolumeFactor": float(factor**3),
        "sourceOccupiedUnitCount": int(np.count_nonzero(occupancy)),
        "coarseOccupiedUnitCount": int(np.count_nonzero(coarse_occupancy)),
        "sourceSurfaceUnitCount": int(source_count),
        "coarseSurfaceUnitCount": int(len(surface_indices)),
        "surfaceUnitCountRatio": float(count_ratio),
        "surfaceUnitReductionFraction": float(1.0 - count_ratio),
        "representativePolicy": "minimum source luminance within each coarse block",
        "missingRepresentativeCount": 0,
        "integerGridAligned": True,
        "unitsRemainDiscreteCubes": True,
        "surfaceConnectionPerformed": False,
        "geometricSmoothingApplied": False,
        "subdivisionApplied": False,
        "marchingCubesApplied": False,
    }
    return CoarsenedPixelGrid(
        occupancy=coarse_occupancy,
        surface_mask=surface_mask,
        surface_indices=surface_indices,
        solid_indices=solid_indices,
        origin=coarse_origin.astype(np.float32),
        unit_size=coarse_unit_size,
        positions=positions,
        normals=normals.astype(np.float32),
        representative_source_indices=representative_source_indices,
        metrics=metrics,
    )
