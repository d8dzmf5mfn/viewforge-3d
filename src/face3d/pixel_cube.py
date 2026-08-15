from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from face3d.glb import export_instanced_voxels
from face3d.io import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)

TEMPLATE_INFERRED_SOURCE_BIT = 8

FACE_BITS = {
    "xMin": 1,
    "xMax": 2,
    "yMin": 4,
    "yMax": 8,
    "zMin": 16,
    "zMax": 32,
}


@dataclass(frozen=True, slots=True)
class PixelCubeSpec:
    side_length_m: float = 0.20
    cells_per_edge: int = 128
    cell_fill_ratio: float = 0.92
    inferred_confidence: float = 0.25

    def __post_init__(self) -> None:
        if not math.isfinite(self.side_length_m) or self.side_length_m <= 0:
            raise ValueError("side_length_m must be finite and positive")
        if self.cells_per_edge < 2:
            raise ValueError("cells_per_edge must be at least 2")
        if not math.isfinite(self.cell_fill_ratio) or not 0 < self.cell_fill_ratio <= 1:
            raise ValueError("cell_fill_ratio must be in (0, 1]")
        if (
            not math.isfinite(self.inferred_confidence)
            or not 0 <= self.inferred_confidence <= 1
        ):
            raise ValueError("inferred_confidence must be in [0, 1]")

    @property
    def cell_pitch_m(self) -> float:
        return self.side_length_m / self.cells_per_edge

    @property
    def surface_cell_count(self) -> int:
        inner = max(self.cells_per_edge - 2, 0)
        return self.cells_per_edge**3 - inner**3


@dataclass(frozen=True, slots=True)
class PixelCuboidSpec:
    cells_xyz: tuple[int, int, int] = (86, 128, 107)
    cell_pitch_m: float = 0.001875
    cell_fill_ratio: float = 0.92
    inferred_confidence: float = 0.25

    def __post_init__(self) -> None:
        if len(self.cells_xyz) != 3 or any(value < 2 for value in self.cells_xyz):
            raise ValueError("cells_xyz must contain three values of at least 2")
        if not math.isfinite(self.cell_pitch_m) or self.cell_pitch_m <= 0:
            raise ValueError("cell_pitch_m must be finite and positive")
        if not math.isfinite(self.cell_fill_ratio) or not 0 < self.cell_fill_ratio <= 1:
            raise ValueError("cell_fill_ratio must be in (0, 1]")
        if (
            not math.isfinite(self.inferred_confidence)
            or not 0 <= self.inferred_confidence <= 1
        ):
            raise ValueError("inferred_confidence must be in [0, 1]")

    @property
    def dimensions_m(self) -> tuple[float, float, float]:
        return tuple(value * self.cell_pitch_m for value in self.cells_xyz)

    @property
    def surface_cell_count(self) -> int:
        x, y, z = self.cells_xyz
        return x * y * z - max(x - 2, 0) * max(y - 2, 0) * max(z - 2, 0)


def surface_cell_indices_xyz(cells_xyz: tuple[int, int, int]) -> np.ndarray:
    if len(cells_xyz) != 3 or any(value < 2 for value in cells_xyz):
        raise ValueError("cells_xyz must contain three values of at least 2")
    axes = [np.arange(value, dtype=np.int32) for value in cells_xyz]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    maximum = np.asarray(cells_xyz, dtype=np.int32) - 1
    boundary = np.any((grid == 0) | (grid == maximum), axis=1)
    return grid[boundary]


def surface_cell_indices(cells_per_edge: int) -> np.ndarray:
    if cells_per_edge < 2:
        raise ValueError("cells_per_edge must be at least 2")
    return surface_cell_indices_xyz((cells_per_edge,) * 3)


def cell_face_masks_xyz(indices: np.ndarray, cells_xyz: tuple[int, int, int]) -> np.ndarray:
    maximum = np.asarray(cells_xyz, dtype=np.int32) - 1
    masks = np.zeros(len(indices), dtype=np.uint8)
    masks[indices[:, 0] == 0] |= FACE_BITS["xMin"]
    masks[indices[:, 0] == maximum[0]] |= FACE_BITS["xMax"]
    masks[indices[:, 1] == 0] |= FACE_BITS["yMin"]
    masks[indices[:, 1] == maximum[1]] |= FACE_BITS["yMax"]
    masks[indices[:, 2] == 0] |= FACE_BITS["zMin"]
    masks[indices[:, 2] == maximum[2]] |= FACE_BITS["zMax"]
    return masks


def cell_face_masks(indices: np.ndarray, cells_per_edge: int) -> np.ndarray:
    return cell_face_masks_xyz(indices, (cells_per_edge,) * 3)


def cell_centers_xyz(
    indices: np.ndarray,
    cells_xyz: tuple[int, int, int],
    cell_pitch_m: float,
) -> np.ndarray:
    half = np.asarray(cells_xyz, dtype=np.float64) * cell_pitch_m / 2
    return (indices.astype(np.float64) + 0.5) * cell_pitch_m - half


def cell_centers(indices: np.ndarray, spec: PixelCubeSpec) -> np.ndarray:
    return cell_centers_xyz(indices, (spec.cells_per_edge,) * 3, spec.cell_pitch_m)


def _traceability_bytes(
    indices: np.ndarray,
    positions: np.ndarray,
    face_masks: np.ndarray,
    spec: PixelCubeSpec,
) -> bytes:
    records: list[bytes] = []
    for cell_id, (grid, position, face_mask) in enumerate(
        zip(indices, positions, face_masks, strict=True)
    ):
        records.append(
            canonical_json_bytes(
                {
                    "cellId": cell_id,
                    "confidence": spec.inferred_confidence,
                    "faceMask": int(face_mask),
                    "gridXYZ": [int(value) for value in grid],
                    "positionMeters": [round(float(value), 9) for value in position],
                    "sourceBits": TEMPLATE_INFERRED_SOURCE_BIT,
                }
            )
        )
    return b"\n".join(records) + b"\n"


def create_pixel_cube(destination: Path, spec: PixelCubeSpec | None = None) -> dict[str, Any]:
    spec = spec or PixelCubeSpec()
    destination = Path(destination)
    indices = surface_cell_indices(spec.cells_per_edge)
    if len(indices) != spec.surface_cell_count:
        raise RuntimeError("surface-cell count does not match the closed-form cube count")

    positions = cell_centers(indices, spec)
    face_masks = cell_face_masks(indices, spec.cells_per_edge)
    confidence = np.full(len(indices), spec.inferred_confidence, dtype=np.float32)
    source_bits = np.full(len(indices), TEMPLATE_INFERRED_SOURCE_BIT, dtype=np.uint8)

    model_path = destination / "models" / "voxels.glb"
    records_path = destination / "pixels" / "cells.jsonl"
    schema_path = destination / "pixels" / "schema.json"
    manifest_path = destination / "manifest.json"

    export_instanced_voxels(
        positions,
        spec.cell_pitch_m,
        confidence,
        source_bits,
        model_path,
        fill_ratio=spec.cell_fill_ratio,
    )
    atomic_write_bytes(
        records_path,
        _traceability_bytes(indices, positions, face_masks, spec),
    )
    schema = {
        "schemaVersion": "1.0.0",
        "format": "face3d-procedural-3d-pixel-cube-jsonl",
        "recordCount": len(indices),
        "recordOrder": "x-major, then y, then z; boundary cells only",
        "units": "meter",
        "fields": {
            "cellId": "uint32",
            "gridXYZ": "uint32[3]",
            "positionMeters": "float64[3]",
            "faceMask": "uint8",
            "confidence": "float32",
            "sourceBits": "uint8",
        },
        "faceBits": FACE_BITS,
        "sourceBits": {"templateInferred": TEMPLATE_INFERRED_SOURCE_BIT},
    }
    atomic_write_json(schema_path, schema)

    procedural_spec = {
        "sideLengthMeters": spec.side_length_m,
        "cellsPerEdge": spec.cells_per_edge,
        "cellPitchMeters": spec.cell_pitch_m,
        "cellFillRatio": spec.cell_fill_ratio,
        "surfaceOnly": True,
    }
    manifest = {
        "schemaVersion": "1.0.0",
        "assetType": "3d-pixel-cube",
        "primaryAsset": True,
        "units": "meter",
        "geometry": {
            "centerMeters": [0.0, 0.0, 0.0],
            "nominalDimensionsMeters": [spec.side_length_m] * 3,
            "nominalSideLengthCentimeters": spec.side_length_m * 100,
            "surfaceOnly": True,
        },
        "pixel": {
            "representation": "closed-six-face-surface-shell",
            "surfaceOnly": True,
            "cellsPerEdge": spec.cells_per_edge,
            "cellPitchMeters": spec.cell_pitch_m,
            "cellFillRatio": spec.cell_fill_ratio,
            "instanceCount": len(indices),
            "expectedInstanceCount": spec.surface_cell_count,
            "confidence": spec.inferred_confidence,
            "sourceBits": TEMPLATE_INFERRED_SOURCE_BIT,
            "sourceLabel": "templateInferred",
        },
        "provenance": {
            "kind": "procedural",
            "observedFrom2D": False,
            "proceduralSpecSha256": sha256_json(procedural_spec),
        },
        "files": {
            "model": "models/voxels.glb",
            "modelSha256": sha256_file(model_path),
            "traceability": "pixels/cells.jsonl",
            "traceabilitySha256": sha256_file(records_path),
            "schema": "pixels/schema.json",
            "schemaSha256": sha256_file(schema_path),
        },
    }
    atomic_write_json(manifest_path, manifest)
    return {
        "ok": True,
        "output": str(destination),
        "model": str(model_path),
        "manifest": str(manifest_path),
        "sideLengthCentimeters": spec.side_length_m * 100,
        "cellPitchCentimeters": spec.cell_pitch_m * 100,
        "instanceCount": len(indices),
    }
