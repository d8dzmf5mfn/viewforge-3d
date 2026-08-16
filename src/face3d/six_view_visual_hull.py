from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

from face3d.io import atomic_write_bytes, atomic_write_json, sha256_file

SIX_VIEW_ROLES = ("front", "back", "left", "right", "top", "bottom")


def _largest_filled_component(mask: np.ndarray) -> np.ndarray:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        raise ValueError("silhouette contains no separable foreground")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    primary = np.where(labels == label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(primary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(primary)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _read_silhouette(path: Path, resolution: int) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim not in {2, 3}:
        raise ValueError(f"unable to read silhouette image: {path.name}")

    alpha: np.ndarray | None = None
    if raw.ndim == 3 and raw.shape[2] == 4:
        alpha = raw[:, :, 3]
        raw = raw[:, :, :3]

    if alpha is not None and int(alpha.max()) > int(alpha.min()):
        threshold = (int(alpha.max()) + int(alpha.min())) / 2
        candidate = np.where(alpha > threshold, 255, 0).astype(np.uint8)
    else:
        color = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR) if raw.ndim == 2 else raw[:, :, :3]
        height, width = color.shape[:2]
        corner_size = max(2, min(height, width, 32) // 4)
        corners = np.concatenate(
            (
                color[:corner_size, :corner_size].reshape(-1, 3),
                color[:corner_size, -corner_size:].reshape(-1, 3),
                color[-corner_size:, :corner_size].reshape(-1, 3),
                color[-corner_size:, -corner_size:].reshape(-1, 3),
            ),
            axis=0,
        )
        background = np.median(corners.astype(np.float32), axis=0)
        distance = np.linalg.norm(color.astype(np.float32) - background, axis=2)
        scaled = np.clip(distance, 0, 255).astype(np.uint8)
        _, candidate = cv2.threshold(
            scaled,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

    primary = _largest_filled_component(candidate)
    rows, columns = np.nonzero(primary)
    if not len(rows):
        raise ValueError(f"silhouette contains no foreground: {path.name}")
    padding = max(2, round(max(primary.shape) * 0.02))
    y0 = max(int(rows.min()) - padding, 0)
    y1 = min(int(rows.max()) + padding + 1, primary.shape[0])
    x0 = max(int(columns.min()) - padding, 0)
    x1 = min(int(columns.max()) + padding + 1, primary.shape[1])
    cropped = primary[y0:y1, x0:x1]
    normalized = cv2.resize(
        cropped,
        (resolution, resolution),
        interpolation=cv2.INTER_NEAREST,
    )
    coverage = float(np.mean(normalized > 0))
    if not 0.01 <= coverage <= 0.98:
        raise ValueError(f"silhouette foreground coverage is invalid: {path.name}")
    return normalized > 0


def _visual_hull(masks: Mapping[str, np.ndarray]) -> np.ndarray:
    resolution = next(iter(masks.values())).shape[0]
    index = np.arange(resolution)
    reverse = index[::-1]

    front = masks["front"][np.ix_(reverse, index)].T[:, None, :]
    back = masks["back"][np.ix_(reverse, reverse)].T[:, None, :]
    left = masks["left"][np.ix_(reverse, reverse)].T[None, :, :]
    right = masks["right"][np.ix_(reverse, index)].T[None, :, :]
    top = masks["top"][np.ix_(reverse, index)].T[:, :, None]
    bottom = masks["bottom"][np.ix_(index, index)].T[:, :, None]
    occupancy = front & back & left & right & top & bottom
    if not np.any(occupancy):
        raise ValueError("six-view silhouette intersection is empty")

    labels, component_count = ndimage.label(occupancy)
    if component_count > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        occupancy = labels == int(np.argmax(sizes))
    occupancy = ndimage.binary_fill_holes(occupancy)
    return np.asarray(occupancy, dtype=bool)


def _mask_iou(expected: np.ndarray, measured: np.ndarray) -> float:
    union = np.count_nonzero(expected | measured)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(expected & measured) / union)


def _projection_metrics(occupancy: np.ndarray, masks: Mapping[str, np.ndarray]) -> dict[str, float]:
    front = np.flip(np.any(occupancy, axis=1).T, axis=0)
    back = np.flip(front, axis=1)
    left = np.flip(np.any(occupancy, axis=0).T, axis=(0, 1))
    right = np.flip(left, axis=1)
    top = np.flip(np.any(occupancy, axis=2).T, axis=0)
    bottom = np.flip(top, axis=0)
    projected = {
        "front": front,
        "back": back,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
    }
    return {role: _mask_iou(masks[role], projected[role]) for role in SIX_VIEW_ROLES}


def reconstruct_six_view_visual_hull(
    views: Mapping[str, Path],
    destination: Path,
    *,
    resolution: int = 96,
    width_m: float = 1.0,
    depth_m: float = 1.0,
    height_m: float = 1.0,
) -> dict[str, Any]:
    if set(views) != set(SIX_VIEW_ROLES):
        missing = sorted(set(SIX_VIEW_ROLES) - set(views))
        extra = sorted(set(views) - set(SIX_VIEW_ROLES))
        raise ValueError(f"invalid six-view roles; missing={missing}, extra={extra}")
    if not 32 <= resolution <= 256:
        raise ValueError("resolution must be between 32 and 256")
    dimensions = np.asarray([width_m, depth_m, height_m], dtype=np.float64)
    if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0) or np.any(dimensions > 1000):
        raise ValueError("nominal dimensions must be finite and in (0, 1000] meters")

    resolved = {role: Path(path).expanduser().resolve() for role, path in views.items()}
    if any(not path.is_file() for path in resolved.values()):
        raise ValueError("all six silhouette images must exist")
    masks = {role: _read_silhouette(resolved[role], resolution) for role in SIX_VIEW_ROLES}
    occupancy = _visual_hull(masks)
    spacing = dimensions / max(resolution - 1, 1)
    padded = np.pad(occupancy.astype(np.float32), 1)
    vertices, faces, _, _ = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=tuple(float(value) for value in spacing),
        allow_degenerate=False,
    )
    vertices -= spacing + dimensions / 2
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.process(validate=True)
    trimesh.repair.fix_normals(mesh)
    if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        raise RuntimeError("visual hull did not produce a usable triangle mesh")

    destination = Path(destination).expanduser().resolve()
    model_path = destination / "models" / "visual-hull.glb"
    qa_path = destination / "qa" / "visual-hull.json"
    manifest_path = destination / "manifest.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    exported = mesh.export(file_type="glb")
    if not isinstance(exported, bytes):
        raise RuntimeError("visual hull GLB export did not return bytes")
    atomic_write_bytes(model_path, exported)

    silhouette_iou = _projection_metrics(occupancy, masks)
    qa = {
        "schemaVersion": 1,
        "route": "six-view-orthographic-visual-hull",
        "previewOnly": True,
        "resolution": resolution,
        "occupiedVoxelCount": int(np.count_nonzero(occupancy)),
        "vertexCount": int(len(mesh.vertices)),
        "triangleCount": int(len(mesh.faces)),
        "componentCount": int(len(mesh.split(only_watertight=False))),
        "watertight": bool(mesh.is_watertight),
        "windingConsistent": bool(mesh.is_winding_consistent),
        "finite": bool(np.all(np.isfinite(mesh.vertices))),
        "volumeCubicMeters": float(abs(mesh.volume)),
        "boundsMeters": np.asarray(mesh.bounds, dtype=float).tolist(),
        "silhouetteIoU": silhouette_iou,
    }
    atomic_write_json(qa_path, qa)
    manifest = {
        "schemaVersion": 1,
        "assetType": "six-view-visual-hull-preview",
        "route": "visual-hull-preview",
        "state": "preview",
        "previewOnly": True,
        "identityAcceptanceAllowed": False,
        "units": "meter",
        "nominalDimensionsMeters": dimensions.tolist(),
        "cameraContract": {
            "projection": "orthographic",
            "worldAxes": {"right": "+X", "back": "+Y", "up": "+Z"},
            "viewImageUp": {
                "front": "+Z",
                "back": "+Z",
                "left": "+Z",
                "right": "+Z",
                "top": "+Y",
                "bottom": "-Y",
            },
        },
        "inputs": {
            role: {"sha256": sha256_file(resolved[role]), "evidence": "user-provided"}
            for role in SIX_VIEW_ROLES
        },
        "limitations": [
            "Concavities not visible in silhouettes cannot be reconstructed.",
            "Independent crop normalization assumes each silhouette spans the declared dimension.",
            "Automated geometry checks do not replace user visual signoff.",
        ],
        "files": {
            "model": "models/visual-hull.glb",
            "modelSha256": sha256_file(model_path),
            "qa": "qa/visual-hull.json",
            "qaSha256": sha256_file(qa_path),
        },
        "userSignoff": False,
    }
    atomic_write_json(manifest_path, manifest)
    return {
        "ok": True,
        "route": manifest["route"],
        "state": manifest["state"],
        "previewOnly": True,
        "model": str(model_path),
        "manifest": str(manifest_path),
        "qa": str(qa_path),
        "vertexCount": qa["vertexCount"],
        "triangleCount": qa["triangleCount"],
        "watertight": qa["watertight"],
    }
