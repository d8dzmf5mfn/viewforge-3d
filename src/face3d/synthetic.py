from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

from face3d.assets import require_assets
from face3d.config import Face3DConfig
from face3d.glb import export_neutral_mesh
from face3d.io import atomic_write_json, sha256_file
from face3d.models import REQUIRED_VIEWS, CameraRecord, ViewRole
from face3d.profiles.face_v1 import FaceProfileV1
from face3d.render import render_flat_mesh
from face3d.stages.flame import FlameModel


def _camera_for_mesh(
    role: ViewRole,
    yaw_radians: float,
    mesh: trimesh.Trimesh,
) -> CameraRecord:
    width = height = 1024
    focal = 1320.0
    rotation = trimesh.transformations.rotation_matrix(np.pi, (1, 0, 0))[:3, :3]
    rotation = rotation @ trimesh.transformations.rotation_matrix(yaw_radians, (0, 1, 0))[:3, :3]
    rotation_vector, _ = cv2.Rodrigues(rotation)
    vertical_span = float(np.ptp(mesh.bounds[:, 1]))
    distance = focal * vertical_span / (height * 0.72) + float(np.max(np.abs(mesh.bounds[:, 2])))
    return CameraRecord(
        role=role,
        width=width,
        height=height,
        focal_length_px=focal,
        principal_point_px=(width / 2, height / 2),
        rotation_vector=tuple(float(value) for value in rotation_vector.reshape(3)),
        translation=(0.0, 0.0, distance),
        yaw_deg=float(np.degrees(yaw_radians)),
        pitch_deg=0.0,
        roll_deg=0.0,
    )


def generate_synthetic_dataset(
    output: Path,
    config: Face3DConfig,
    *,
    count: int = 3,
) -> dict[str, Any]:
    require_assets(config, names=("faceLandmarker", "flameModel", "flameLandmarks"))
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    flame = FlameModel.load(
        config.resolve_asset(config.assets.flame_model),
        config.resolve_asset(config.assets.flame_landmarks),
        config.fit.shape_coefficients,
    )
    profile = FaceProfileV1()
    identities: list[dict[str, Any]] = []
    yaw_by_role = {
        ViewRole.FRONT: 0.0,
        ViewRole.LEFT45: np.radians(-45),
        ViewRole.RIGHT45: np.radians(45),
    }
    for identity_index in range(count):
        identity_id = f"flame-synthetic-{identity_index + 1:03d}"
        identity_dir = output / identity_id
        identity_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(config.seed + identity_index)
        coefficients = rng.normal(0.0, 0.65, size=config.fit.shape_coefficients)
        vertices = flame.shaped_vertices(coefficients)
        mesh = trimesh.Trimesh(vertices=vertices, faces=flame.faces, process=False)
        export_neutral_mesh(mesh, identity_dir / "ground-truth.glb")
        cameras: list[CameraRecord] = []
        hashes: dict[str, str] = {}
        for role in REQUIRED_VIEWS:
            camera = _camera_for_mesh(role, yaw_by_role[role], mesh)
            cameras.append(camera)
            image_path = identity_dir / f"{role.value}.png"
            render_flat_mesh(mesh, camera, image_path, width=1024, height=1024)
            model_landmarks = flame.landmark_vertices(vertices, profile.expected_yaw(role))
            rotation, _ = cv2.Rodrigues(np.asarray(camera.rotation_vector))
            camera_points = model_landmarks @ rotation.T + np.asarray(camera.translation)
            pixels = camera_points[:, :2] / camera_points[:, 2:3] * camera.focal_length_px
            pixels += np.asarray(camera.principal_point_px)
            np.savez_compressed(
                identity_dir / f"{role.value}-ground-truth.npz",
                ibug68=pixels.astype(np.float32),
                vertices=vertices.astype(np.float32),
            )
            hashes[role.value] = sha256_file(image_path)
        ground_truth = {
            "schemaVersion": 1,
            "identity": identity_id,
            "source": "FLAME-2023-Open",
            "shapeCoefficients": coefficients.tolist(),
            "cameras": [camera.model_dump(mode="json") for camera in cameras],
            "inputSha256": hashes,
            "expected": {
                "meanBidirectionalChamferFaceWidthMax": 0.015,
                "p95ErrorFaceWidthMax": 0.03,
            },
        }
        atomic_write_json(identity_dir / "ground-truth.json", ground_truth)
        identities.append(ground_truth)
    dataset = {
        "schemaVersion": 1,
        "count": count,
        "configSha256": sha256_file(config.source_path),
        "identities": identities,
    }
    atomic_write_json(output / "dataset.json", dataset)
    return {"ok": True, "output": str(output), "count": count}
