from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image

from face3d.config import Face3DConfig
from face3d.errors import fail
from face3d.io import atomic_write_bytes, atomic_write_json
from face3d.models import CameraRecord
from face3d.profiles import FaceProfileV2
from face3d.stages.flame import FlameModel


def geometry_hash(vertices: np.ndarray, faces: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(vertices, dtype="<f4").tobytes())
    digest.update(np.ascontiguousarray(faces, dtype="<u4").tobytes())
    return digest.hexdigest()


@dataclass(slots=True)
class EyeballAsset:
    center: np.ndarray
    radius: float
    gaze: np.ndarray

    def mesh(self) -> trimesh.Trimesh:
        sphere = trimesh.creation.uv_sphere(radius=self.radius, count=[64, 32])
        sphere.apply_translation(self.center)
        local = np.asarray(sphere.vertices, dtype=np.float64) - self.center
        unit = local / np.maximum(np.linalg.norm(local, axis=1, keepdims=True), 1e-12)
        uv = np.column_stack(
            (
                0.5 + np.arctan2(unit[:, 0], unit[:, 2]) / (2.0 * np.pi),
                0.5 - np.arcsin(np.clip(unit[:, 1], -1.0, 1.0)) / np.pi,
            )
        ).astype(np.float32)
        sphere.visual = trimesh.visual.TextureVisuals(uv=uv, material=_eye_material())
        return sphere


@dataclass(slots=True)
class UnifiedHeadAsset:
    skin_vertices: np.ndarray
    skin_faces: np.ndarray
    render_to_skin: np.ndarray
    render_faces: np.ndarray
    uv: np.ndarray
    regions: dict[str, np.ndarray]
    left_eye: EyeballAsset
    right_eye: EyeballAsset
    geometry_sha256: str
    anatomy: dict[str, Any]

    @property
    def render_vertices(self) -> np.ndarray:
        return self.skin_vertices[self.render_to_skin]

    @property
    def skin_mesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(
            vertices=self.skin_vertices,
            faces=self.skin_faces,
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
        destination.parent.mkdir(parents=True, exist_ok=True)
        output = io.BytesIO()
        np.savez_compressed(
            output,
            skin_vertices=self.skin_vertices.astype(np.float32),
            skin_faces=self.skin_faces.astype(np.int32),
            render_to_skin=self.render_to_skin.astype(np.int32),
            render_faces=self.render_faces.astype(np.int32),
            uv=self.uv.astype(np.float32),
            left_eye_center=self.left_eye.center.astype(np.float32),
            left_eye_radius=np.asarray(self.left_eye.radius, dtype=np.float32),
            left_eye_gaze=self.left_eye.gaze.astype(np.float32),
            right_eye_center=self.right_eye.center.astype(np.float32),
            right_eye_radius=np.asarray(self.right_eye.radius, dtype=np.float32),
            right_eye_gaze=self.right_eye.gaze.astype(np.float32),
            geometry_sha256=np.asarray(self.geometry_sha256),
            anatomy_json=np.asarray(json.dumps(self.anatomy, sort_keys=True)),
            **{
                f"region_{name}": np.asarray(indices, dtype=np.int32)
                for name, indices in self.regions.items()
            },
        )
        atomic_write_bytes(destination, output.getvalue())

    @classmethod
    def load(cls, source: Path) -> UnifiedHeadAsset:
        with np.load(source, allow_pickle=False) as payload:
            regions = {
                key.removeprefix("region_"): np.asarray(payload[key], dtype=np.int64)
                for key in payload.files
                if key.startswith("region_")
            }
            return cls(
                skin_vertices=np.asarray(payload["skin_vertices"], dtype=np.float64),
                skin_faces=np.asarray(payload["skin_faces"], dtype=np.int64),
                render_to_skin=np.asarray(payload["render_to_skin"], dtype=np.int64),
                render_faces=np.asarray(payload["render_faces"], dtype=np.int64),
                uv=np.asarray(payload["uv"], dtype=np.float32),
                regions=regions,
                left_eye=EyeballAsset(
                    center=np.asarray(payload["left_eye_center"], dtype=np.float64),
                    radius=float(payload["left_eye_radius"]),
                    gaze=np.asarray(payload["left_eye_gaze"], dtype=np.float64),
                ),
                right_eye=EyeballAsset(
                    center=np.asarray(payload["right_eye_center"], dtype=np.float64),
                    radius=float(payload["right_eye_radius"]),
                    gaze=np.asarray(payload["right_eye_gaze"], dtype=np.float64),
                ),
                geometry_sha256=str(payload["geometry_sha256"]),
                anatomy=json.loads(str(payload["anatomy_json"])),
            )

    def export_head_glb(self, destination: Path, atlas: Image.Image) -> None:
        head = self.render_mesh
        material = trimesh.visual.material.PBRMaterial(
            name="FaceV2.ProjectedSkin",
            baseColorFactor=(1.0, 1.0, 1.0, 1.0),
            baseColorTexture=atlas,
            metallicFactor=0.0,
            roughnessFactor=0.62,
            doubleSided=False,
        )
        head.visual = trimesh.visual.TextureVisuals(uv=self.uv, material=material)
        head.metadata.update(
            {
                "name": "HeadSkin",
                "geometryHash": self.geometry_sha256,
                "topology": self.anatomy.get("route", {}).get(
                    "topology",
                    "single-continuous-flame-head",
                ),
            }
        )
        scene = trimesh.Scene()
        scene.add_geometry(head, node_name="HeadSkin", geom_name="HeadSkin")
        scene.add_geometry(self.left_eye.mesh(), node_name="Eyeball.L", geom_name="Eyeball.L")
        scene.add_geometry(self.right_eye.mesh(), node_name="Eyeball.R", geom_name="Eyeball.R")
        scene.metadata["geometryHash"] = self.geometry_sha256
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(
            destination,
            trimesh.exchange.gltf.export_glb(scene, include_normals=True),
        )


def _eye_material() -> trimesh.visual.material.PBRMaterial:
    width, height = 1024, 512
    yy, xx = np.mgrid[:height, :width]
    u = xx / (width - 1)
    v = yy / (height - 1)
    angle = np.sqrt(((u - 0.5) / 0.075) ** 2 + ((v - 0.5) / 0.15) ** 2)
    pupil = np.sqrt(((u - 0.5) / 0.028) ** 2 + ((v - 0.5) / 0.056) ** 2)
    texture = np.full((height, width, 3), (229, 226, 216), dtype=np.uint8)
    iris_mask = angle <= 1.0
    texture[iris_mask] = np.asarray((74, 104, 111), dtype=np.uint8)
    texture[pupil <= 1.0] = np.asarray((17, 20, 22), dtype=np.uint8)
    highlight = ((u - 0.475) / 0.010) ** 2 + ((v - 0.465) / 0.020) ** 2 <= 1.0
    texture[highlight & iris_mask] = 245
    return trimesh.visual.material.PBRMaterial(
        name="FaceV2.Eyeball",
        baseColorTexture=Image.fromarray(texture),
        metallicFactor=0.0,
        roughnessFactor=0.28,
    )


def _project(points: np.ndarray, camera: CameraRecord) -> tuple[np.ndarray, np.ndarray]:
    rotation, _ = cv2.Rodrigues(np.asarray(camera.rotation_vector, dtype=np.float64))
    camera_points = np.asarray(points, dtype=np.float64) @ rotation.T + np.asarray(
        camera.translation, dtype=np.float64
    )
    depth = camera_points[:, 2]
    pixels = np.zeros((len(points), 2), dtype=np.float64)
    valid = depth > 1e-6
    pixels[valid] = camera_points[valid, :2] / depth[valid, None]
    pixels[valid] *= camera.focal_length_px
    pixels[valid] += np.asarray(camera.principal_point_px)
    return pixels, depth


def _backproject(pixel: np.ndarray, depth: float, camera: CameraRecord) -> np.ndarray:
    camera_point = np.asarray(
        [
            (pixel[0] - camera.principal_point_px[0]) * depth / camera.focal_length_px,
            (pixel[1] - camera.principal_point_px[1]) * depth / camera.focal_length_px,
            depth,
        ],
        dtype=np.float64,
    )
    rotation, _ = cv2.Rodrigues(np.asarray(camera.rotation_vector, dtype=np.float64))
    return (camera_point - np.asarray(camera.translation, dtype=np.float64)) @ rotation


def _fit_eyeballs(
    run_dir: Path,
    initial_centers: np.ndarray,
    initial_radii: np.ndarray,
    cameras: list[CameraRecord],
    config: Face3DConfig,
) -> tuple[EyeballAsset, EyeballAsset, dict[str, Any]]:
    profile = FaceProfileV2()
    candidates: list[list[np.ndarray]] = [[], []]
    radius_candidates: list[list[float]] = [[], []]
    reprojection: list[float] = []
    for camera in cameras:
        with np.load(run_dir / "working" / "landmarks" / f"{camera.role.value}.npz") as data:
            landmarks = np.asarray(data["all"], dtype=np.float64)
        groups = [
            np.asarray(landmarks[list(indices), :2], dtype=np.float64)
            * np.asarray([camera.width, camera.height])
            for indices in profile.iris_landmark_groups.values()
        ]
        targets = np.stack([group.mean(axis=0) for group in groups])
        target_radii = np.asarray(
            [np.median(np.linalg.norm(group - group.mean(axis=0), axis=1)) for group in groups]
        )
        projected, depths = _project(initial_centers, camera)
        direct = np.linalg.norm(projected[0] - targets[0]) + np.linalg.norm(
            projected[1] - targets[1]
        )
        swapped = np.linalg.norm(projected[0] - targets[1]) + np.linalg.norm(
            projected[1] - targets[0]
        )
        order = (0, 1) if direct <= swapped else (1, 0)
        for eye_index, target_index in enumerate(order):
            candidates[eye_index].append(
                _backproject(targets[target_index], float(depths[eye_index]), camera)
            )
            radius_candidates[eye_index].append(
                float(target_radii[target_index] * depths[eye_index] / camera.focal_length_px)
            )

    centers = initial_centers.copy()
    radii = initial_radii.copy()
    for index in range(2):
        observed_center = np.median(np.stack(candidates[index]), axis=0)
        displacement = observed_center - initial_centers[index]
        maximum_shift = initial_radii[index] * 0.18
        displacement_length = float(np.linalg.norm(displacement))
        if displacement_length > maximum_shift:
            displacement *= maximum_shift / displacement_length
        centers[index] += displacement
        observed_radius = float(np.median(radius_candidates[index]))
        radii[index] = float(
            np.clip(
                observed_radius,
                initial_radii[index] * config.anatomy.eye_radius_min_scale,
                initial_radii[index] * config.anatomy.eye_radius_max_scale,
            )
        )
    shared_radius = float(np.mean(radii))
    maximum_difference = shared_radius * config.anatomy.eye_radius_symmetry_max
    radii = np.clip(
        radii,
        shared_radius - maximum_difference / 2,
        shared_radius + maximum_difference / 2,
    )

    for camera in cameras:
        with np.load(run_dir / "working" / "landmarks" / f"{camera.role.value}.npz") as data:
            landmarks = np.asarray(data["all"], dtype=np.float64)
        targets = np.stack(
            [
                np.asarray(landmarks[list(indices), :2]).mean(axis=0)
                * np.asarray([camera.width, camera.height])
                for indices in profile.iris_landmark_groups.values()
            ]
        )
        projected, _ = _project(centers, camera)
        direct = np.linalg.norm(projected[0] - targets[0]) + np.linalg.norm(
            projected[1] - targets[1]
        )
        swapped = np.linalg.norm(projected[0] - targets[1]) + np.linalg.norm(
            projected[1] - targets[0]
        )
        reprojection.append(float(min(direct, swapped) / 2))

    gaze = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    eyes = (
        EyeballAsset(centers[0], float(radii[0]), gaze.copy()),
        EyeballAsset(centers[1], float(radii[1]), gaze.copy()),
    )
    metrics = {
        "completeEyeballNodes": 2,
        "left": {"center": centers[0].tolist(), "radius": float(radii[0])},
        "right": {"center": centers[1].tolist(), "radius": float(radii[1])},
        "radiusDifferenceRatio": float(abs(radii[0] - radii[1]) / max(shared_radius, 1e-12)),
        "irisReprojectionErrorPx": float(np.mean(reprojection)),
    }
    return eyes[0], eyes[1], metrics


def _constrain_eyelid(
    vertices: np.ndarray,
    indices: np.ndarray,
    eye: EyeballAsset,
    clearance_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = vertices[indices]
    distances = np.linalg.norm(points - eye.center, axis=1)
    count = max(12, int(np.ceil(len(indices) * 0.30)))
    nearest = np.argsort(distances)[:count]
    contact_indices = indices[nearest]
    vectors = vertices[contact_indices] - eye.center
    lengths = np.linalg.norm(vectors, axis=1)
    directions = vectors / np.maximum(lengths[:, None], 1e-12)
    gap = np.clip(lengths - eye.radius, 0.0, clearance_max * eye.radius)
    vertices[contact_indices] = eye.center + directions * (eye.radius + gap)[:, None]
    final_gap = np.linalg.norm(vertices[contact_indices] - eye.center, axis=1) - eye.radius
    return contact_indices, final_gap


def _mesh_anatomy(mesh: trimesh.Trimesh) -> dict[str, Any]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    sorted_edges = np.sort(
        np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
    )
    _, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    boundary_edges = int(np.count_nonzero(counts == 1))
    non_manifold = int(np.count_nonzero(counts > 2))
    area = np.asarray(mesh.area_faces)
    vertices = np.asarray(mesh.vertices)
    # A vertex-count quantile moves downward when local eye topology is refined.
    # Use a fixed geometric cap and normalize by local edge scale so UV seams or
    # adaptive triangles cannot be mistaken for a pointed cranium.
    top_threshold = float(vertices[:, 1].max() - np.ptp(vertices[:, 1]) * 0.18)
    top = np.flatnonzero(vertices[:, 1] >= top_threshold)
    normalized_curvature: list[float] = []
    for index in top:
        neighbors = np.asarray(mesh.vertex_neighbors[index], dtype=np.int64)
        if len(neighbors):
            edge_scale = float(
                np.median(np.linalg.norm(vertices[neighbors] - vertices[index], axis=1))
            )
            if edge_scale > 1e-12:
                normalized_curvature.append(
                    float(
                        np.linalg.norm(vertices[index] - vertices[neighbors].mean(axis=0))
                        / edge_scale
                    )
                )
    positive = np.asarray(normalized_curvature, dtype=np.float64)
    positive = positive[positive > 0]
    spike_ratio = (
        float(np.quantile(positive, 0.99) / max(float(np.median(positive)), 1e-12))
        if len(positive)
        else 0.0
    )
    return {
        "connectedComponents": int(len(mesh.split(only_watertight=False))),
        "boundaryEdges": boundary_edges,
        "nonManifoldEdges": non_manifold,
        "degenerateTriangles": int(np.count_nonzero(area <= 1e-14)),
        "finite": bool(np.isfinite(vertices).all()),
        "topCurvatureSpikeRatio": spike_ratio,
    }


def build_unified_head(run_dir: Path, config: Face3DConfig) -> UnifiedHeadAsset:
    if not config.is_v2:
        fail("config-invalid", "UnifiedHeadAsset 仅属于 Face v2", stage="head-v2")
    fit_path = run_dir / "working" / "fit.npz"
    prepared_path = config.resolve_asset(config.assets.flame_prepared)  # type: ignore[arg-type]
    if not fit_path.is_file() or not prepared_path.is_file():
        fail(
            "asset-missing",
            "Face v2 缺少拟合结果或已准备的统一拓扑",
            stage="head-v2",
            details={"fit": str(fit_path), "prepared": str(prepared_path)},
        )
    with np.load(fit_path, allow_pickle=False) as fitted:
        base_vertices = np.asarray(fitted["vertices"], dtype=np.float64)
        base_faces = np.asarray(fitted["faces"], dtype=np.int64)
    with np.load(prepared_path, allow_pickle=False) as prepared:
        expected_faces = np.asarray(prepared["faces"], dtype=np.int64)
        render_to_skin = np.asarray(prepared["render_to_subdiv"], dtype=np.int64)
        render_faces = np.asarray(prepared["render_faces"], dtype=np.int64)
        uv = np.asarray(prepared["uv"], dtype=np.float32)
        template_vertices = np.asarray(prepared["template_vertices"], dtype=np.float64)
        prepared_eye_radii = np.asarray(prepared["eye_radii"], dtype=np.float64)
        regions = {
            key.removeprefix("region_"): np.asarray(prepared[key], dtype=np.int64)
            for key in prepared.files
            if key.startswith("region_")
        }

    vertices = base_vertices
    faces = base_faces
    for _ in range(config.anatomy.subdivision_levels):
        vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    if not np.array_equal(faces, expected_faces) or len(vertices) != len(template_vertices):
        fail(
            "topology-mismatch",
            "拟合头模与已锁定 Face v2 拓扑不一致",
            stage="head-v2",
        )

    flame = FlameModel.load(
        config.resolve_asset(config.assets.flame_model),
        config.resolve_asset(config.assets.flame_landmarks),
        config.fit.shape_coefficients,
    )
    shaped_eye_centers = np.stack(flame.eye_centers(base_vertices))
    template_width = float(np.ptp(template_vertices[:, 0]))
    shaped_width = float(np.ptp(vertices[:, 0]))
    initial_radii = prepared_eye_radii * shaped_width / max(template_width, 1e-12)
    cameras_payload = json.loads((run_dir / "working" / "cameras.json").read_text())
    cameras = [CameraRecord.model_validate(item) for item in cameras_payload["cameras"]]
    left_eye, right_eye, eye_metrics = _fit_eyeballs(
        run_dir, shaped_eye_centers, initial_radii, cameras, config
    )
    vertices = np.asarray(vertices, dtype=np.float64).copy()
    left_contact, left_gap = _constrain_eyelid(
        vertices,
        regions["left_eyelid"],
        left_eye,
        config.anatomy.eyelid_clearance_ratio_max,
    )
    right_contact, right_gap = _constrain_eyelid(
        vertices,
        regions["right_eyelid"],
        right_eye,
        config.anatomy.eyelid_clearance_ratio_max,
    )
    skin_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    topology = _mesh_anatomy(skin_mesh)
    render_vertices = vertices[render_to_skin]
    digest = geometry_hash(render_vertices, render_faces)
    gaps = np.concatenate((left_gap / left_eye.radius, right_gap / right_eye.radius))
    anatomy = {
        "schemaVersion": "2.0.0",
        "unifiedHead": {
            **topology,
            "geometryHash": digest,
            "canonicalTopology": True,
            "proceduralCranium": False,
        },
        "ears": {
            "source": "FLAME-continuous-topology",
            "carrierPresent": False,
            "rootSharedWithScalp": True,
            "leftVertexCount": int(len(regions["left_ear"])),
            "rightVertexCount": int(len(regions["right_ear"])),
        },
        "eyes": {
            **eye_metrics,
            "leftContactVertexCount": int(len(left_contact)),
            "rightContactVertexCount": int(len(right_contact)),
            "contactGapP99R": float(np.quantile(gaps, 0.99)),
            "penetrationCount": int(np.count_nonzero(gaps < -1e-7)),
        },
    }
    asset = UnifiedHeadAsset(
        skin_vertices=vertices,
        skin_faces=np.asarray(faces, dtype=np.int64),
        render_to_skin=render_to_skin,
        render_faces=render_faces,
        uv=uv,
        regions=regions,
        left_eye=left_eye,
        right_eye=right_eye,
        geometry_sha256=digest,
        anatomy=anatomy,
    )
    asset.save(run_dir / "working" / "unified-head.npz")
    atomic_write_json(run_dir / "qa" / "anatomy.json", anatomy)
    return asset
