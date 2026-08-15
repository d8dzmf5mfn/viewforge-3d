from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh
from PIL import Image

from face3d.config import Face3DConfig, load_config
from face3d.io import atomic_write_json
from face3d.models import REQUIRED_VIEWS, CameraRecord, ViewRole
from face3d.package import package_run
from face3d.profiles.face_v1 import MEDIAPIPE_TO_IBUG68
from face3d.report import build_report, write_notices
from face3d.stages.fit import _render_silhouette
from face3d.stages.intake import confirm_masks
from face3d.stages.template_fit import (
    _head_vertex_indices,
    _load_template,
    _observed_vertex_mask,
    collision_safe_deformation,
    deformation_stability_weights,
    eyelid_contact_corrected_eye,
    eyelid_support_corrected_vertices,
    run_template_fit,
    spectral_deformation_basis,
    triangle_orientation_metrics,
)
from face3d.stages.template_qa import run_template_qa
from face3d.stages.template_skin import run_template_skin
from face3d.unified_head import EyeballAsset, UnifiedHeadAsset


def _require_optional_template_assets(config: Face3DConfig) -> None:
    template_head = config.resolve_optional_asset(config.assets.template_head)
    template_landmarks = config.resolve_optional_asset(config.assets.template_landmarks)
    template_manifest = config.resolve_optional_asset(config.assets.template_manifest)
    paths = (
        template_head,
        template_landmarks,
        template_manifest,
        config.resolve_asset(config.skin.uv_albedo_source),
        config.resolve_asset(config.skin.micro_albedo_source),
    )
    if any(path is None or not path.is_file() for path in paths):
        pytest.skip("optional local template and material assets are not installed")


def _project(points: np.ndarray, camera: CameraRecord) -> tuple[np.ndarray, np.ndarray]:
    rotation, _ = cv2.Rodrigues(np.asarray(camera.rotation_vector, dtype=np.float64))
    camera_points = points @ rotation.T + np.asarray(camera.translation, dtype=np.float64)
    pixels = camera_points[:, :2] / camera_points[:, 2:3] * camera.focal_length_px + np.asarray(
        camera.principal_point_px
    )
    return pixels, camera_points[:, 2]


def _synthetic_camera(
    role: ViewRole,
    vertices: np.ndarray,
    angle_deg: float,
    size: int,
) -> CameraRecord:
    center = np.mean(np.stack((vertices.min(axis=0), vertices.max(axis=0))), axis=0)
    rotation = trimesh.transformations.rotation_matrix(np.pi, (1, 0, 0))[:3, :3]
    rotation = (
        rotation
        @ trimesh.transformations.rotation_matrix(
            np.deg2rad(angle_deg),
            (0, 1, 0),
        )[:3, :3]
    )
    rotation_vector, _ = cv2.Rodrigues(rotation)
    rotation_vector = rotation_vector.reshape(3)
    translation = np.asarray([0.0, 0.0, 8.0]) - center @ rotation.T
    return CameraRecord(
        role=role,
        width=size,
        height=size,
        focal_length_px=size * 1.2,
        principal_point_px=(size / 2, size / 2),
        rotation_vector=tuple(rotation_vector),
        translation=tuple(translation),
        yaw_deg=angle_deg,
        pitch_deg=0.0,
        roll_deg=0.0,
    )


def _known_nonrigid_identity(
    asset: UnifiedHeadAsset,
    basis_size: int,
    variant: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    vertices = np.asarray(asset.skin_vertices, dtype=np.float64)
    faces = np.asarray(asset.skin_faces, dtype=np.int64)
    head_vertices = vertices[_head_vertex_indices(asset)]
    face_width = float(np.ptp(head_vertices[:, 0]))
    center = np.mean(
        np.stack((head_vertices.min(axis=0), head_vertices.max(axis=0))),
        axis=0,
    )
    profiles = (
        {
            "scale": (1.035, 0.982, 1.028),
            "nose": 0.0070,
            "jaw": 0.0120,
            "crown": 0.0040,
            "ear": (0.0060, 0.0020, -0.0015),
        },
        {
            "scale": (0.972, 1.026, 0.968),
            "nose": 0.0060,
            "jaw": -0.0100,
            "crown": 0.0050,
            "ear": (0.0070, -0.0015, 0.0010),
        },
        {
            "scale": (1.018, 1.034, 1.044),
            "nose": 0.0080,
            "jaw": 0.0140,
            "crown": -0.0030,
            "ear": (0.0065, 0.0025, -0.0020),
        },
    )
    if variant < 0 or variant >= len(profiles):
        raise ValueError(f"unknown synthetic identity variant: {variant}")
    profile = profiles[variant]
    scale = np.asarray(profile["scale"], dtype=np.float64)
    scaled = (vertices - center) * scale + center

    if basis_size < 8:
        raise ValueError("known identity benchmark requires at least eight low-frequency modes")
    stability, _ = deformation_stability_weights(vertices, faces)
    truth = scaled.copy()

    def gaussian(center_point: np.ndarray, radius: float) -> np.ndarray:
        distance = np.linalg.norm(scaled - center_point, axis=1) / max(radius, 1e-12)
        return np.exp(-0.5 * distance**2)

    nose_center = np.mean(scaled[np.asarray(asset.regions["nose"], dtype=np.int64)], axis=0)
    jaw_center = np.mean(scaled[np.asarray(asset.regions["jaw"], dtype=np.int64)], axis=0)
    crown_center = np.asarray(
        [
            center[0],
            center[1] + (np.max(head_vertices[:, 1]) - center[1]) * scale[1],
            center[2],
        ],
        dtype=np.float64,
    )
    nose_weight = gaussian(nose_center, face_width * 0.18) * stability
    jaw_weight = gaussian(jaw_center, face_width * 0.28) * stability
    crown_weight = gaussian(crown_center, face_width * 0.34) * stability
    truth[:, 2] += nose_weight * face_width * float(profile["nose"])
    truth[:, 0] += (scaled[:, 0] - center[0]) * jaw_weight * float(profile["jaw"])
    truth[:, 1] += crown_weight * face_width * float(profile["crown"])

    ear_nonrigid_maximum = 0.0
    for side, name in ((1.0, "left_ear"), (-1.0, "right_ear")):
        region = np.asarray(asset.regions[name], dtype=np.int64)
        ear_center = np.mean(scaled[region], axis=0)
        influence = gaussian(ear_center, face_width * 0.13) * stability
        ear_outward, ear_vertical, ear_depth = profile["ear"]
        ear_delta = np.column_stack(
            (
                side * influence * face_width * float(ear_outward),
                influence * face_width * float(ear_vertical),
                influence * face_width * float(ear_depth),
            )
        )
        truth += ear_delta
        ear_nonrigid_maximum = max(
            ear_nonrigid_maximum,
            float(np.max(np.linalg.norm(ear_delta[region], axis=1)) / face_width),
        )

    candidate_delta = truth - scaled
    accepted_fraction = 0.0
    for exponent in range(14):
        fraction = 0.5**exponent
        candidate = scaled + candidate_delta * fraction
        orientation = triangle_orientation_metrics(scaled, candidate, faces)
        if (
            orientation["flippedTriangleCount"] == 0
            and orientation["minimumSignedAreaRatio"] > 0.10
        ):
            truth = candidate
            accepted_fraction = fraction
            break
    assert accepted_fraction > 0.0
    orientation = triangle_orientation_metrics(scaled, truth, faces)
    assert orientation["flippedTriangleCount"] == 0
    assert orientation["minimumSignedAreaRatio"] > 0.10
    return truth.astype(np.float64), {
        "faceWidth": face_width,
        "variant": variant,
        "maximumDisplacementFaceWidth": float(
            np.max(np.linalg.norm(truth - vertices, axis=1)) / face_width
        ),
        "maximumNonrigidDisplacementFaceWidth": float(
            np.max(np.linalg.norm(truth - scaled, axis=1)) / face_width
        ),
        "observedMeanNonrigidDisplacementFaceWidth": float(
            np.mean(
                np.linalg.norm(
                    truth[_observed_vertex_mask(asset)] - scaled[_observed_vertex_mask(asset)],
                    axis=1,
                )
            )
            / face_width
        ),
        "acceptedNonrigidFraction": accepted_fraction,
        "earNonrigidMaximumFaceWidth": ear_nonrigid_maximum * accepted_fraction,
    }


def _similarity_align(
    source: np.ndarray,
    target: np.ndarray,
    fit_mask: np.ndarray,
) -> np.ndarray:
    source_fit = np.asarray(source, dtype=np.float64)[fit_mask]
    target_fit = np.asarray(target, dtype=np.float64)[fit_mask]
    source_center = np.mean(source_fit, axis=0)
    target_center = np.mean(target_fit, axis=0)
    source_centered = source_fit - source_center
    target_centered = target_fit - target_center
    left, singular_values, right = np.linalg.svd(source_centered.T @ target_centered)
    rotation = left @ right
    aligned_variance = float(np.sum(singular_values))
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
        aligned_variance -= 2.0 * float(singular_values[-1])
    scale = float(aligned_variance / max(float(np.sum(source_centered * source_centered)), 1e-12))
    return (np.asarray(source, dtype=np.float64) - source_center) @ rotation * scale + (
        target_center
    )


def test_spectral_basis_is_deterministic_and_finite() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    vertices[:, 0] += np.linspace(0.0, 0.013, len(vertices))

    first_basis, first_values = spectral_deformation_basis(vertices, mesh.faces, 8)
    second_basis, second_values = spectral_deformation_basis(vertices, mesh.faces, 8)

    assert first_basis.shape == (len(vertices), 8)
    assert np.isfinite(first_basis).all()
    assert np.all(first_values > 0)
    assert np.allclose(first_values, second_values, atol=1e-10)
    assert np.allclose(first_basis, second_basis, atol=1e-8)


def test_triangle_orientation_metrics_detects_flip() -> None:
    reference = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    deformed = reference[[0, 2, 1]]
    metrics = triangle_orientation_metrics(reference, deformed, np.asarray([[0, 1, 2]]))

    assert metrics["flippedTriangleCount"] == 1
    assert metrics["minimumSignedAreaRatio"] < 0


def test_collision_safe_deformation_separates_intersecting_triangles() -> None:
    reference = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -0.5, 1.5],
            [0.0, 0.5, 1.5],
            [0.0, 0.0, 2.5],
        ],
        dtype=np.float64,
    )
    candidate = reference.copy()
    candidate[3:, 2] -= 2.0
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)

    corrected, metrics = collision_safe_deformation(reference, candidate, faces)

    assert metrics["initialPairCount"] == 1
    assert metrics["finalPairCount"] == 0
    assert metrics["finalUnsafeTriangleCount"] == 0
    assert metrics["passed"] is True
    assert not np.allclose(corrected, candidate)


def test_collision_safe_deformation_restores_triangle_area_margin() -> None:
    reference = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    candidate = reference.copy()
    candidate[2, 1] = 0.01
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)

    corrected, metrics = collision_safe_deformation(reference, candidate, faces)
    orientation = triangle_orientation_metrics(reference, corrected, faces)

    assert metrics["initialUnsafeTriangleCount"] == 1
    assert metrics["finalUnsafeTriangleCount"] == 0
    assert metrics["passed"] is True
    assert orientation["minimumSignedAreaRatio"] >= 0.03


def test_eyelid_contact_correction_minimally_grows_eye() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    vertices = np.column_stack(
        (
            np.cos(angles) * 1.0308,
            np.sin(angles) * 1.0308,
            np.zeros_like(angles),
        )
    )
    eye = EyeballAsset(
        center=np.zeros(3, dtype=np.float64),
        radius=1.0,
        gaze=np.asarray([0.0, 0.0, 1.0]),
    )

    corrected, metrics = eyelid_contact_corrected_eye(
        eye,
        vertices,
        np.arange(len(vertices)),
        0.03,
    )

    assert corrected.radius > eye.radius
    assert corrected.radius <= metrics["maximumRadius"]
    assert metrics["contactGapP99R"] <= 0.03
    assert metrics["passed"] is True


def test_eyelid_support_clamps_only_active_contact_band() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    radii = np.linspace(0.999, 1.0308, len(angles))
    vertices = np.column_stack(
        (
            np.cos(angles) * radii,
            np.sin(angles) * radii,
            np.zeros_like(angles),
        )
    )
    eye = EyeballAsset(
        center=np.zeros(3, dtype=np.float64),
        radius=1.0,
        gaze=np.asarray([0.0, 0.0, 1.0]),
    )

    corrected, metrics = eyelid_support_corrected_vertices(
        vertices,
        eye,
        np.arange(len(vertices)),
        0.03,
    )
    gaps = np.linalg.norm(corrected, axis=1) - 1.0

    assert metrics["movedVertexCount"] > 0
    assert float(np.min(gaps)) >= 0.001 - 1e-6
    assert float(np.max(gaps)) <= 0.0295 + 1e-6


def test_deformation_stability_weights_lock_fragile_faces() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1e-10, 0.0, 0.0],
            [0.0, 1e-10, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 3, 4], [1, 5, 2]], dtype=np.int64)

    weights, metrics = deformation_stability_weights(vertices, faces)

    assert metrics["fragileFaceCount"] == 1
    assert np.all(weights[[0, 3, 4]] == 0.0)
    assert np.all((weights >= 0.0) & (weights <= 1.0))


def test_cpu_silhouette_rasterizer_unions_triangle_soup_without_holes() -> None:
    grid = np.linspace(-1.0, 1.0, 11)
    vertices = np.asarray(
        [(x, y, 0.0) for y in grid for x in grid],
        dtype=np.float64,
    )
    faces: list[tuple[int, int, int]] = []
    for row in range(10):
        for column in range(10):
            lower_left = row * 11 + column
            lower_right = lower_left + 1
            upper_left = lower_left + 11
            upper_right = upper_left + 1
            faces.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )
    camera = CameraRecord(
        role=ViewRole.FRONT,
        width=512,
        height=512,
        focal_length_px=200.0,
        principal_point_px=(256.0, 256.0),
        rotation_vector=(0.0, 0.0, 0.0),
        translation=(0.0, 0.0, 2.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )

    rendered = _render_silhouette(vertices, np.asarray(faces, dtype=np.int64), camera)

    assert np.all(rendered[157:356, 157:356] == 255)


def test_template_fit_reproduces_exact_synthetic_three_views(tmp_path: Path) -> None:
    base = load_config(Path("configs/face-v3.yaml"))
    _require_optional_template_assets(base)
    config = base.model_copy(
        update={
            "fit": base.fit.model_copy(
                update={
                    "adam_iterations": 3,
                    "lbfgs_iterations": 0,
                    "low_frequency_basis_size": 8,
                }
            ),
            "acceptance": base.acceptance.model_copy(
                update={
                    "front_landmark_nme_v2_max": 0.002,
                    "side_landmark_nme_v2_max": 0.002,
                    "front_silhouette_iou_v2_min": 0.99,
                    "side_silhouette_iou_v2_min": 0.99,
                }
            ),
            "skin": base.skin.model_copy(
                update={
                    "atlas_resolution": 256,
                    "detail_resolution": 256,
                }
            ),
        }
    )
    asset, binding = _load_template(config)
    run = tmp_path / "run"
    (run / "working" / "landmarks").mkdir(parents=True)
    (run / "working" / "masks").mkdir(parents=True)
    (run / "references").mkdir(parents=True)
    (run / "overlays").mkdir(parents=True)
    size = 512
    angles = {
        ViewRole.FRONT: 0.0,
        ViewRole.LEFT45: -45.0,
        ViewRole.RIGHT45: 45.0,
    }
    views = []
    for role in REQUIRED_VIEWS:
        camera = _synthetic_camera(role, asset.skin_vertices, angles[role], size)
        pixels, depth = _project(binding.points[binding.valid], camera)
        landmarks = np.zeros((478, 3), dtype=np.float32)
        landmarks[:, :2] = 0.5
        landmarks[binding.valid, :2] = pixels / size
        landmarks[binding.valid, 2] = depth
        landmark_path = run / "working" / "landmarks" / f"{role.value}.npz"
        np.savez_compressed(
            landmark_path,
            all=landmarks,
            ibug68=(landmarks[np.asarray(MEDIAPIPE_TO_IBUG68), :2] * size),
        )
        mask = _render_silhouette(asset.skin_vertices, asset.skin_faces, camera)
        mask_path = run / "working" / "masks" / f"{role.value}.png"
        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(run / "overlays" / f"landmarks-{role.value}.png"), mask)
        cv2.imwrite(str(run / "overlays" / f"silhouette-{role.value}.png"), mask)
        reference = run / "references" / f"{role.value}.png"
        Image.new("RGB", (size, size), (28, 30, 32)).save(reference)
        views.append(
            {
                "role": role.value,
                "width": size,
                "height": size,
                "landmarks_path": str(landmark_path),
                "mask_path": str(mask_path),
                "normalized_path": str(reference),
            }
        )
    atomic_write_json(
        run / "working" / "intake.json",
        {"contractVersion": 3, "views": views},
    )
    confirm_masks(run)

    metrics = run_template_fit(run, config)

    assert metrics["passed"] is True
    assert metrics["sdfUsed"] is False
    assert metrics["orientation"]["flippedTriangleCount"] == 0
    assert metrics["maximumDisplacementFaceWidth"] < 0.001
    assert all(value["silhouetteIoU"] >= 0.99 for value in metrics["perView"].values())
    scene = trimesh.load(run / "models" / "fitted-head.glb", force="scene")
    assert {"HeadSkin", "Eyeball.L", "Eyeball.R"} <= set(scene.graph.nodes_geometry)

    skin = run_template_skin(run, config)
    assert skin["passed"]
    assert skin["geometryRecreated"] is False
    assert skin["neutralAndSkinSharePositionIndex"] is True
    assert skin["geometryHash"] == metrics["fittedGeometrySha256"]
    assert skin["traceRecordCount"] == len(asset.skin_vertices)
    assert (run / "models" / "head.glb").is_file()

    qa = run_template_qa(run, config)
    assert qa["passed"]
    assert qa["geometry"]["selfIntersectionPairCount"] == 0
    assert qa["eyes"]["intersectionCount"] == 0
    assert qa["eyes"]["fittingBindingPassed"] is True
    assert qa["eyes"]["contourSymmetry"]["passed"] is True
    assert qa["sdf"]["role"] == "qa-only"
    assert qa["sdf"]["surfaceGenerated"] is False

    manifest, report = build_report(
        run,
        config,
        {"elapsedSeconds": 1.0, "peakRssBytes": 512 * 1024**2, "deterministic": True},
    )
    write_notices(run)
    assert manifest["schemaVersion"] == "3.0.0"
    assert "pixel" not in manifest
    assert "voxel" not in manifest
    assert manifest["sdf"]["role"] == "qa-only"
    assert report["summary"]["automatedGatesPassed"] is True
    assert report["summary"]["finalAcceptance"] is False
    assert report["summary"]["visualBaselineReviewed"] is False
    assert (run / "qa" / "fixed-view-side.png").is_file()
    assert (run / "qa" / "fixed-view-skin-side.png").is_file()

    package = tmp_path / "synthetic.face3d"
    result = package_run(run, package, config)
    assert result["ok"]
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
    assert "models/head.glb" in names
    assert "projection/skin-projection.npz" in names
    assert "models/voxels.glb" not in names
    assert "pixels/pixels.bin" not in names


@pytest.mark.parametrize("variant", (0, 1, 2))
def test_template_fit_recovers_known_nonrigid_identity_and_side_profile(
    tmp_path: Path,
    variant: int,
) -> None:
    base = load_config(Path("configs/face-v3.yaml"))
    _require_optional_template_assets(base)
    config = base.model_copy(
        update={
            "fit": base.fit.model_copy(
                update={
                    "adam_iterations": 500,
                    "lbfgs_iterations": 100,
                    "low_frequency_basis_size": 16,
                }
            )
        }
    )
    asset, binding = _load_template(config)
    truth_vertices, truth_metrics = _known_nonrigid_identity(
        asset,
        config.fit.low_frequency_basis_size,
        variant,
    )
    run = tmp_path / "known-identity-run"
    (run / "working" / "landmarks").mkdir(parents=True)
    (run / "working" / "masks").mkdir(parents=True)
    (run / "references").mkdir(parents=True)
    size = 512
    angles = {
        ViewRole.FRONT: 0.0,
        ViewRole.LEFT45: -45.0,
        ViewRole.RIGHT45: 45.0,
    }
    truth_points = np.einsum(
        "nvc,nv->nc",
        truth_vertices[asset.skin_faces[binding.triangle[binding.valid]]],
        binding.barycentric[binding.valid],
    )
    views = []
    for role in REQUIRED_VIEWS:
        camera = _synthetic_camera(role, truth_vertices, angles[role], size)
        pixels, depth = _project(truth_points, camera)
        landmarks = np.zeros((478, 3), dtype=np.float32)
        landmarks[:, :2] = 0.5
        landmarks[binding.valid, :2] = pixels / size
        landmarks[binding.valid, 2] = depth
        landmark_path = run / "working" / "landmarks" / f"{role.value}.npz"
        np.savez_compressed(
            landmark_path,
            all=landmarks,
            ibug68=(landmarks[np.asarray(MEDIAPIPE_TO_IBUG68), :2] * size),
        )
        mask = _render_silhouette(truth_vertices, asset.skin_faces, camera)
        mask_path = run / "working" / "masks" / f"{role.value}.png"
        cv2.imwrite(str(mask_path), mask)
        reference = run / "references" / f"{role.value}.png"
        Image.new("RGB", (size, size), (116, 88, 73)).save(reference)
        views.append(
            {
                "role": role.value,
                "width": size,
                "height": size,
                "landmarks_path": str(landmark_path),
                "mask_path": str(mask_path),
                "normalized_path": str(reference),
            }
        )
    atomic_write_json(
        run / "working" / "intake.json",
        {"contractVersion": 3, "views": views},
    )
    confirm_masks(run)

    fit_metrics = run_template_fit(run, config)
    skin_metrics = run_template_skin(run, config)
    qa_metrics = run_template_qa(run, config)
    fitted = UnifiedHeadAsset.load(run / "working" / "unified-head.npz")
    fitted_vertices = np.asarray(fitted.skin_vertices, dtype=np.float64)
    face_width = truth_metrics["faceWidth"]
    observed = _observed_vertex_mask(asset)
    observed_error = (
        np.linalg.norm(
            fitted_vertices[observed] - truth_vertices[observed],
            axis=1,
        )
        / face_width
    )
    template_error = (
        np.linalg.norm(
            asset.skin_vertices[observed] - truth_vertices[observed],
            axis=1,
        )
        / face_width
    )
    ear_indices = np.unique(np.concatenate((asset.regions["left_ear"], asset.regions["right_ear"])))
    ear_error = (
        np.linalg.norm(
            fitted_vertices[ear_indices] - truth_vertices[ear_indices],
            axis=1,
        )
        / face_width
    )
    aligned_fitted = _similarity_align(fitted_vertices, truth_vertices, observed)
    aligned_template = _similarity_align(asset.skin_vertices, truth_vertices, observed)
    aligned_observed_error = (
        np.linalg.norm(
            aligned_fitted[observed] - truth_vertices[observed],
            axis=1,
        )
        / face_width
    )
    aligned_template_error = (
        np.linalg.norm(
            aligned_template[observed] - truth_vertices[observed],
            axis=1,
        )
        / face_width
    )
    aligned_ear_error = (
        np.linalg.norm(
            aligned_fitted[ear_indices] - truth_vertices[ear_indices],
            axis=1,
        )
        / face_width
    )

    side_camera = _synthetic_camera(ViewRole.LEFT45, truth_vertices, 90.0, size)
    truth_side = _render_silhouette(truth_vertices, asset.skin_faces, side_camera)
    fitted_side = _render_silhouette(fitted_vertices, asset.skin_faces, side_camera)
    side_iou = float(np.count_nonzero((truth_side > 0) & (fitted_side > 0))) / max(
        np.count_nonzero((truth_side > 0) | (fitted_side > 0)),
        1,
    )

    benchmark = {
        "schemaVersion": 1,
        "variant": variant,
        "coordinateGauge": "similarity-aligned-observed-region",
        "directErrorRole": "diagnostic-only-absolute-scale-is-unobservable",
        "truthMaximumDisplacementFaceWidth": truth_metrics["maximumDisplacementFaceWidth"],
        "truthMaximumNonrigidDisplacementFaceWidth": truth_metrics[
            "maximumNonrigidDisplacementFaceWidth"
        ],
        "truthEarNonrigidMaximumFaceWidth": truth_metrics["earNonrigidMaximumFaceWidth"],
        "truthAcceptedNonrigidFraction": truth_metrics["acceptedNonrigidFraction"],
        "observedMeanDirectErrorFaceWidth": float(np.mean(observed_error)),
        "observedP95DirectErrorFaceWidth": float(np.quantile(observed_error, 0.95)),
        "earMeanDirectErrorFaceWidth": float(np.mean(ear_error)),
        "observedMeanSimilarityAlignedErrorFaceWidth": float(np.mean(aligned_observed_error)),
        "observedP95SimilarityAlignedErrorFaceWidth": float(
            np.quantile(aligned_observed_error, 0.95)
        ),
        "earMeanSimilarityAlignedErrorFaceWidth": float(np.mean(aligned_ear_error)),
        "templateObservedMeanErrorFaceWidth": float(np.mean(template_error)),
        "templateSimilarityAlignedMeanErrorFaceWidth": float(np.mean(aligned_template_error)),
        "errorReductionRatio": float(
            1.0 - np.mean(aligned_observed_error) / max(np.mean(aligned_template_error), 1e-12)
        ),
        "pureSideSilhouetteIoU": side_iou,
        "fitPassed": fit_metrics["passed"],
        "flippedTriangleCount": fit_metrics["orientation"]["flippedTriangleCount"],
        "minimumSignedAreaRatio": fit_metrics["orientation"]["minimumSignedAreaRatio"],
        "collisionFinalization": fit_metrics["collisionFinalization"],
        "postContactCollisionFinalization": fit_metrics["postContactCollisionFinalization"],
        "eyelidSupportFinalization": fit_metrics["eyelidSupportFinalization"],
        "skinMaximumVertexDifference": skin_metrics["maximumVertexDifference"],
        "skinGeometryHashesMatch": bool(
            skin_metrics["geometryHash"]
            == skin_metrics["neutralGeometryHash"]
            == skin_metrics["skinGeometryHash"]
        ),
        "selfIntersectionPairCount": qa_metrics["geometry"]["selfIntersectionPairCount"],
        "eyeIntersectionCount": qa_metrics["eyes"]["intersectionCount"],
        "leftEyelidContactGapP99R": qa_metrics["eyes"]["left"]["contactGapP99R"],
        "rightEyelidContactGapP99R": qa_metrics["eyes"]["right"]["contactGapP99R"],
        "sdfRole": qa_metrics["sdf"]["role"],
        "sdfSurfaceGenerated": qa_metrics["sdf"]["surfaceGenerated"],
        "fullQaPassed": qa_metrics["passed"],
    }
    atomic_write_json(run / "qa" / "synthetic-ground-truth.json", benchmark)

    contract = json.loads(
        Path("quality/template-head-v0-contract.json").read_text(encoding="utf-8")
    )["syntheticTruth"]["required"]
    assert (
        truth_metrics["maximumDisplacementFaceWidth"]
        >= contract["minimumTruthMaximumDisplacementFaceWidth"]
    )
    assert (
        truth_metrics["maximumNonrigidDisplacementFaceWidth"]
        >= contract["minimumTruthMaximumNonrigidDisplacementFaceWidth"]
    )
    assert (
        truth_metrics["earNonrigidMaximumFaceWidth"]
        >= contract["minimumTruthEarNonrigidDisplacementFaceWidth"]
    )
    assert truth_metrics["acceptedNonrigidFraction"] == contract["minimumAcceptedNonrigidFraction"]
    assert fit_metrics["passed"] is True
    assert skin_metrics["passed"] is True
    assert qa_metrics["passed"] is True
    assert np.isfinite(observed_error).all()
    assert np.isfinite(ear_error).all()
    assert (
        float(np.mean(aligned_observed_error))
        <= contract["maximumObservedMeanSimilarityAlignedErrorFaceWidth"]
    )
    assert (
        float(np.quantile(aligned_observed_error, 0.95))
        <= contract["maximumObservedP95SimilarityAlignedErrorFaceWidth"]
    )
    assert (
        float(np.mean(aligned_ear_error))
        <= contract["maximumEarMeanSimilarityAlignedErrorFaceWidth"]
    )
    assert (
        float(np.mean(aligned_observed_error))
        <= float(np.mean(aligned_template_error)) * contract["maximumFittedToTemplateErrorRatio"]
    )
    assert side_iou >= contract["minimumPureSideSilhouetteIoU"]
    assert (
        fit_metrics["orientation"]["flippedTriangleCount"]
        <= contract["maximumFlippedTriangleCount"]
    )
    assert (
        fit_metrics["orientation"]["minimumSignedAreaRatio"] >= contract["minimumSignedAreaRatio"]
    )
    assert (
        fit_metrics["collisionFinalization"]["meanRetainedDisplacementFraction"]
        >= contract["minimumCollisionMeanRetainedDisplacementFraction"]
    )
    assert (
        fit_metrics["collisionFinalization"]["touchedVertexCount"] / len(fitted_vertices)
        <= contract["maximumCollisionTouchedVertexFraction"]
    )
    assert (
        fit_metrics["collisionFinalization"]["finalUnsafeTriangleCount"]
        <= contract["maximumCollisionFinalUnsafeTriangleCount"]
    )
    assert (
        fit_metrics["postContactCollisionFinalization"]["finalPairCount"]
        <= contract["maximumPostContactSelfIntersectionPairCount"]
    )
    assert (
        fit_metrics["postContactCollisionFinalization"]["finalUnsafeTriangleCount"]
        <= contract["maximumPostContactUnsafeTriangleCount"]
    )
    for side in ("left", "right"):
        assert (
            fit_metrics["eyelidSupportFinalization"][side]["maximumMovementEyeRadius"]
            <= contract["maximumEyelidSupportMovementEyeRadius"]
        )
    assert skin_metrics["maximumVertexDifference"] <= contract["maximumSkinVertexDifference"]
    assert (
        skin_metrics["geometryHash"]
        == skin_metrics["neutralGeometryHash"]
        == skin_metrics["skinGeometryHash"]
    )
    assert (
        qa_metrics["geometry"]["selfIntersectionPairCount"]
        <= contract["maximumSelfIntersectionPairCount"]
    )
    assert qa_metrics["eyes"]["intersectionCount"] <= contract["maximumEyeIntersectionCount"]
    assert qa_metrics["eyes"]["left"]["contactGapP99R"] <= contract["maximumEyelidGapP99R"]
    assert qa_metrics["eyes"]["right"]["contactGapP99R"] <= contract["maximumEyelidGapP99R"]
    assert qa_metrics["sdf"]["role"] == "qa-only"
    assert qa_metrics["sdf"]["surfaceGenerated"] is False
