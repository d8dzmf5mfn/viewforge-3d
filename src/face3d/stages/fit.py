from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional
import trimesh
from PIL import Image

from face3d.config import Face3DConfig
from face3d.errors import fail
from face3d.io import atomic_write_bytes, atomic_write_json
from face3d.models import REQUIRED_VIEWS, CameraRecord, ViewRole
from face3d.profiles import FaceProfileV1, FaceProfileV2
from face3d.stages.flame import FlameModel, FlameRegionMasks


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True).clamp_min(1e-10)
    axis = axis_angle / theta
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        *axis_angle.shape[:-1], 3, 3
    )
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand(
        *axis_angle.shape[:-1], 3, 3
    )
    outer = axis.unsqueeze(-1) * axis.unsqueeze(-2)
    theta = theta.unsqueeze(-1)
    return torch.cos(theta) * identity + (1 - torch.cos(theta)) * outer + torch.sin(theta) * skew


def _project(
    points: torch.Tensor,
    rotation_vector: torch.Tensor,
    translation: torch.Tensor,
    focal: torch.Tensor,
    principal: torch.Tensor,
) -> torch.Tensor:
    rotation = _axis_angle_to_matrix(rotation_vector)
    camera_points = points @ rotation.transpose(-1, -2) + translation
    z = camera_points[..., 2:3].clamp_min(1e-5)
    return camera_points[..., :2] / z * focal + principal


def _initial_camera(
    model_points: np.ndarray,
    image_points: np.ndarray,
    width: int,
    height: int,
    focal: float,
) -> tuple[np.ndarray, np.ndarray]:
    camera = np.asarray(
        [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    ok, rotation, translation = cv2.solvePnP(
        model_points.astype(np.float64),
        image_points.astype(np.float64),
        camera,
        np.zeros((5, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        fail("fit-initialization-failed", "OpenCV PnP 初始化失败", stage="fit")
    return rotation.reshape(3), translation.reshape(3)


def _mask_boundary_distance(mask: np.ndarray) -> np.ndarray:
    boundary = cv2.Canny(mask, 32, 96)
    return cv2.distanceTransform(255 - boundary, cv2.DIST_L2, 5).astype(np.float64)


def _sample_distance_field(distance: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    height, width = distance.shape[-2:]
    normalized = torch.stack(
        (
            points[:, 0] / max(width - 1, 1) * 2 - 1,
            points[:, 1] / max(height - 1, 1) * 2 - 1,
        ),
        dim=-1,
    )
    grid = normalized.reshape(1, 1, -1, 2)
    sampled = torch_functional.grid_sample(
        distance.reshape(1, 1, height, width),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(-1)


def _dynamic_silhouette_points(projected: torch.Tensor, bins: int = 28) -> torch.Tensor:
    detached = projected.detach()
    y_min = torch.quantile(detached[:, 1], 0.01)
    y_max = torch.quantile(detached[:, 1], 0.99)
    edges = torch.linspace(y_min, y_max, bins + 1, dtype=projected.dtype)
    selected: list[torch.Tensor] = []
    for index in range(bins):
        candidates = torch.nonzero(
            (detached[:, 1] >= edges[index]) & (detached[:, 1] < edges[index + 1]),
            as_tuple=False,
        ).flatten()
        if candidates.numel() == 0:
            continue
        xs = detached[candidates, 0]
        selected.extend((candidates[torch.argmin(xs)], candidates[torch.argmax(xs)]))
    if not selected:
        return projected[:0]
    return projected[torch.stack(selected)]


def _render_silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera: CameraRecord,
) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(camera.rotation_vector, dtype=np.float64))
    camera_vertices = vertices @ rotation.T + np.asarray(camera.translation)
    z = np.clip(camera_vertices[:, 2], 1e-6, None)
    pixels = camera_vertices[:, :2] / z[:, None] * camera.focal_length_px
    pixels += np.asarray(camera.principal_point_px)
    visible_faces = faces[np.all(camera_vertices[faces, 2] > 1e-6, axis=1)]
    triangles_float = pixels[visible_faces]
    finite_faces = np.all(np.isfinite(triangles_float), axis=(1, 2))
    bounded_faces = np.all(np.abs(triangles_float) < 1_000_000, axis=(1, 2))
    triangles = np.rint(triangles_float[finite_faces & bounded_faces]).astype(np.int32)
    mask = np.zeros((camera.height, camera.width), dtype=np.uint8)
    # ``fillPoly`` applies an even/odd fill rule across every contour passed in one
    # call. Feeding it a triangle soup therefore cancels adjacent and overlapping
    # triangles and creates false holes. A silhouette is the union of projected
    # triangles, so rasterize each convex triangle independently.
    for triangle in triangles:
        cv2.fillConvexPoly(mask, triangle, 255, lineType=cv2.LINE_8)
    return mask


def _silhouette_iou(rendered: np.ndarray, target: np.ndarray) -> float:
    rendered_fg = rendered > 0
    target_fg = target > 127
    union = np.count_nonzero(rendered_fg | target_fg)
    return float(np.count_nonzero(rendered_fg & target_fg) / max(union, 1))


def _save_silhouette_overlay(
    reference: Path,
    rendered: np.ndarray,
    output: Path,
    target: np.ndarray | None = None,
) -> None:
    rgb = np.asarray(Image.open(reference).convert("RGB")).copy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if target is not None:
        target_contours, _ = cv2.findContours(
            target,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(bgr, target_contours, -1, (238, 80, 210), 2)
    rendered_contours, _ = cv2.findContours(
        rendered,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(bgr, rendered_contours, -1, (64, 226, 128), 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), bgr)


def _export_glb(mesh: trimesh.Trimesh, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=(0.62, 0.65, 0.70, 1.0),
        metallicFactor=0.0,
        roughnessFactor=0.72,
    )
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    atomic_write_bytes(destination, trimesh.exchange.gltf.export_glb(mesh))


def run_fit(run_dir: Path, config: Face3DConfig) -> dict[str, Any]:
    torch.use_deterministic_algorithms(config.deterministic)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    intake_path = run_dir / "working" / "intake.json"
    intake = json.loads(intake_path.read_text(encoding="utf-8"))
    views = {ViewRole(item["role"]): item for item in intake["views"]}
    flame = FlameModel.load(
        config.resolve_asset(config.assets.flame_model),
        config.resolve_asset(config.assets.flame_landmarks),
        config.fit.shape_coefficients,
    )
    dtype = torch.float64
    template = torch.as_tensor(flame.vertices_template, dtype=dtype)
    shapedirs = torch.as_tensor(flame.shape_directions, dtype=dtype)
    faces = torch.as_tensor(flame.faces, dtype=torch.long)
    profile = FaceProfileV2() if config.is_v2 else FaceProfileV1()
    targets: list[torch.Tensor] = []
    dense_targets: list[torch.Tensor] = []
    dense_depth_targets: list[torch.Tensor] = []
    dense_target_numpy: list[np.ndarray] = []
    target_numpy: list[np.ndarray] = []
    landmark_faces: list[torch.Tensor] = []
    landmark_barycentric: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    sizes: list[tuple[int, int]] = []
    initial_focal = float(
        np.mean([max(item["width"], item["height"]) for item in views.values()]) * 1.2
    )
    initial_rotations: list[np.ndarray] = []
    initial_translations: list[np.ndarray] = []
    neutral_vertices = flame.vertices_template
    dense_indices_numpy = np.empty(0, dtype=np.int64)
    if config.is_v2:
        prepared_path = config.resolve_optional_asset(config.assets.flame_prepared)
        masks_path = config.resolve_optional_asset(config.assets.flame_masks)
        if prepared_path is None or masks_path is None:
            fail("asset-missing", "Face v2 缺少稠密映射或区域 masks", stage="fit")
        with np.load(prepared_path, allow_pickle=False) as prepared:
            dense_indices_numpy = np.asarray(prepared["dense_vertex_index"], dtype=np.int64)
        if len(dense_indices_numpy) != 468:
            fail(
                "asset-invalid",
                "Face v2 稠密 MediaPipe-FLAME 映射必须包含 468 点",
                stage="fit",
            )
        flame_regions = FlameRegionMasks.load(masks_path, flame.vertices_template)
    else:
        flame_regions = None

    for role in REQUIRED_VIEWS:
        view = views[role]
        landmark_data = np.load(view["landmarks_path"])
        target = landmark_data["ibug68"].astype(np.float64)
        if config.is_v2:
            dense = landmark_data["all"][:468].astype(np.float64)
            dense_targets.append(
                torch.as_tensor(
                    dense[:, :2] * np.asarray([view["width"], view["height"]]),
                    dtype=dtype,
                )
            )
            dense_target_numpy.append(
                dense[:, :2] * np.asarray([view["width"], view["height"]])
            )
            dense_depth_targets.append(torch.as_tensor(dense[:, 2], dtype=dtype))
        target_numpy.append(target)
        targets.append(torch.as_tensor(target, dtype=dtype))
        dynamic_faces, dynamic_barycentric = flame.landmarks.indices_for_yaw(
            profile.expected_yaw(role)
        )
        landmark_faces.append(torch.as_tensor(dynamic_faces, dtype=torch.long))
        landmark_barycentric.append(torch.as_tensor(dynamic_barycentric, dtype=dtype))
        neutral_landmarks = flame.landmark_vertices(neutral_vertices, profile.expected_yaw(role))
        rotation, translation = _initial_camera(
            neutral_landmarks[17:], target[17:], view["width"], view["height"], initial_focal
        )
        initial_rotations.append(rotation)
        initial_translations.append(translation)
        mask = cv2.imread(view["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            fail("mask-review-required", f"无法读取 mask: {role.value}", stage="fit")
        distances.append(torch.as_tensor(_mask_boundary_distance(mask), dtype=dtype))
        sizes.append((view["width"], view["height"]))

    beta = torch.zeros(config.fit.shape_coefficients, dtype=dtype, requires_grad=True)
    rotations = torch.tensor(np.asarray(initial_rotations), dtype=dtype, requires_grad=True)
    translations = torch.tensor(np.asarray(initial_translations), dtype=dtype, requires_grad=True)
    log_focal = torch.tensor(math.log(initial_focal), dtype=dtype, requires_grad=True)
    parameters = [beta, rotations, translations, log_focal]
    dense_indices = torch.as_tensor(dense_indices_numpy, dtype=torch.long)
    template_mesh = trimesh.Trimesh(
        vertices=flame.vertices_template,
        faces=flame.faces,
        process=False,
    )
    template_normals = torch.as_tensor(template_mesh.vertex_normals, dtype=dtype)
    unique_edges_numpy = np.unique(
        np.sort(
            np.concatenate(
                (
                    flame.faces[:, [0, 1]],
                    flame.faces[:, [1, 2]],
                    flame.faces[:, [2, 0]],
                )
            ),
            axis=1,
        ),
        axis=0,
    )
    unique_edges = torch.as_tensor(unique_edges_numpy, dtype=torch.long)
    observed_mask_numpy = np.zeros(len(flame.vertices_template), dtype=bool)
    if config.is_v2:
        observed_mask_numpy[dense_indices_numpy] = True
        assert flame_regions is not None
        observed_mask_numpy[flame_regions.left_ear] = True
        observed_mask_numpy[flame_regions.right_ear] = True
        for _ in range(2):
            adjacent_faces = np.any(observed_mask_numpy[flame.faces], axis=1)
            observed_mask_numpy[flame.faces[adjacent_faces].reshape(-1)] = True
    observed_mask = torch.as_tensor(observed_mask_numpy.astype(np.float64), dtype=dtype)
    face_width = float(np.ptp(flame.vertices_template[:, 0]))
    maximum_offset = face_width * config.fit.maximum_normal_offset_face_width
    local_offset_raw = torch.zeros(
        len(flame.vertices_template),
        dtype=dtype,
        requires_grad=config.is_v2,
    )
    if config.is_v2:
        parameters.append(local_offset_raw)
    symmetry_pairs = torch.as_tensor(flame.symmetry_pairs, dtype=torch.long)
    center_x = torch.tensor(flame.symmetry_center_x, dtype=dtype)
    landmark_weights = torch.ones(68, dtype=dtype)
    landmark_weights[:17] = 0.7

    def objective() -> torch.Tensor:
        base_vertices = template + torch.einsum("vck,k->vc", shapedirs, beta)
        normal_offset = (
            torch.tanh(local_offset_raw) * maximum_offset * observed_mask
            if config.is_v2
            else torch.zeros_like(local_offset_raw)
        )
        vertices = base_vertices + template_normals * normal_offset[:, None]
        focal = torch.exp(log_focal)
        landmark_loss = torch.zeros((), dtype=dtype)
        contour_loss = torch.zeros((), dtype=dtype)
        dense_landmark_loss = torch.zeros((), dtype=dtype)
        relative_depth_loss = torch.zeros((), dtype=dtype)
        for index, _role in enumerate(REQUIRED_VIEWS):
            triangles = vertices[faces[landmark_faces[index]]]
            landmarks = torch.einsum("lvc,lv->lc", triangles, landmark_barycentric[index])
            principal = torch.tensor([sizes[index][0] / 2, sizes[index][1] / 2], dtype=dtype)
            projected_landmarks = _project(
                landmarks, rotations[index], translations[index], focal, principal
            )
            diagonal = math.hypot(*sizes[index])
            residual = (projected_landmarks - targets[index]) / diagonal
            landmark_loss = landmark_loss + torch.mean(
                landmark_weights * torch.linalg.vector_norm(residual, dim=1)
            )
            projected_vertices = _project(
                vertices, rotations[index], translations[index], focal, principal
            )
            silhouette_points = _dynamic_silhouette_points(projected_vertices)
            contour_loss = contour_loss + torch.mean(
                _sample_distance_field(distances[index], silhouette_points) / diagonal
            )
            if config.is_v2:
                dense_vertices = vertices[dense_indices]
                projected_dense = _project(
                    dense_vertices,
                    rotations[index],
                    translations[index],
                    focal,
                    principal,
                )
                dense_landmark_loss = dense_landmark_loss + torch.mean(
                    torch.linalg.vector_norm(
                        (projected_dense - dense_targets[index]) / diagonal,
                        dim=1,
                    )
                )
                rotation = _axis_angle_to_matrix(rotations[index])
                camera_dense = dense_vertices @ rotation.transpose(-1, -2) + translations[index]
                model_depth = camera_dense[:, 2]
                model_depth = (model_depth - torch.mean(model_depth)) / torch.clamp(
                    torch.std(model_depth), min=1e-6
                )
                target_depth = dense_depth_targets[index]
                target_depth = (target_depth - torch.mean(target_depth)) / torch.clamp(
                    torch.std(target_depth), min=1e-6
                )
                relative_depth_loss = relative_depth_loss + torch_functional.smooth_l1_loss(
                    model_depth,
                    target_depth,
                )
        prior_loss = torch.mean(beta.square())
        if symmetry_pairs.numel():
            first = vertices[symmetry_pairs[:, 0]]
            second = vertices[symmetry_pairs[:, 1]]
            symmetry = torch.stack(
                (
                    first[:, 0] + second[:, 0] - 2 * center_x,
                    first[:, 1] - second[:, 1],
                    first[:, 2] - second[:, 2],
                ),
                dim=1,
            )
            symmetry_loss = torch.mean(symmetry.square())
        else:
            symmetry_loss = torch.zeros((), dtype=dtype)
        focal_prior = (log_focal - math.log(initial_focal)).square()
        normalized_offset = normal_offset / max(maximum_offset, 1e-12)
        local_offset_loss = torch.mean(normalized_offset.square())
        laplacian_loss = torch.mean(
            (normalized_offset[unique_edges[:, 0]] - normalized_offset[unique_edges[:, 1]])
            .square()
        )
        original_edge_length = torch.linalg.vector_norm(
            base_vertices[unique_edges[:, 0]] - base_vertices[unique_edges[:, 1]],
            dim=1,
        )
        deformed_edge_length = torch.linalg.vector_norm(
            vertices[unique_edges[:, 0]] - vertices[unique_edges[:, 1]],
            dim=1,
        )
        arap_loss = torch.mean(
            ((deformed_edge_length - original_edge_length) / max(face_width, 1e-12)).square()
        )
        return (
            config.fit.landmark_weight * landmark_loss
            + config.fit.contour_weight * contour_loss
            + config.fit.shape_prior_weight * prior_loss
            + config.fit.symmetry_weight * symmetry_loss
            + config.fit.dense_landmark_weight * dense_landmark_loss
            + config.fit.relative_depth_weight * relative_depth_loss
            + config.fit.local_offset_weight * local_offset_loss
            + config.fit.laplacian_weight * laplacian_loss
            + config.fit.arap_weight * arap_loss
            + 0.002 * focal_prior
        )

    optimizer = torch.optim.Adam(parameters, lr=config.fit.learning_rate)
    for _ in range(config.fit.adam_iterations):
        optimizer.zero_grad()
        loss = objective()
        if not torch.isfinite(loss):
            fail("fit-non-finite", "联合拟合出现非有限损失", stage="fit")
        loss.backward()
        optimizer.step()
    if config.fit.lbfgs_iterations:
        lbfgs = torch.optim.LBFGS(
            parameters,
            max_iter=config.fit.lbfgs_iterations,
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            value = objective()
            value.backward()
            return value

        lbfgs.step(closure)

    with torch.no_grad():
        base_vertices = template + torch.einsum("vck,k->vc", shapedirs, beta)
        normal_offset = (
            torch.tanh(local_offset_raw) * maximum_offset * observed_mask
            if config.is_v2
            else torch.zeros_like(local_offset_raw)
        )
        vertices = (base_vertices + template_normals * normal_offset[:, None]).cpu().numpy()
        offset_values = normal_offset.cpu().numpy()
        coefficients = beta.cpu().numpy()
        focal = float(torch.exp(log_focal).cpu())
        rotation_values = rotations.cpu().numpy()
        translation_values = translations.cpu().numpy()
        final_loss = float(objective().cpu())

    cameras: list[CameraRecord] = []
    per_view: dict[str, Any] = {}
    gate_failures: list[dict[str, Any]] = []
    fitted_landmarks: dict[str, np.ndarray] = {}
    for index, role in enumerate(REQUIRED_VIEWS):
        view = views[role]
        rotation_matrix, _ = cv2.Rodrigues(rotation_values[index])
        angles = cv2.RQDecomp3x3(rotation_matrix)[0]
        camera = CameraRecord(
            role=role,
            width=view["width"],
            height=view["height"],
            focal_length_px=focal,
            principal_point_px=(view["width"] / 2, view["height"] / 2),
            rotation_vector=tuple(float(value) for value in rotation_values[index]),
            translation=tuple(float(value) for value in translation_values[index]),
            pitch_deg=float(angles[0]),
            yaw_deg=float(angles[1]),
            roll_deg=float(angles[2]),
        )
        cameras.append(camera)
        model_landmarks = flame.landmark_vertices(vertices, profile.expected_yaw(role))
        rotation = rotation_matrix
        camera_points = model_landmarks @ rotation.T + translation_values[index]
        projected = camera_points[:, :2] / np.clip(camera_points[:, 2:3], 1e-6, None) * focal
        projected += np.asarray(camera.principal_point_px)
        fitted_landmarks[role.value] = model_landmarks
        diagonal = float(np.linalg.norm(np.ptp(target_numpy[index], axis=0)))
        nme = float(np.mean(np.linalg.norm(projected - target_numpy[index], axis=1)) / diagonal)
        dense_nme = None
        if config.is_v2:
            dense_camera = vertices[dense_indices_numpy] @ rotation.T + translation_values[index]
            dense_projected = (
                dense_camera[:, :2]
                / np.clip(dense_camera[:, 2:3], 1e-6, None)
                * focal
                + np.asarray(camera.principal_point_px)
            )
            dense_nme = float(
                np.mean(
                    np.linalg.norm(dense_projected - dense_target_numpy[index], axis=1)
                )
                / diagonal
            )
        rendered = _render_silhouette(vertices, flame.faces, camera)
        target_mask = cv2.imread(view["mask_path"], cv2.IMREAD_GRAYSCALE)
        iou = _silhouette_iou(rendered, target_mask)
        _save_silhouette_overlay(
            Path(view["normalized_path"]),
            rendered,
            run_dir / "overlays" / f"fit-silhouette-{role.value}.png",
        )
        if config.is_v2:
            nme_limit = (
                config.acceptance.front_landmark_nme_v2_max
                if role == ViewRole.FRONT
                else config.acceptance.side_landmark_nme_v2_max
            )
            iou_limit = (
                config.acceptance.front_silhouette_iou_v2_min
                if role == ViewRole.FRONT
                else config.acceptance.side_silhouette_iou_v2_min
            )
        else:
            nme_limit = (
                config.acceptance.front_landmark_nme_max
                if role == ViewRole.FRONT
                else config.acceptance.side_landmark_nme_max
            )
            iou_limit = (
                config.acceptance.front_silhouette_iou_min
                if role == ViewRole.FRONT
                else config.acceptance.side_silhouette_iou_min
            )
        per_view[role.value] = {
            "landmarkNME": nme,
            "denseLandmarkNME": dense_nme,
            "landmarkErrorPx": float(nme * diagonal),
            "silhouetteIoU": iou,
            "landmarkThreshold": nme_limit,
            "silhouetteThreshold": iou_limit,
            "passed": (
                nme <= nme_limit
                and (dense_nme is None or dense_nme <= nme_limit)
                and iou >= iou_limit
            ),
        }
        if nme > nme_limit or (dense_nme is not None and dense_nme > nme_limit) or iou < iou_limit:
            gate_failures.append({"role": role.value, **per_view[role.value]})

    working = run_dir / "working"
    fit_npz = working / "fit.npz"
    fit_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fit_npz,
        vertices=vertices.astype(np.float32),
        faces=flame.faces.astype(np.int32),
        shape_coefficients=coefficients.astype(np.float32),
        normal_offsets=offset_values.astype(np.float32),
        feature_landmarks=np.stack([fitted_landmarks[role.value] for role in REQUIRED_VIEWS]),
    )
    cameras_path = working / "cameras.json"
    atomic_write_json(
        cameras_path,
        {
            "schemaVersion": 2 if config.is_v2 else 1,
            "cameras": [camera.model_dump(mode="json") for camera in cameras],
        },
    )
    metrics = {
        "objective": final_loss,
        "perView": per_view,
        "shapeCoefficientCount": config.fit.shape_coefficients,
        "denseLandmarkCount": int(len(dense_indices_numpy)),
        "normalOffsetVertexCount": int(np.count_nonzero(np.abs(offset_values) > 1e-8)),
        "normalOffsetMaximumFaceWidth": float(
            np.max(np.abs(offset_values), initial=0.0) / max(face_width, 1e-12)
        ),
        "regularization": {
            "laplacian": config.fit.laplacian_weight,
            "arap": config.fit.arap_weight,
        },
        "sharedFocalLengthPx": focal,
        "passed": not gate_failures,
    }
    metrics_path = working / "fit-metrics.json"
    atomic_write_json(metrics_path, metrics)
    fitted_mesh = trimesh.Trimesh(vertices=vertices, faces=flame.faces, process=False)
    _export_glb(fitted_mesh, run_dir / "models" / "fitted.glb")
    if gate_failures:
        fail(
            "fit-gate-failed",
            "三视图联合拟合未通过 Gate B",
            stage="fit",
            details={"failures": gate_failures, "metrics": str(metrics_path)},
        )
    return metrics
