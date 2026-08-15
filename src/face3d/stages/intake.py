from __future__ import annotations

import io
import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageOps
from pillow_heif import register_heif_opener

from face3d.config import Face3DConfig
from face3d.errors import Face3DError, fail
from face3d.io import atomic_write_json, read_json, sha256_file
from face3d.models import REQUIRED_VIEWS, ViewRecord, ViewRole
from face3d.profiles import FaceProfileV1, FaceProfileV2, FaceProfileV3

register_heif_opener()

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".heic", ".heif")
FACE_OVAL = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)
PNP_INDICES = (1, 152, 33, 263, 61, 291)
PNP_MODEL = np.asarray(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)
PNP_MODEL_TO_CAMERA = np.diag([1.0, -1.0, -1.0])


def discover_views(input_dir: Path) -> dict[ViewRole, Path]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        fail("input-missing", f"输入目录不存在: {input_dir}", stage="input")
    found: dict[ViewRole, Path] = {}
    for role in REQUIRED_VIEWS:
        candidates = [input_dir / f"{role.value}{suffix}" for suffix in SUPPORTED_EXTENSIONS]
        existing = [path for path in candidates if path.is_file()]
        if len(existing) > 1:
            fail(
                "duplicate-view",
                f"角色 {role.value} 有多个候选文件",
                stage="input",
                details={"files": [str(path) for path in existing]},
            )
        if existing:
            found[role] = existing[0]
    missing = [role.value for role in REQUIRED_VIEWS if role not in found]
    if missing:
        fail(
            "missing-view",
            "缺少固定角色图片",
            stage="input",
            details={"missing": missing, "expectedExtensions": SUPPORTED_EXTENSIONS},
        )
    hashes = {role.value: sha256_file(path) for role, path in found.items()}
    if len(set(hashes.values())) != len(hashes):
        fail(
            "duplicate-view",
            "三视图包含完全相同的文件",
            stage="input",
            details={"sha256": hashes},
        )
    return found


def normalize_image(source: Path, destination: Path) -> np.ndarray:
    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened)
            profile = image.info.get("icc_profile")
            if profile:
                try:
                    image = ImageCms.profileToProfile(
                        image,
                        ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                        ImageCms.createProfile("sRGB"),
                        outputMode="RGB",
                    )
                except Exception:
                    image = image.convert("RGB")
            rgb = image.convert("RGB")
            destination.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(destination, format="PNG", optimize=True)
            return np.asarray(rgb)
    except Exception as exc:
        fail(
            "image-decode-failed",
            f"无法读取图片 {source.name}: {exc}",
            stage="input",
            details={"path": str(source)},
        )


@contextmanager
def face_landmarker(model_path: Path, config: Face3DConfig) -> Iterator[Any]:
    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path),
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=2,
            min_face_detection_confidence=config.input.minimum_face_detection_confidence,
            min_face_presence_confidence=config.input.minimum_face_presence_confidence,
            min_tracking_confidence=config.input.minimum_face_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        with vision.FaceLandmarker.create_from_options(options) as detector:
            yield detector
    except Face3DError:
        raise
    except Exception as exc:
        fail(
            "landmarker-init-failed",
            f"MediaPipe Face Landmarker 初始化失败: {exc}",
            stage="input",
            details={"model": str(model_path)},
        )


def _detect(
    detector: Any, rgb: np.ndarray, role: ViewRole
) -> tuple[np.ndarray, dict[str, float], np.ndarray | None]:
    import mediapipe as mp

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = detector.detect(image)
    if len(result.face_landmarks) != 1:
        fail(
            "face-count-invalid",
            f"{role.value} 必须且只能检测到一张脸，当前检测到 {len(result.face_landmarks)} 张",
            stage="input",
            details={"role": role.value, "faceCount": len(result.face_landmarks)},
        )
    landmarks = np.asarray(
        [(point.x, point.y, point.z) for point in result.face_landmarks[0]], dtype=np.float32
    )
    blendshapes: dict[str, float] = {}
    if result.face_blendshapes:
        blendshapes = {
            category.category_name: float(category.score) for category in result.face_blendshapes[0]
        }
    transformation = None
    if result.facial_transformation_matrixes:
        candidate = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
        if candidate.shape == (4, 4) and np.all(np.isfinite(candidate)):
            transformation = candidate
    return landmarks, blendshapes, transformation


def _pose_from_transformation(transformation: np.ndarray) -> dict[str, float]:
    angles = cv2.RQDecomp3x3(np.asarray(transformation[:3, :3], dtype=np.float64))[0]
    return {
        "pitch": float(angles[0]),
        "yaw": float(-angles[1]),
        "roll": float(-angles[2]),
    }


def _eye_line_roll_degrees(landmarks: np.ndarray, width: int, height: int) -> float:
    eye_line = (
        landmarks[263, :2] - landmarks[33, :2]
    ) * np.asarray([width, height])
    angle = float(np.degrees(np.arctan2(eye_line[1], eye_line[0])))
    return float((angle + 90.0) % 180.0 - 90.0)


def _pose_from_landmarks(landmarks: np.ndarray, width: int, height: int) -> dict[str, float]:
    image_points = landmarks[np.asarray(PNP_INDICES), :2] * np.asarray([width, height])
    focal = float(max(width, height) * 1.2)
    camera = np.asarray(
        [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    ok, rotation_vector, _ = cv2.solvePnP(
        PNP_MODEL,
        image_points.astype(np.float64),
        camera,
        np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        fail("pose-estimation-failed", "无法估计头部姿态", stage="input")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    # PNP_MODEL uses +Y up and looks toward -Z, while OpenCV's camera uses
    # +Y down and looks toward +Z. A neutral face therefore contains a
    # constant 180-degree X rotation unless the model basis is removed first.
    rotation = rotation @ PNP_MODEL_TO_CAMERA
    angles = cv2.RQDecomp3x3(rotation)[0]
    pnp_roll = float(angles[2])
    return {
        "pitch": float(angles[0]),
        "yaw": float(angles[1]),
        "roll": (
            pnp_roll
            if abs(pnp_roll) <= 90.0
            else _eye_line_roll_degrees(landmarks, width, height)
        ),
    }


def _head_proposal(
    landmarks: np.ndarray,
    width: int,
    height: int,
    yaw_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return probable-head and sure-face masks for GrabCut initialization."""
    points = landmarks[np.asarray(FACE_OVAL), :2] * np.asarray([width, height])
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    face_width = max(float(x_max - x_min), 1.0)
    face_height = max(float(y_max - y_min), 1.0)
    center_x = float((x_min + x_max) / 2)
    yaw_weight = float(np.clip(abs(yaw_deg) / 45.0, 0.0, 1.0))
    # Face-oval landmarks stop near the ear. In a three-quarter view the
    # posterior bald cranium therefore lies outside a symmetric face ellipse.
    # Extend the probable-foreground proposal toward the rear; GrabCut still
    # decides the exact RGB boundary inside this permissive limit.
    center_x += float(np.sign(yaw_deg)) * face_width * 0.10 * yaw_weight

    head_top = max(0.0, float(y_min) - 0.34 * face_height)
    head_bottom = min(float(height - 1), float(y_max) + 0.08 * face_height)
    head_center_y = (head_top + head_bottom) / 2
    head_radius_x = (0.70 + 0.10 * yaw_weight) * face_width
    head_radius_y = max((head_bottom - head_top) / 2, 1.0)
    angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    ellipse = np.column_stack(
        (
            center_x + head_radius_x * np.cos(angles),
            head_center_y + head_radius_y * np.sin(angles),
        )
    )

    neck_bottom = min(float(height - 1), float(y_max) + 0.38 * face_height)
    neck = np.asarray(
        [
            (center_x - 0.30 * face_width, y_max - 0.03 * face_height),
            (center_x + 0.30 * face_width, y_max - 0.03 * face_height),
            (center_x + 0.42 * face_width, neck_bottom),
            (center_x - 0.42 * face_width, neck_bottom),
        ],
        dtype=np.float64,
    )
    proposal_points = np.vstack((ellipse, neck))
    proposal_points[:, 0] = np.clip(proposal_points[:, 0], 0, width - 1)
    proposal_points[:, 1] = np.clip(proposal_points[:, 1], 0, height - 1)

    center = points.mean(axis=0)
    sure_face = (points - center) * np.asarray([0.86, 0.90]) + center
    sure_face[:, 0] = np.clip(sure_face[:, 0], 0, width - 1)
    sure_face[:, 1] = np.clip(sure_face[:, 1], 0, height - 1)
    return (
        cv2.convexHull(proposal_points.astype(np.int32)),
        cv2.convexHull(sure_face.astype(np.int32)),
    )


def _auto_mask(
    rgb: np.ndarray,
    landmarks: np.ndarray,
    yaw_deg: float,
    mask_method: str = "grabcut",
) -> tuple[np.ndarray, float]:
    height, width = rgb.shape[:2]
    if mask_method == "white-background":
        return _white_background_mask(rgb, landmarks, yaw_deg)
    head_proposal, sure_face = _head_proposal(
        landmarks,
        width,
        height,
        yaw_deg,
    )
    grab_mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    cv2.fillConvexPoly(grab_mask, head_proposal, cv2.GC_PR_FGD)
    cv2.fillConvexPoly(grab_mask, sure_face, cv2.GC_FGD)
    border = max(2, round(min(width, height) * 0.012))
    grab_mask[:border] = cv2.GC_BGD
    grab_mask[-border:] = cv2.GC_BGD
    grab_mask[:, :border] = cv2.GC_BGD
    grab_mask[:, -border:] = cv2.GC_BGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            grab_mask,
            None,
            background_model,
            foreground_model,
            5,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        grab_mask = np.where(grab_mask == cv2.GC_FGD, cv2.GC_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        cv2.fillConvexPoly(grab_mask, sure_face, cv2.GC_FGD)
    binary = np.where((grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD), 255, 0).astype(
        np.uint8
    )
    proposal_limit = np.zeros_like(binary)
    cv2.fillConvexPoly(proposal_limit, head_proposal, 255)
    binary[proposal_limit == 0] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if components > 1:
        seed = np.rint(
            landmarks[np.asarray((1, 6, 13, 152)), :2].mean(axis=0) * np.asarray([width, height])
        ).astype(int)
        seed[0] = int(np.clip(seed[0], 0, width - 1))
        seed[1] = int(np.clip(seed[1], 0, height - 1))
        label = int(labels[seed[1], seed[0]])
        if label == 0:
            label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        binary.fill(0)
        cv2.drawContours(binary, contours, -1, 255, thickness=cv2.FILLED)
    test_points = np.round(
        landmarks[np.asarray(FACE_OVAL), :2] * np.asarray([width, height])
    ).astype(int)
    test_points[:, 0] = np.clip(test_points[:, 0], 0, width - 1)
    test_points[:, 1] = np.clip(test_points[:, 1], 0, height - 1)
    coverage = float(np.mean(binary[test_points[:, 1], test_points[:, 0]] > 0))
    return binary, coverage


def _white_background_mask(
    rgb: np.ndarray,
    landmarks: np.ndarray,
    yaw_deg: float,
) -> tuple[np.ndarray, float]:
    """Extract a clean head and neck silhouette from a controlled white backdrop."""
    height, width = rgb.shape[:2]
    oval = landmarks[np.asarray(FACE_OVAL), :2] * np.asarray([width, height])
    x_min, y_min = oval.min(axis=0)
    x_max, y_max = oval.max(axis=0)
    face_width = max(float(x_max - x_min), 1.0)
    face_height = max(float(y_max - y_min), 1.0)
    yaw_weight = float(np.clip(abs(yaw_deg) / 45.0, 0.0, 1.0))
    posterior = float(np.sign(yaw_deg))

    face_center_x = float((x_min + x_max) / 2)
    head_center_x = face_center_x + posterior * face_width * 0.10 * yaw_weight
    head_top = max(
        0.0,
        float(y_min) - (0.34 + 0.32 * yaw_weight) * face_height,
    )
    head_bottom = min(float(height - 1), float(y_max) + 0.08 * face_height)
    head_center_y = (head_top + head_bottom) / 2
    head_radius_x = (0.70 + 0.10 * yaw_weight) * face_width
    head_radius_y = max((head_bottom - head_top) / 2, 1.0)
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    head = np.column_stack(
        (
            head_center_x + head_radius_x * np.cos(angles),
            head_center_y + head_radius_y * np.sin(angles),
        )
    )

    neck_center_x = face_center_x + posterior * face_width * 0.16 * yaw_weight
    neck_top = float(y_max) - 0.03 * face_height
    neck_bottom = min(
        float(height - 1),
        float(y_max) + (0.14 + 0.18 * yaw_weight) * face_height,
    )
    neck = np.asarray(
        [
            (neck_center_x - 0.22 * face_width, neck_top),
            (neck_center_x + 0.22 * face_width, neck_top),
            (neck_center_x + 0.20 * face_width, neck_bottom),
            (neck_center_x - 0.20 * face_width, neck_bottom),
        ],
        dtype=np.float64,
    )

    anatomical_limit = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(anatomical_limit, np.rint(head).astype(np.int32), 255)
    cv2.fillConvexPoly(anatomical_limit, np.rint(neck).astype(np.int32), 255)

    corner_size = max(4, round(min(width, height) * 0.04))
    border_pixels = np.concatenate(
        (
            rgb[:corner_size, :corner_size].reshape(-1, 3),
            rgb[:corner_size, -corner_size:].reshape(-1, 3),
            rgb[-corner_size:, :corner_size].reshape(-1, 3),
            rgb[-corner_size:, -corner_size:].reshape(-1, 3),
        ),
        axis=0,
    ).astype(np.float32)
    background = np.median(border_pixels, axis=0)
    border_distance = np.max(np.abs(border_pixels - background), axis=1)
    threshold = max(6.0, float(np.percentile(border_distance, 99.5)) + 3.0)
    distance = np.max(np.abs(rgb.astype(np.float32) - background), axis=2)
    binary = np.where(distance >= threshold, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary[anatomical_limit == 0] = 0

    components, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if components > 1:
        sure_face = np.zeros_like(binary)
        center = oval.mean(axis=0)
        sure_points = (oval - center) * np.asarray([0.82, 0.88]) + center
        cv2.fillConvexPoly(sure_face, np.rint(sure_points).astype(np.int32), 255)
        overlap = np.bincount(labels[sure_face > 0], minlength=components)
        overlap[0] = 0
        label = int(np.argmax(overlap))
        if overlap[label] == 0:
            label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = np.where(labels == label, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    binary.fill(0)
    cv2.drawContours(binary, contours, -1, 255, thickness=cv2.FILLED)
    binary[anatomical_limit == 0] = 0

    test_points = np.rint(oval).astype(int)
    test_points[:, 0] = np.clip(test_points[:, 0], 0, width - 1)
    test_points[:, 1] = np.clip(test_points[:, 1], 0, height - 1)
    coverage = float(np.mean(binary[test_points[:, 1], test_points[:, 0]] > 0))
    return binary, coverage


def _draw_landmark_overlay(rgb: np.ndarray, ibug: np.ndarray, destination: Path) -> None:
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    radius = max(2, round(min(image.size) / 500))
    for index, (x, y) in enumerate(ibug):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(37, 99, 235))
        if index in (0, 8, 16, 30, 36, 45, 48, 54):
            draw.text((x + radius + 1, y - radius), str(index), fill=(255, 255, 255))
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)


def _draw_mask_overlay(rgb: np.ndarray, mask: np.ndarray, destination: Path) -> None:
    overlay = rgb.astype(np.float32)
    foreground = mask > 0
    overlay[foreground] = overlay[foreground] * 0.72 + np.asarray([37, 99, 235]) * 0.28
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bgr = cv2.cvtColor(np.clip(overlay, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr, contours, -1, (86, 255, 147), 2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), bgr)


def _mask_confirmation(run_dir: Path, masks: dict[ViewRole, Path]) -> bool:
    path = run_dir / "working" / "masks" / "confirmed.json"
    if not path.is_file():
        return False
    try:
        confirmation = read_json(path)
        hashes = confirmation.get("sha256", {})
        return all(hashes.get(role.value) == sha256_file(mask) for role, mask in masks.items())
    except Exception:
        return False


def confirm_masks(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    mask_dir = run_dir / "working" / "masks"
    masks = {role: mask_dir / f"{role.value}.png" for role in REQUIRED_VIEWS}
    missing = [str(path) for path in masks.values() if not path.is_file()]
    if missing:
        fail(
            "mask-review-required",
            "没有可确认的完整 mask 集合",
            stage="input",
            details={"missing": missing},
        )
    payload = {
        "schemaVersion": 1,
        "confirmed": True,
        "sha256": {role.value: sha256_file(path) for role, path in masks.items()},
    }
    atomic_write_json(mask_dir / "confirmed.json", payload)
    return payload


def _validate_pose(records: list[ViewRecord], config: Face3DConfig) -> None:
    by_role = {record.role: record for record in records}
    front = by_role[ViewRole.FRONT].pose_deg
    failures: list[dict[str, Any]] = []
    if abs(front["yaw"]) > config.input.front_yaw_limit_deg:
        failures.append({"role": "front", "axis": "yaw", "measured": front["yaw"]})
    for role in (ViewRole.LEFT45, ViewRole.RIGHT45):
        pose = by_role[role].pose_deg
        yaw_error = abs(abs(pose["yaw"]) - config.input.side_target_yaw_deg)
        if yaw_error > config.input.side_yaw_tolerance_deg:
            failures.append({"role": role.value, "axis": "yaw", "measured": pose["yaw"]})
        if not _yaw_matches_role(role, pose["yaw"]):
            failures.append(
                {"role": role.value, "axis": "yaw-direction", "measured": pose["yaw"]}
            )
    side_product = (
        by_role[ViewRole.LEFT45].pose_deg["yaw"] * by_role[ViewRole.RIGHT45].pose_deg["yaw"]
    )
    if side_product >= 0:
        failures.append({"roles": ["left45", "right45"], "axis": "yaw-sign"})
    for record in records:
        for axis, limit in (
            ("pitch", config.input.pitch_limit_deg),
            ("roll", config.input.roll_limit_deg),
        ):
            if abs(record.pose_deg[axis]) > limit:
                failures.append(
                    {"role": record.role.value, "axis": axis, "measured": record.pose_deg[axis]}
                )
    if failures:
        fail(
            "pose-out-of-range",
            "一张或多张照片的头部姿态不符合采集规范",
            stage="input",
            details={"failures": failures},
        )


def _identity_consistency(landmarks_by_role: dict[ViewRole, np.ndarray]) -> None:
    ratios: list[float] = []
    for landmarks in landmarks_by_role.values():
        eye_width = np.linalg.norm(landmarks[33, :3] - landmarks[263, :3])
        nose_mouth = np.linalg.norm(landmarks[1, :3] - landmarks[13, :3])
        ratios.append(float(nose_mouth / max(eye_width, 1e-6)))
    coefficient_of_variation = float(np.std(ratios) / max(np.mean(ratios), 1e-6))
    if coefficient_of_variation > 0.40:
        fail(
            "identity-inconsistent",
            "三视图的稳定面部比例明显不一致",
            stage="input",
            details={"ratios": ratios, "coefficientOfVariation": coefficient_of_variation},
        )


def _yaw_matches_role(role: ViewRole, yaw: float) -> bool:
    if role == ViewRole.FRONT:
        return True
    expected_sign = -1.0 if role == ViewRole.LEFT45 else 1.0
    return yaw * expected_sign > 0.0


def run_intake(
    input_dir: Path,
    run_dir: Path,
    config: Face3DConfig,
    *,
    stop_for_mask_review: bool = True,
) -> dict[str, Any]:
    views = discover_views(input_dir)
    run_dir = run_dir.expanduser().resolve()
    references_dir = run_dir / "references"
    landmarks_dir = run_dir / "working" / "landmarks"
    masks_dir = run_dir / "working" / "masks"
    overlays_dir = run_dir / "overlays"
    profile = (
        FaceProfileV3()
        if config.is_v3
        else (FaceProfileV2() if config.is_v2 else FaceProfileV1())
    )
    records: list[ViewRecord] = []
    raw_landmarks: dict[ViewRole, np.ndarray] = {}
    mask_paths = {role: masks_dir / f"{role.value}.png" for role in REQUIRED_VIEWS}
    masks_confirmed_before = _mask_confirmation(run_dir, mask_paths)
    landmarker_path = config.resolve_asset(config.assets.face_landmarker)
    if not landmarker_path.is_file():
        fail(
            "asset-missing",
            "缺少 MediaPipe Face Landmarker 模型",
            stage="assets",
            details={"path": str(landmarker_path)},
        )

    with face_landmarker(landmarker_path, config) as detector:
        for role in REQUIRED_VIEWS:
            source = views[role]
            normalized = references_dir / f"{role.value}.png"
            rgb = normalize_image(source, normalized)
            height, width = rgb.shape[:2]
            if max(width, height) > np.iinfo(np.uint16).max:
                fail(
                    "image-too-large",
                    f"{role.value} 超过 pixel-direct v1 的 65535px 坐标上限",
                    stage="input",
                    details={"role": role.value, "width": width, "height": height},
                )
            if min(width, height) < config.input.minimum_short_side:
                fail(
                    "image-too-small",
                    f"{role.value} 短边小于 {config.input.minimum_short_side}px",
                    stage="input",
                    details={"role": role.value, "width": width, "height": height},
                )
            landmarks, blendshapes, _ = _detect(detector, rgb, role)
            if config.uses_refined_landmarks and len(landmarks) < 478:
                fail(
                    "landmarks-incomplete",
                    f"{role.value} 缺少 Face v2 所需的虹膜地标",
                    stage="input",
                    details={"role": role.value, "measured": len(landmarks), "required": 478},
                )
            raw_landmarks[role] = landmarks
            # The pixel-direct camera equations use the same stable landmark
            # model as this PnP solution. MediaPipe's effect transformation can
            # under-estimate three-quarter yaw for non-canonical face shapes,
            # causing visually valid +/-45-degree inputs to fail Gate A.
            pose = _pose_from_landmarks(landmarks, width, height)
            grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            sharpness = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
            if sharpness < config.input.minimum_laplacian_variance:
                fail(
                    "image-blurry",
                    f"{role.value} 清晰度不足",
                    stage="input",
                    details={"role": role.value, "measured": sharpness},
                )
            face_height = max(float(landmarks[:, 1].max() - landmarks[:, 1].min()), 1e-6)
            mouth_gap = float(abs(landmarks[13, 1] - landmarks[14, 1]) / face_height)
            jaw_open = blendshapes.get("jawOpen", 0.0)
            if (
                mouth_gap > config.input.maximum_mouth_gap_ratio
                or jaw_open > config.input.maximum_jaw_open_score
            ):
                fail(
                    "expression-not-neutral",
                    f"{role.value} 不是闭嘴中性表情",
                    stage="input",
                    details={
                        "role": role.value,
                        "mouthGapRatio": mouth_gap,
                        "jawOpen": jaw_open,
                        "maximumJawOpenScore": config.input.maximum_jaw_open_score,
                    },
                )
            ibug = landmarks[np.asarray(profile.landmark_mapping), :2] * np.asarray([width, height])
            landmarks_path = landmarks_dir / f"{role.value}.npz"
            landmarks_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                landmarks_path,
                all=landmarks,
                ibug68=ibug.astype(np.float32),
                image_size=np.asarray([width, height], dtype=np.int32),
            )
            _draw_landmark_overlay(rgb, ibug, overlays_dir / f"landmarks-{role.value}.png")
            mask_path = mask_paths[role]
            if mask_path.is_file() and masks_confirmed_before:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != (height, width):
                    fail(
                        "mask-review-required",
                        f"已确认 mask 尺寸无效: {role.value}",
                        stage="input",
                    )
                oval_points = np.round(
                    landmarks[np.asarray(FACE_OVAL), :2] * np.asarray([width, height])
                ).astype(int)
                oval_points[:, 0] = np.clip(oval_points[:, 0], 0, width - 1)
                oval_points[:, 1] = np.clip(oval_points[:, 1], 0, height - 1)
                mask_coverage = float(np.mean(mask[oval_points[:, 1], oval_points[:, 0]] > 127))
            else:
                mask, mask_coverage = _auto_mask(
                    rgb,
                    landmarks,
                    pose["yaw"],
                    config.input.mask_method,
                )
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mask_path), mask)
            _draw_mask_overlay(rgb, mask, overlays_dir / f"silhouette-{role.value}.png")
            if mask_coverage < config.input.minimum_mask_face_coverage:
                fail(
                    "mask-review-required",
                    f"{role.value} 自动 mask 未覆盖稳定面部轮廓",
                    stage="input",
                    details={"role": role.value, "coverage": mask_coverage},
                )
            records.append(
                ViewRecord(
                    role=role,
                    source_path=source,
                    normalized_path=normalized,
                    width=width,
                    height=height,
                    sha256=sha256_file(source),
                    normalized_sha256=sha256_file(normalized),
                    landmarks_path=landmarks_path,
                    mask_path=mask_path,
                    pose_deg=pose,
                    sharpness=sharpness,
                    mouth_gap_ratio=mouth_gap,
                    mask_coverage=mask_coverage,
                    mask_confirmed=masks_confirmed_before,
                )
            )

    _validate_pose(records, config)
    _identity_consistency(raw_landmarks)
    payload = {
        "schemaVersion": 1,
        "contractVersion": config.schema_version,
        "profile": profile.id,
        "views": [record.model_dump(mode="json") for record in records],
        "maskConfirmed": masks_confirmed_before,
    }
    intake_path = run_dir / "working" / "intake.json"
    atomic_write_json(intake_path, payload)
    if (
        config.input.require_mask_confirmation
        and not masks_confirmed_before
        and stop_for_mask_review
    ):
        fail(
            "mask-review-required",
            "自动 mask 已生成；检查/修正后运行 confirm-masks",
            stage="input",
            details={
                "run": str(run_dir),
                "masks": {role.value: str(path) for role, path in mask_paths.items()},
            },
        )
    return payload


def validate_only(
    input_dir: Path,
    config: Face3DConfig,
    output: Path | None = None,
) -> dict[str, Any]:
    if output is not None:
        return run_intake(input_dir, output, config, stop_for_mask_review=False)
    with tempfile.TemporaryDirectory(prefix="face3d-validate-") as temporary:
        result = run_intake(input_dir, Path(temporary), config, stop_for_mask_review=False)
        result["maskConfirmed"] = False
        result["note"] = "验证使用临时自动 mask；重建运行仍需确认正式 mask"
        return json.loads(json.dumps(result))
