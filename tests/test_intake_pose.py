import math

import cv2
import numpy as np
import pytest

from face3d.models import ViewRole
from face3d.stages.intake import (
    FACE_OVAL,
    PNP_INDICES,
    PNP_MODEL,
    PNP_MODEL_TO_CAMERA,
    _auto_mask,
    _eye_line_roll_degrees,
    _pose_from_landmarks,
    _pose_from_transformation,
    _yaw_matches_role,
)


def test_white_background_mask_keeps_cranium_and_excludes_collar() -> None:
    size = 256
    rgb = np.full((size, size, 3), 255, dtype=np.uint8)
    skin = (252, 218, 205)
    cv2.ellipse(rgb, (128, 86), (68, 72), 0, 0, 360, skin, thickness=cv2.FILLED)
    cv2.rectangle(rgb, (108, 145), (148, 210), skin, thickness=cv2.FILLED)
    cv2.fillConvexPoly(
        rgb,
        np.asarray(
            (
                (45, 205),
                (105, 188),
                (128, 220),
                (151, 188),
                (211, 205),
                (225, 255),
                (31, 255),
            )
        ),
        (205, 210, 220),
    )

    landmarks = np.zeros((478, 3), dtype=np.float64)
    angles = np.linspace(0, 2 * np.pi, len(FACE_OVAL), endpoint=False)
    landmarks[np.asarray(FACE_OVAL), 0] = (128 + 50 * np.cos(angles)) / size
    landmarks[np.asarray(FACE_OVAL), 1] = (105 + 55 * np.sin(angles)) / size

    mask, coverage = _auto_mask(
        rgb,
        landmarks,
        yaw_deg=35.0,
        mask_method="white-background",
    )

    assert mask[18, 128] == 255
    assert mask[182, 128] == 255
    assert mask[215, 80] == 0
    assert coverage == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("end", "expected"),
    [((0.8, 0.5), 0.0), ((0.8, 0.6), 9.4623222), ((-0.4, 0.5), 0.0)],
)
def test_eye_line_roll_uses_undirected_axis(
    end: tuple[float, float], expected: float
) -> None:
    landmarks = np.zeros((468, 3), dtype=np.float64)
    landmarks[33, :2] = (0.2, 0.5)
    landmarks[263, :2] = end

    roll = _eye_line_roll_degrees(landmarks, 1200, 1200)

    assert abs(roll) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    ("role", "yaw", "expected"),
    [
        (ViewRole.FRONT, 0.0, True),
        (ViewRole.LEFT45, -35.0, True),
        (ViewRole.LEFT45, 35.0, False),
        (ViewRole.RIGHT45, 35.0, True),
        (ViewRole.RIGHT45, -35.0, False),
    ],
)
def test_yaw_direction_matches_semantic_role(
    role: ViewRole, yaw: float, expected: bool
) -> None:
    assert _yaw_matches_role(role, yaw) is expected


@pytest.mark.parametrize("yaw", [0.0, -45.0, 45.0])
def test_pose_removes_pnp_model_basis_rotation(yaw: float) -> None:
    width = height = 1200
    focal = max(width, height) * 1.2
    camera = np.asarray(
        [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    theta = math.radians(yaw)
    expected = np.asarray(
        [
            [math.cos(theta), 0.0, math.sin(theta)],
            [0.0, 1.0, 0.0],
            [-math.sin(theta), 0.0, math.cos(theta)],
        ],
        dtype=np.float64,
    )
    raw_rotation = expected @ PNP_MODEL_TO_CAMERA
    rotation_vector, _ = cv2.Rodrigues(raw_rotation)
    image_points, _ = cv2.projectPoints(
        PNP_MODEL,
        rotation_vector,
        np.asarray([0.0, 0.0, 700.0]),
        camera,
        np.zeros((4, 1)),
    )
    landmarks = np.zeros((468, 3), dtype=np.float64)
    landmarks[np.asarray(PNP_INDICES), :2] = image_points.reshape(-1, 2) / np.asarray(
        [width, height]
    )

    pose = _pose_from_landmarks(landmarks, width, height)

    assert pose["pitch"] == pytest.approx(0.0, abs=0.1)
    assert pose["yaw"] == pytest.approx(yaw, abs=0.1)
    assert pose["roll"] == pytest.approx(0.0, abs=0.1)


@pytest.mark.parametrize("yaw", [0.0, -45.0, 45.0])
def test_mediapipe_transformation_uses_project_yaw_convention(yaw: float) -> None:
    theta = math.radians(-yaw)
    transformation = np.eye(4, dtype=np.float64)
    transformation[:3, :3] = np.asarray(
        [
            [math.cos(theta), 0.0, math.sin(theta)],
            [0.0, 1.0, 0.0],
            [-math.sin(theta), 0.0, math.cos(theta)],
        ]
    )

    pose = _pose_from_transformation(transformation)

    assert pose["pitch"] == pytest.approx(0.0, abs=0.1)
    assert pose["yaw"] == pytest.approx(yaw, abs=0.1)
    assert pose["roll"] == pytest.approx(0.0, abs=0.1)
