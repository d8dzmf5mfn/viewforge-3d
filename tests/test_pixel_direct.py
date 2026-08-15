import math

import numpy as np

from face3d.models import ViewRole
from face3d.stages.pixel_direct import estimate_depth_anchors


def _synthetic_landmarks() -> tuple[dict[ViewRole, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(42)
    count = 468
    x = rng.uniform(-0.34, 0.34, count)
    y = rng.uniform(-0.46, 0.46, count)
    depth = 0.04 + 0.20 * np.exp(-((x / 0.13) ** 2 + (y / 0.24) ** 2))
    x[234], x[454] = -0.34, 0.34
    y[10], y[152] = -0.5, 0.5
    x[1], y[1], depth[1] = 0.0, 0.0, 0.30
    for index in (93, 132, 234, 323, 361, 454):
        depth[index] = 0.02
    result: dict[ViewRole, np.ndarray] = {}
    for role, yaw in (
        (ViewRole.FRONT, 0.0),
        (ViewRole.LEFT45, -45.0),
        (ViewRole.RIGHT45, 45.0),
    ):
        theta = -math.radians(yaw)
        projected_x = math.cos(theta) * x + math.sin(theta) * depth
        landmarks = np.zeros((count, 3), dtype=np.float64)
        landmarks[:, 0] = 0.5 + projected_x * 0.6
        landmarks[:, 1] = 0.5 + y * 0.6
        result[role] = landmarks
    return result, depth


def test_depth_anchors_recover_multiview_shape() -> None:
    landmarks, expected = _synthetic_landmarks()
    estimate = estimate_depth_anchors(
        landmarks,
        {role: (1000, 1000) for role in ViewRole},
        {ViewRole.FRONT: 0.0, ViewRole.LEFT45: -45.0, ViewRole.RIGHT45: 45.0},
        0.38,
    )
    correlation = np.corrcoef(estimate.raw_anchor_depth, expected)[0, 1]
    assert correlation > 0.98
    assert estimate.world_anchor_depth[1] > np.median(
        estimate.world_anchor_depth[[93, 132, 234, 323, 361, 454]]
    )
    assert float(estimate.agreement.mean()) > 0.95
