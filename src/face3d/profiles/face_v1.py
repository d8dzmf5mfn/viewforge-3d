from __future__ import annotations

from collections.abc import Mapping, Sequence

from face3d.models import REQUIRED_VIEWS, ViewRole
from face3d.profiles.base import SubjectProfile

# MediaPipe Face Landmarker indices ordered to match the conventional FLAME/ibug 68-point layout.
# Contour samples are deliberately denser around the ears and jaw than a six-point PnP subset.
MEDIAPIPE_TO_IBUG68: tuple[int, ...] = (
    127,
    234,
    93,
    132,
    58,
    172,
    136,
    150,
    152,
    377,
    400,
    378,
    365,
    397,
    323,
    454,
    356,
    70,
    63,
    105,
    66,
    107,
    336,
    296,
    334,
    293,
    300,
    168,
    6,
    197,
    195,
    129,
    98,
    2,
    327,
    358,
    33,
    160,
    158,
    133,
    153,
    144,
    362,
    385,
    387,
    263,
    373,
    380,
    61,
    40,
    37,
    0,
    267,
    270,
    291,
    321,
    314,
    17,
    84,
    91,
    78,
    82,
    13,
    312,
    308,
    317,
    14,
    87,
)


class FaceProfileV1(SubjectProfile):
    id = "face-v1"

    @property
    def required_views(self) -> Sequence[ViewRole]:
        return REQUIRED_VIEWS

    @property
    def landmark_mapping(self) -> Sequence[int]:
        return MEDIAPIPE_TO_IBUG68

    @property
    def feature_landmark_groups(self) -> Mapping[str, Sequence[int]]:
        return {
            "jaw": range(0, 17),
            "eyebrows": range(17, 27),
            "nose": range(27, 36),
            "eyes": range(36, 48),
            "mouth": range(48, 68),
        }

    def expected_yaw(self, role: ViewRole) -> float:
        return {
            ViewRole.FRONT: 0.0,
            ViewRole.LEFT45: -45.0,
            ViewRole.RIGHT45: 45.0,
        }[role]
