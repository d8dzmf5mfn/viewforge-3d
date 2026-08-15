from __future__ import annotations

from collections.abc import Mapping, Sequence

from face3d.models import ViewRole
from face3d.profiles.face_v1 import FaceProfileV1


class FaceProfileV2(FaceProfileV1):
    """Quality-first face profile with explicit eye and ear contracts."""

    id = "face-v2"

    @property
    def iris_landmark_groups(self) -> Mapping[str, Sequence[int]]:
        # MediaPipe Face Landmarker returns 468-472 for the left iris and
        # 473-477 for the right iris when the bundled task supports refinement.
        return {"left": range(468, 473), "right": range(473, 478)}

    @property
    def eyelid_landmark_groups(self) -> Mapping[str, Sequence[int]]:
        return {
            "left": (33, 160, 158, 133, 153, 144),
            "right": (362, 385, 387, 263, 373, 380),
        }

    def expected_yaw(self, role: ViewRole) -> float:
        return super().expected_yaw(role)
