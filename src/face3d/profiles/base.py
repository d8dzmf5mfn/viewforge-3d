from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from face3d.models import ViewRole


class SubjectProfile(ABC):
    id: str

    @property
    @abstractmethod
    def required_views(self) -> Sequence[ViewRole]: ...

    @property
    @abstractmethod
    def landmark_mapping(self) -> Sequence[int]: ...

    @property
    @abstractmethod
    def feature_landmark_groups(self) -> Mapping[str, Sequence[int]]: ...

    @abstractmethod
    def expected_yaw(self, role: ViewRole) -> float: ...
