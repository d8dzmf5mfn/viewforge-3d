from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ViewRole(StrEnum):
    FRONT = "front"
    LEFT45 = "left45"
    RIGHT45 = "right45"


REQUIRED_VIEWS = (ViewRole.FRONT, ViewRole.LEFT45, ViewRole.RIGHT45)


class CameraRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: ViewRole
    width: int
    height: int
    focal_length_px: float
    principal_point_px: tuple[float, float]
    rotation_vector: tuple[float, float, float]
    translation: tuple[float, float, float]
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None


class ViewRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: ViewRole
    source_path: Path
    normalized_path: Path
    width: int
    height: int
    sha256: str
    normalized_sha256: str
    landmarks_path: Path
    mask_path: Path
    pose_deg: dict[str, float]
    sharpness: float
    mouth_gap_ratio: float
    mask_coverage: float
    mask_confirmed: bool = False
    warnings: list[str] = Field(default_factory=list)


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gate: str
    status: str
    measured: float | int | bool | str | None = None
    threshold: float | int | bool | str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
