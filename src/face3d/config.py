from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from face3d.errors import Face3DError, fail


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AssetConfig(StrictModel):
    flame_model: Path | None = None
    flame_landmarks: Path | None = None
    flame_masks: Path | None = None
    flame_prepared: Path | None = None
    template_head: Path | None = None
    template_landmarks: Path | None = None
    template_manifest: Path | None = None
    face_landmarker: Path
    canonical_face_model: Path


class InputConfig(StrictModel):
    minimum_short_side: int = Field(ge=256)
    front_yaw_limit_deg: float = Field(gt=0)
    side_target_yaw_deg: float = Field(gt=0, le=90)
    side_yaw_tolerance_deg: float = Field(gt=0)
    pitch_limit_deg: float = Field(gt=0)
    roll_limit_deg: float = Field(gt=0)
    minimum_laplacian_variance: float = Field(gt=0)
    maximum_mouth_gap_ratio: float = Field(gt=0, lt=0.5)
    maximum_jaw_open_score: float = Field(default=0.20, ge=0, le=1)
    minimum_face_detection_confidence: float = Field(default=0.65, gt=0, le=1)
    minimum_face_presence_confidence: float = Field(default=0.65, gt=0, le=1)
    minimum_face_tracking_confidence: float = Field(default=0.65, gt=0, le=1)
    mask_method: Literal["grabcut", "white-background"] = "grabcut"
    minimum_mask_face_coverage: float = Field(gt=0, le=1)
    require_mask_confirmation: bool = True


class PixelConfig(StrictModel):
    grid_size: int = Field(ge=64, le=512)
    coarse_depth_grid: int = Field(ge=8, le=128)
    complex_region_radius_pixels: float = Field(gt=0)
    maximum_cells: int = Field(gt=0)
    depth_scale_face_width: float = Field(gt=0, le=1)
    cell_fill_ratio: float = Field(gt=0, le=1)
    base_thickness_pixels: float = Field(gt=0, le=4)

    @model_validator(mode="after")
    def validate_coarse_grid(self) -> PixelConfig:
        if self.coarse_depth_grid >= self.grid_size:
            raise ValueError("coarse_depth_grid must be smaller than grid_size")
        return self


class FitConfig(StrictModel):
    shape_coefficients: int = Field(ge=1, le=300)
    shared_focal_length: bool = True
    adam_iterations: int = Field(ge=0)
    lbfgs_iterations: int = Field(ge=0)
    learning_rate: float = Field(gt=0)
    landmark_weight: float = Field(ge=0)
    contour_weight: float = Field(ge=0)
    shape_prior_weight: float = Field(ge=0)
    symmetry_weight: float = Field(ge=0)
    dense_landmark_weight: float = Field(default=0.55, ge=0)
    relative_depth_weight: float = Field(default=0.025, ge=0)
    local_offset_weight: float = Field(default=0.04, ge=0)
    laplacian_weight: float = Field(default=0.08, ge=0)
    arap_weight: float = Field(default=0.10, ge=0)
    maximum_normal_offset_face_width: float = Field(default=0.015, gt=0, le=0.05)
    low_frequency_basis_size: int = Field(default=64, ge=8, le=256)
    maximum_vertex_displacement_face_width: float = Field(default=0.18, gt=0, le=0.5)
    inversion_barrier_weight: float = Field(default=0.25, ge=0)
    ear_constraint_weight: float = Field(default=0.35, ge=0)
    eyelid_constraint_weight: float = Field(default=0.40, ge=0)


class SDFConfig(StrictModel):
    resolution: int = Field(ge=32, le=512)
    padding_fraction: float = Field(gt=0, lt=0.5)
    surface_band_voxels: float = Field(gt=0)
    maximum_instances: int = Field(gt=0)
    query_chunk_points: int = Field(ge=1000)


class MeshConfig(StrictModel):
    target_triangles: int = Field(gt=0)
    minimum_triangles: int = Field(gt=0)
    maximum_triangles: int = Field(gt=0)
    taubin_iterations: int = Field(ge=0)
    taubin_lambda: float
    taubin_nu: float
    feature_protection_radius_voxels: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_triangle_budget(self) -> MeshConfig:
        if not self.minimum_triangles <= self.target_triangles <= self.maximum_triangles:
            raise ValueError("target_triangles must be within minimum/maximum")
        return self


class SkinConfig(StrictModel):
    atlas_resolution: int = Field(ge=256, le=4096)
    detail_resolution: int = Field(ge=128, le=2048)
    uv_albedo_source: Path
    micro_albedo_source: Path
    minimum_observed_vertex_fraction: float = Field(gt=0, le=1)
    jpeg_quality: int = Field(ge=75, le=100)
    uv_method: Literal["legacy-centered", "xatlas"] = "legacy-centered"
    local_warp_fraction_max: float = Field(default=0.005, ge=0, le=0.02)
    eye_atlas_resolution: int = Field(default=512, ge=128, le=2048)


class OutputConfig(StrictModel):
    geometry_only: bool = False


class AnatomyConfig(StrictModel):
    subdivision_levels: int = Field(default=2, ge=0, le=3)
    eye_radius_min_scale: float = Field(default=0.90, gt=0)
    eye_radius_max_scale: float = Field(default=1.10, gt=0)
    eye_radius_symmetry_max: float = Field(default=0.03, ge=0, le=0.25)
    eyelid_clearance_ratio_max: float = Field(default=0.03, gt=0, le=0.20)
    ear_silhouette_iou_min: float = Field(default=0.85, ge=0, le=1)

    @model_validator(mode="after")
    def validate_eye_radius_range(self) -> AnatomyConfig:
        if self.eye_radius_min_scale > self.eye_radius_max_scale:
            raise ValueError("eye_radius_min_scale must not exceed eye_radius_max_scale")
        return self


class AcceptanceConfig(StrictModel):
    front_landmark_nme_max: float = Field(gt=0)
    side_landmark_nme_max: float = Field(gt=0)
    front_silhouette_iou_min: float = Field(gt=0, le=1)
    side_silhouette_iou_min: float = Field(gt=0, le=1)
    silhouette_iou_drop_max: float = Field(ge=0, le=1)
    feature_drift_voxels_max: float = Field(gt=0)
    normal_variance_reduction_min: float = Field(ge=0, le=1)
    hausdorff_voxels_max: float = Field(gt=0)
    package_size_mb_max: float = Field(gt=0)
    runtime_minutes_max: float = Field(gt=0)
    peak_memory_gb_max: float = Field(gt=0)
    front_landmark_nme_v2_max: float = Field(default=0.015, gt=0)
    side_landmark_nme_v2_max: float = Field(default=0.020, gt=0)
    front_silhouette_iou_v2_min: float = Field(default=0.95, gt=0, le=1)
    side_silhouette_iou_v2_min: float = Field(default=0.92, gt=0, le=1)
    texture_face_error_px_max: float = Field(default=2.0, gt=0)
    texture_ear_error_px_max: float = Field(default=4.0, gt=0)
    maximum_surface_distance_voxels: float = Field(default=0.75, gt=0)


class Face3DConfig(StrictModel):
    schema_version: Literal[1, 2, 3]
    profile: Literal["face-v1", "face-v2", "face-v3"]
    mode: Literal["pixel-direct", "pixel-flame-hybrid", "template-head-v0"]
    deterministic: bool = True
    seed: int
    assets: AssetConfig
    input: InputConfig
    pixel: PixelConfig
    fit: FitConfig
    sdf: SDFConfig
    mesh: MeshConfig
    skin: SkinConfig
    output: OutputConfig = OutputConfig()
    anatomy: AnatomyConfig = AnatomyConfig()
    acceptance: AcceptanceConfig
    source_path: Path = Field(exclude=True)
    project_root: Path = Field(exclude=True)

    def resolve_asset(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.project_root / path).resolve()

    def resolve_optional_asset(self, path: Path | None) -> Path | None:
        return None if path is None else self.resolve_asset(path)

    @property
    def is_v2(self) -> bool:
        return self.schema_version == 2

    @property
    def is_v3(self) -> bool:
        return self.schema_version == 3

    @property
    def uses_refined_landmarks(self) -> bool:
        return self.schema_version >= 2

    @model_validator(mode="after")
    def validate_version_contract(self) -> Face3DConfig:
        expected = {
            1: ("face-v1", "pixel-direct"),
            2: ("face-v2", "pixel-flame-hybrid"),
            3: ("face-v3", "template-head-v0"),
        }[self.schema_version]
        if (self.profile, self.mode) != expected:
            raise ValueError(
                "schema/profile/mode must be 1/face-v1/pixel-direct or "
                "2/face-v2/pixel-flame-hybrid or 3/face-v3/template-head-v0"
            )
        if self.schema_version <= 2 and (
            self.assets.flame_model is None or self.assets.flame_landmarks is None
        ):
            raise ValueError("face-v1 and face-v2 require flame_model and flame_landmarks")
        if self.is_v2 and (self.assets.flame_masks is None or self.assets.flame_prepared is None):
            raise ValueError("face-v2 requires flame_masks and flame_prepared assets")
        if self.is_v3 and (
            self.assets.template_head is None
            or self.assets.template_landmarks is None
            or self.assets.template_manifest is None
        ):
            raise ValueError(
                "face-v3 requires template_head, template_landmarks and template_manifest"
            )
        return self


def load_config(path: Path) -> Face3DConfig:
    path = path.expanduser().resolve()
    if not path.is_file():
        fail("config-missing", f"配置文件不存在: {path}", stage="config")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("root must be a mapping")
        project_root = path.parent.parent if path.parent.name == "configs" else path.parent
        return Face3DConfig.model_validate(
            {**payload, "source_path": path, "project_root": project_root.resolve()}
        )
    except Face3DError:
        raise
    except Exception as exc:
        fail(
            "config-invalid",
            f"配置文件无效: {exc}",
            stage="config",
            details={"path": str(path)},
        )
