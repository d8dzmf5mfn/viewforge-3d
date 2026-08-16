from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssetKind(StrEnum):
    MODEL = "model"
    BLEND = "blend"
    CONFIG = "config"
    JSON = "json"
    IMAGE = "image"
    VIDEO = "video"


class JobKind(StrEnum):
    BUILD_SKELETON = "build_skeleton"
    CREATE_BONE_ANIMATION = "create_bone_animation"
    BIND_RIGID_COMPONENTS = "bind_rigid_components"
    BUILD_DECLARATIVE_MODEL = "build_declarative_model"
    RENDER_MODEL_PREVIEW = "render_model_preview"
    GENERATE_PIXEL_CUBE = "generate_pixel_cube"
    RECONSTRUCT_SIX_VIEW_VISUAL_HULL = "reconstruct_six_view_visual_hull"
    VALIDATE_FACE_MULTIVIEW = "validate_face_multiview"
    RECONSTRUCT_FACE_MULTIVIEW = "reconstruct_face_multiview"
    CONTINUE_FACE_RECONSTRUCTION = "continue_face_reconstruction"
    PACKAGE_FACE_RECONSTRUCTION = "package_face_reconstruction"


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    REVIEW_REQUIRED = "review_required"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AssetSummary(PublicModel):
    id: str
    name: str
    kind: AssetKind
    extension: str
    size_bytes: int = Field(ge=0)
    sha256: str


class ArtifactSummary(PublicModel):
    id: str
    job_id: str
    name: str
    extension: str
    size_bytes: int = Field(ge=0)
    sha256: str


class JobSummary(PublicModel):
    id: str
    kind: JobKind
    state: JobState
    created_at: str
    updated_at: str
    artifacts: list[ArtifactSummary] = Field(default_factory=list)
    status_message: str | None = None
    failure: str | None = None


class LocalStatus(PublicModel):
    ready: bool
    server_version: str
    workspace_configured: bool
    blender_available: bool
    plugin_runtime_available: bool
    modeling_runtime_available: bool
    blender_tools_available: bool
    capabilities: list[str]
    active_jobs: int = Field(ge=0)
    asset_count: int = Field(ge=0)
    endpoint: str


class AssetList(PublicModel):
    assets: list[AssetSummary]


class JobList(PublicModel):
    jobs: list[JobSummary]


class ArtifactList(PublicModel):
    job_id: str
    artifacts: list[ArtifactSummary]


class JSONArtifact(PublicModel):
    artifact: ArtifactSummary
    document: dict[str, Any]


class ModelingAssetState(PublicModel):
    name: str
    required: bool
    exists: bool
    hash_matches: bool
    license: str


class ModelingProfileStatus(PublicModel):
    profile: str
    schema_version: int
    mode: str
    required_views: list[str]
    assets_ready: bool
    assets: list[ModelingAssetState]


class DeclarativeMaterial(PublicModel):
    color: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    metallic: float = 0.0
    roughness: float = 0.5


class DeclarativeBevel(PublicModel):
    width: float = 0.01
    segments: int = 2


class DeclarativeWorld(PublicModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    background_color: tuple[float, float, float, float] = Field(
        default=(0.05, 0.05, 0.05, 1.0),
        alias="backgroundColor",
    )


class DeclarativeModelObject(PublicModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    primitive: Literal[
        "cube",
        "uv_sphere",
        "ico_sphere",
        "cylinder",
        "cone",
        "torus",
        "mesh",
    ]
    parameters: dict[str, float | int] | None = None
    vertices: list[tuple[float, float, float]] | None = None
    faces: list[list[int]] | None = None
    location: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_degrees: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        alias="rotationDegrees",
    )
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    material: DeclarativeMaterial | None = None
    bevel: DeclarativeBevel | None = None
    shade_smooth: bool = Field(default=False, alias="shadeSmooth")


class DeclarativeModelSpec(PublicModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal[1] = Field(default=1, alias="schemaVersion")
    name: str | None = None
    units: Literal["meters"] = "meters"
    world: DeclarativeWorld = Field(default_factory=DeclarativeWorld)
    objects: list[DeclarativeModelObject] = Field(min_length=1, max_length=256)


SkeletonProfile = Literal["humanoid-v1", "quadruped-v1"]
RenderView = Literal[
    "perspective",
    "front",
    "back",
    "left",
    "right",
    "top",
    "bottom",
]
RenderMaterialMode = Literal["original", "neutral"]
RenderBackground = Literal["studio_dark", "studio_light", "transparent"]
