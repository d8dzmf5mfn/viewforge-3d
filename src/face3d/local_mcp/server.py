from __future__ import annotations

import argparse
import base64
import contextlib
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, ImageContent, TextContent, ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse

from .jobs import JobLauncher
from .models import (
    ArtifactList,
    AssetList,
    AssetSummary,
    DeclarativeModelSpec,
    JobList,
    JobSummary,
    JSONArtifact,
    LocalStatus,
    ModelingAssetState,
    ModelingProfileStatus,
    RenderBackground,
    RenderMaterialMode,
    RenderView,
    SkeletonProfile,
)
from .storage import (
    SERVER_VERSION,
    ArtifactStore,
    AssetStore,
    ConfigurationStore,
    JobStore,
    LocalPaths,
)

READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
LOCAL_WRITE = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)
LOCAL_IDEMPOTENT_WRITE = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
IMAGE_ARTIFACT_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
MAX_IMAGE_ARTIFACT_BYTES = 10 * 1024 * 1024


class Runtime:
    def __init__(
        self,
        paths: LocalPaths | None = None,
        endpoint_host: str | None = None,
        endpoint_port: int | None = None,
    ) -> None:
        self.paths = paths or LocalPaths()
        self.endpoint_host = endpoint_host
        self.endpoint_port = endpoint_port
        self.paths.ensure()
        self.configuration = ConfigurationStore(self.paths)
        self.assets = AssetStore(self.paths, self.configuration)
        self.artifacts = ArtifactStore(self.paths)
        self.jobs = JobStore(self.paths, self.artifacts)
        self.launcher = JobLauncher(
            self.paths,
            self.configuration,
            self.assets,
            self.artifacts,
            self.jobs,
        )

    def status(self) -> LocalStatus:
        configuration = self.configuration.load()
        workspace = self.configuration.workspace()
        blender = self.configuration.blender()
        plugin_root = self.configuration.plugin_root()
        modeling_runtime_available = self.configuration.modeling_runtime_available()
        blender_tools_available = blender is not None and plugin_root is not None
        ready = workspace is not None and (modeling_runtime_available or blender_tools_available)
        capabilities = ["asset_registry", "job_artifacts"] if workspace is not None else []
        if modeling_runtime_available:
            capabilities.extend(
                [
                    "pixel_cube",
                    "six_view_visual_hull",
                    "face_multiview_reconstruction",
                ]
            )
        if blender_tools_available:
            capabilities.extend(
                [
                    "declarative_blender_modeling",
                    "model_rendering",
                    "biological_skeleton",
                    "bone_animation",
                    "rigid_binding",
                ]
            )
        host = self.endpoint_host or configuration.mcp_host
        port = self.endpoint_port or configuration.mcp_port
        return LocalStatus(
            ready=ready,
            server_version=SERVER_VERSION,
            workspace_configured=workspace is not None,
            blender_available=blender is not None,
            plugin_runtime_available=plugin_root is not None,
            modeling_runtime_available=modeling_runtime_available,
            blender_tools_available=blender_tools_available,
            capabilities=capabilities,
            active_jobs=self.jobs.active_count(),
            asset_count=len(self.assets.list()),
            endpoint=f"http://{host}:{port}/mcp",
        )

    def resolve_json_artifact(self, artifact_id: str) -> JSONArtifact:
        artifact = self.artifacts.get(artifact_id)
        path = self.artifacts.resolve(artifact_id)
        if path.suffix.lower() != ".json":
            raise ValueError("Only JSON artifacts can be read through this tool.")
        if path.stat().st_size > 256 * 1024:
            raise ValueError("The JSON artifact exceeds the local read limit.")
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("The artifact is not a JSON object.")
        roots = [
            self.paths.root,
            self.configuration.workspace(),
            self.configuration.plugin_root(),
        ]
        return JSONArtifact(
            artifact=artifact,
            document=_sanitize_document(payload, [root for root in roots if root is not None]),
        )

    def resolve_image_artifact(self, artifact_id: str) -> CallToolResult:
        artifact = self.artifacts.get(artifact_id)
        path = self.artifacts.resolve(artifact_id)
        mime_type = IMAGE_ARTIFACT_MIME_TYPES.get(path.suffix.lower())
        if mime_type is None:
            raise ValueError("Only PNG, JPEG, or WebP artifacts can be returned as images.")
        if path.stat().st_size > MAX_IMAGE_ARTIFACT_BYTES:
            raise ValueError("The image artifact exceeds the local read limit.")
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        f"{artifact.name} · {artifact.size_bytes} bytes · "
                        f"sha256:{artifact.sha256}"
                    ),
                ),
                ImageContent(
                    type="image",
                    data=base64.b64encode(path.read_bytes()).decode("ascii"),
                    mime_type=mime_type,
                ),
            ]
        )

    def inspect_modeling_profile(self, config_asset_id: str) -> ModelingProfileStatus:
        if not self.configuration.modeling_runtime_available():
            raise RuntimeError("The ViewForge modeling runtime is unavailable.")
        from face3d.assets import asset_status
        config_path = self.launcher.references.resolve(config_asset_id, {".yaml", ".yml"})
        workspace = self.configuration.workspace()
        if workspace is None:
            raise RuntimeError("Configure a local workspace before inspecting a modeling profile.")
        from .jobs import load_workspace_modeling_config

        config = load_workspace_modeling_config(config_path, workspace)
        status = asset_status(config, require_recorded=True)
        required = set(status["coreAssets"])
        return ModelingProfileStatus(
            profile=config.profile,
            schema_version=config.schema_version,
            mode=config.mode,
            required_views=["front", "left45", "right45"],
            assets_ready=bool(status["ready"]),
            assets=[
                ModelingAssetState(
                    name=name,
                    required=name in required,
                    exists=bool(details["exists"]),
                    hash_matches=bool(details["hashMatches"]),
                    license=str(details["license"]),
                )
                for name, details in sorted(status["models"].items())
            ],
        )


def _sanitize_document(value: Any, roots: list[Path]) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_document(item, roots) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_document(item, roots) for item in value]
    if not isinstance(value, str):
        return value
    sanitized = value
    for root in roots:
        sanitized = sanitized.replace(str(root), "[local]")
    if sanitized.startswith("/"):
        return f"[local]/{Path(sanitized).name}"
    return sanitized


def create_server(runtime: Runtime | None = None) -> MCPServer:
    local = runtime or Runtime()
    server = MCPServer(
        name="viewforge-local",
        title="ViewForge Local",
        description=(
            "Local-only multiview reconstruction, declarative modeling, model rendering, "
            "biological skeleton, animation, and rigid-binding tools."
        ),
        version=SERVER_VERSION,
        instructions=(
            "Use asset and artifact IDs instead of local paths after registration. "
            "Geometry actions create immutable job outputs and never mutate the source asset. "
            "Arbitrary Python and Blender script execution is intentionally unavailable. "
            "Call get_viewforge_job until a queued or running job reaches succeeded, failed, or "
            "review_required. After a render job succeeds, call read_image_artifact with the "
            "render-preview-sheet.png artifact ID to inspect the result."
        ),
    )

    @server.custom_route("/healthz", methods=["GET"], include_in_schema=False)
    async def healthz(_: Request) -> JSONResponse:
        status = local.status()
        return JSONResponse(
            {
                "status": "ready" if status.ready else "needs-configuration",
                "serverVersion": status.server_version,
                "workspaceConfigured": status.workspace_configured,
                "blenderAvailable": status.blender_available,
                "pluginRuntimeAvailable": status.plugin_runtime_available,
                "modelingRuntimeAvailable": status.modeling_runtime_available,
                "blenderToolsAvailable": status.blender_tools_available,
                "capabilities": status.capabilities,
            }
        )

    @server.tool(
        name="viewforge_status",
        title="Check ViewForge Local",
        description="Use this when you need to verify that the local ViewForge runtime is ready.",
        annotations=READ_ONLY,
        structured_output=True,
    )
    def viewforge_status() -> LocalStatus:
        return local.status()

    @server.tool(
        name="register_local_asset",
        title="Register local asset",
        description=(
            "Use this when a user-approved model, Blend, YAML config, JSON, image, or video "
            "inside the configured workspace must be referenced by a private asset ID."
        ),
        annotations=LOCAL_IDEMPOTENT_WRITE,
        structured_output=True,
    )
    def register_local_asset(path: str) -> AssetSummary:
        return local.assets.register(path)

    @server.tool(
        name="list_local_assets",
        title="List local assets",
        description="Use this when you need the private asset IDs already registered locally.",
        annotations=READ_ONLY,
        structured_output=True,
    )
    def list_local_assets() -> AssetList:
        return AssetList(assets=local.assets.list())

    @server.tool(
        name="build_biological_skeleton",
        title="Build biological skeleton",
        description=(
            "Use this when an approved Blend, GLB, or glTF source needs an immutable bone-only "
            "humanoid or quadruped Armature with QA artifacts. source_id accepts either an "
            "asset_ ID or a generated artifact_ ID. Inline landmarks and component maps are "
            "accepted when no registered JSON reference exists."
        ),
        annotations=LOCAL_WRITE,
        structured_output=True,
    )
    def build_biological_skeleton(
        source_id: str,
        profile: SkeletonProfile = "humanoid-v1",
        landmarks_asset_id: str | None = None,
        component_map_asset_id: str | None = None,
        front_annotation_asset_id: str | None = None,
        side_annotation_asset_id: str | None = None,
        landmarks: dict[str, Any] | None = None,
        component_map: dict[str, Any] | None = None,
    ) -> JobSummary:
        return local.launcher.build_skeleton(
            source_id=source_id,
            profile=profile,
            landmarks_asset_id=landmarks_asset_id,
            component_map_asset_id=component_map_asset_id,
            front_annotation_asset_id=front_annotation_asset_id,
            side_annotation_asset_id=side_annotation_asset_id,
            landmarks=landmarks,
            component_map=component_map,
        )

    @server.tool(
        name="create_bone_animation",
        title="Create bone animation",
        description=(
            "Use this when a bone-only Blend and matching skeleton/relative-coordinate documents "
            "need a new immutable Blender Action. Blend and skeleton inputs accept asset_ or "
            "artifact_ IDs; coordinates may be a JSON reference or an inline document."
        ),
        annotations=LOCAL_WRITE,
        structured_output=True,
    )
    def create_bone_animation(
        input_blend_id: str,
        skeleton_id: str,
        coordinates_asset_id: str | None = None,
        coordinates: dict[str, Any] | None = None,
    ) -> JobSummary:
        return local.launcher.create_animation(
            input_blend_id=input_blend_id,
            skeleton_id=skeleton_id,
            coordinates_asset_id=coordinates_asset_id,
            coordinates=coordinates,
        )

    @server.tool(
        name="bind_rigid_components",
        title="Bind rigid model components",
        description=(
            "Use this when a segmented model must follow an animated Armature through rigid bone "
            "parenting without skin weights or mesh deformation. Blend and skeleton inputs "
            "accept asset_ or artifact_ IDs; the component mapping may be a JSON reference or "
            "an inline document."
        ),
        annotations=LOCAL_WRITE,
        structured_output=True,
    )
    def bind_rigid_components(
        input_blend_id: str,
        skeleton_id: str,
        mapping_asset_id: str | None = None,
        mapping: dict[str, Any] | None = None,
    ) -> JobSummary:
        return local.launcher.bind_components(
            input_blend_id=input_blend_id,
            skeleton_id=skeleton_id,
            mapping_asset_id=mapping_asset_id,
            mapping=mapping,
        )

    @server.tool(
        name="inspect_modeling_profile",
        title="Inspect modeling profile",
        description=(
            "Use this read-only check on a registered ViewForge YAML config before face "
            "reconstruction. It reports the admitted view roles and whether every licensed, "
            "hash-locked local model asset is ready without exposing filesystem paths."
        ),
        annotations=READ_ONLY,
        structured_output=True,
    )
    def inspect_modeling_profile(config_asset_id: str) -> ModelingProfileStatus:
        return local.inspect_modeling_profile(config_asset_id)

    @server.tool(
        name="build_declarative_blender_model",
        title="Build declarative Blender model",
        description=(
            "Use this to build a new immutable Blend and GLB from an inline schemaVersion 1 spec "
            "or a registered JSON spec. The typed spec supports allowlisted primitives or explicit "
            "vertices/faces, transforms, solid materials, smoothing, and bevels. It does not "
            "execute code or accept Python scripts."
        ),
        annotations=LOCAL_WRITE,
        structured_output=True,
    )
    def build_declarative_blender_model(
        spec: DeclarativeModelSpec | None = None,
        spec_asset_id: str | None = None,
    ) -> JobSummary:
        return local.launcher.build_declarative_model(
            spec_asset_id=spec_asset_id,
            spec=(
                spec.model_dump(by_alias=True, exclude_none=True)
                if spec is not None
                else None
            ),
        )

    @server.tool(
        name="render_model_preview",
        title="Render model preview",
        description=(
            "Use this to render immutable fixed-view PNG previews from a registered or generated "
            "Blend or GLB. It uses only the plugin-owned Blender renderer, disables embedded "
            "auto-execution, never overwrites the source, and emits individual views, a contact "
            "sheet, and a render manifest. Omit views for perspective/front/right/back/left."
        ),
        annotations=LOCAL_WRITE,
        structured_output=True,
    )
    def render_model_preview(
        source_id: str,
        views: list[RenderView] | None = None,
        resolution: int = 768,
        material_mode: RenderMaterialMode = "original",
        background: RenderBackground = "studio_dark",
    ) -> JobSummary:
        return local.launcher.render_model_preview(
            source_id=source_id,
            views=views,
            resolution=resolution,
            material_mode=material_mode,
            background=background,
        )

    @server.tool(
        name="generate_pixel_cube",
        title="Generate procedural Pixel cube",
        description=(
            "Use this for deterministic local non-Blender geometry generation. It creates a "
            "traceable six-face Pixel shell GLB and manifest from dimensions and resolution."
        ),
        annotations=LOCAL_WRITE,
        structured_output=True,
    )
    def generate_pixel_cube(side_cm: float = 20.0, cells_per_edge: int = 128) -> JobSummary:
        return local.launcher.generate_pixel_cube(side_cm, cells_per_edge)

    @server.tool(
        name="reconstruct_six_view_visual_hull",
        title="Reconstruct six-view visual hull",
        description=(
            "Use this for six reviewed orthographic silhouette images named by role: front, back, "
            "left, right, top, and bottom. It intersects the silhouettes locally and exports a "
            "watertight-candidate GLB with QA and provenance. The result is explicitly "
            "preview-only because silhouettes cannot recover hidden concavities."
        ),
        annotations=LOCAL_WRITE,
        structured_output=True,
    )
    def reconstruct_six_view_visual_hull(
        front_asset_id: str,
        back_asset_id: str,
        left_asset_id: str,
        right_asset_id: str,
        top_asset_id: str,
        bottom_asset_id: str,
        resolution: int = 96,
        width_m: float = 1.0,
        depth_m: float = 1.0,
        height_m: float = 1.0,
    ) -> JobSummary:
        return local.launcher.reconstruct_six_view_visual_hull(
            front_asset_id=front_asset_id,
            back_asset_id=back_asset_id,
            left_asset_id=left_asset_id,
            right_asset_id=right_asset_id,
            top_asset_id=top_asset_id,
            bottom_asset_id=bottom_asset_id,
            resolution=resolution,
            width_m=width_m,
            depth_m=depth_m,
            height_m=height_m,
        )

    @server.tool(
        name="validate_face_multiview",
        title="Validate face multiview inputs",
        description=(
            "Use this before production face reconstruction with registered front, left45, "
            "right45 images and a registered ViewForge YAML config. It runs the local venv-backed "
            "input, landmark, pose, identity, and silhouette checks and writes review artifacts."
        ),
        annotations=LOCAL_WRITE,
        structured_output=True,
    )
    def validate_face_multiview(
        front_asset_id: str,
        left45_asset_id: str,
        right45_asset_id: str,
        config_asset_id: str,
    ) -> JobSummary:
        return local.launcher.validate_face_multiview(
            front_asset_id,
            left45_asset_id,
            right45_asset_id,
            config_asset_id,
        )

    @server.tool(
        name="reconstruct_face_multiview",
        title="Reconstruct face from multiview inputs",
        description=(
            "Use this for the current continuous-template face profile with registered front, "
            "left45, right45 images and a YAML config. Missing or unhashed licensed assets fail "
            "closed. The first run normally enters review_required after generating masks."
        ),
        annotations=LOCAL_WRITE,
        structured_output=True,
    )
    def reconstruct_face_multiview(
        front_asset_id: str,
        left45_asset_id: str,
        right45_asset_id: str,
        config_asset_id: str,
    ) -> JobSummary:
        return local.launcher.reconstruct_face_multiview(
            front_asset_id,
            left45_asset_id,
            right45_asset_id,
            config_asset_id,
        )

    @server.tool(
        name="continue_face_reconstruction",
        title="Continue reviewed face reconstruction",
        description=(
            "Use this only after a person has reviewed the generated masks of a review_required "
            "face reconstruction. approve_masks must be true. A new immutable run is copied, "
            "confirmed, and resumed; the source job remains unchanged."
        ),
        annotations=LOCAL_WRITE,
        structured_output=True,
    )
    def continue_face_reconstruction(
        source_job_id: str,
        approve_masks: bool,
    ) -> JobSummary:
        return local.launcher.continue_face_reconstruction(source_job_id, approve_masks)

    @server.tool(
        name="package_face_reconstruction",
        title="Package face reconstruction",
        description=(
            "Use this only for a succeeded face reconstruction job. It runs the local package "
            "gates and emits an immutable .viewforge3d artifact; it does not publish or upload it."
        ),
        annotations=LOCAL_WRITE,
        structured_output=True,
    )
    def package_face_reconstruction(source_job_id: str) -> JobSummary:
        return local.launcher.package_face_reconstruction(source_job_id)

    @server.tool(
        name="get_viewforge_job",
        title="Get ViewForge job",
        description="Use this when you need the current state and artifacts of one local job.",
        annotations=READ_ONLY,
        structured_output=True,
    )
    def get_viewforge_job(job_id: str) -> JobSummary:
        return local.jobs.public_by_id(job_id)

    @server.tool(
        name="list_viewforge_jobs",
        title="List ViewForge jobs",
        description="Use this when you need recent local jobs and their terminal states.",
        annotations=READ_ONLY,
        structured_output=True,
    )
    def list_viewforge_jobs() -> JobList:
        return JobList(jobs=local.jobs.public_list())

    @server.tool(
        name="list_job_artifacts",
        title="List job artifacts",
        description="Use this when you need private artifact IDs produced by a completed job.",
        annotations=READ_ONLY,
        structured_output=True,
    )
    def list_job_artifacts(job_id: str) -> ArtifactList:
        local.jobs.load(job_id)
        return ArtifactList(job_id=job_id, artifacts=local.artifacts.for_job(job_id))

    @server.tool(
        name="read_json_artifact",
        title="Read JSON artifact",
        description=(
            "Use this when you need a sanitized QA, skeleton, binding, or animation JSON artifact "
            "without exposing local filesystem paths."
        ),
        annotations=READ_ONLY,
        structured_output=True,
    )
    def read_json_artifact(artifact_id: str) -> JSONArtifact:
        return local.resolve_json_artifact(artifact_id)

    @server.tool(
        name="read_image_artifact",
        title="Read rendered image",
        description=(
            "Use this only for a selected PNG, JPEG, or WebP artifact after a render or QA job. "
            "It returns that local image to the conversation without exposing its filesystem path."
        ),
        annotations=READ_ONLY,
        structured_output=False,
    )
    def read_image_artifact(artifact_id: str) -> CallToolResult:
        return local.resolve_image_artifact(artifact_id)

    return server


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the private ViewForge Local MCP server.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="streamable-http",
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.transport == "stdio":
        with contextlib.suppress(KeyboardInterrupt):
            create_server(Runtime()).run("stdio")
        return
    bootstrap = ConfigurationStore(LocalPaths()).load()
    host = arguments.host or bootstrap.mcp_host
    port = arguments.port or bootstrap.mcp_port
    runtime = Runtime(endpoint_host=host, endpoint_port=port)
    server = create_server(runtime)
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("ViewForge Local only binds to a loopback host.")
    if not 1024 <= port <= 65535:
        raise SystemExit("MCP port must be between 1024 and 65535.")
    with contextlib.suppress(KeyboardInterrupt):
        server.run(
            "streamable-http",
            host=host,
            port=port,
            streamable_http_path="/mcp",
            json_response=True,
            stateless_http=True,
        )


if __name__ == "__main__":
    main()
