from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from face3d.errors import Face3DError

from .models import JobKind, JobState, JobSummary
from .storage import (
    ArtifactStore,
    AssetStore,
    ConfigurationStore,
    JobStore,
    LocalPaths,
    StoredJob,
    atomic_write_json,
    is_within,
    sha256_file,
    utc_now,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".heic", ".heif"}
CONFIG_EXTENSIONS = {".yaml", ".yml"}
BLENDER_JOB_KINDS = {
    JobKind.BUILD_SKELETON,
    JobKind.CREATE_BONE_ANIMATION,
    JobKind.BIND_RIGID_COMPONENTS,
    JobKind.BUILD_DECLARATIVE_MODEL,
    JobKind.RENDER_MODEL_PREVIEW,
}
RENDER_VIEWS = {
    "perspective",
    "front",
    "back",
    "left",
    "right",
    "top",
    "bottom",
}
DEFAULT_RENDER_VIEWS = ["perspective", "front", "right", "back", "left"]
RENDER_MATERIAL_MODES = {"original", "neutral"}
RENDER_BACKGROUNDS = {"studio_dark", "studio_light", "transparent"}


def load_workspace_modeling_config(
    config_path: Path,
    workspace: Path,
    *,
    expected_sha256: str | None = None,
) -> Any:
    """Load a modeling config only when all of its local assets stay in the workspace."""
    from face3d.assets import asset_paths, manifest_path
    from face3d.config import load_config

    workspace = workspace.resolve()
    config_path = config_path.expanduser().resolve()
    if not is_within(config_path, workspace):
        raise ValueError("The modeling config must stay inside the configured workspace.")
    if expected_sha256 is not None and sha256_file(config_path) != expected_sha256:
        raise ValueError("The registered modeling config changed after the job was created.")

    config = load_config(config_path)
    scoped_paths = {
        "projectRoot": config.project_root,
        "assetManifest": manifest_path(config),
        **asset_paths(config),
    }
    if any(not is_within(path.expanduser().resolve(), workspace) for path in scoped_paths.values()):
        raise ValueError("The modeling config references a path outside the configured workspace.")
    return config


@dataclass(frozen=True)
class JobRequest:
    schema_version: int
    job_id: str
    kind: str
    blender_executable: str | None = None
    plugin_root: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    runtime: Literal["blender", "modeling"] = "blender"


class ReferenceResolver:
    def __init__(self, assets: AssetStore, artifacts: ArtifactStore) -> None:
        self.assets = assets
        self.artifacts = artifacts

    def resolve(self, reference_id: str, extensions: set[str]) -> Path:
        if reference_id.startswith("asset_"):
            path = self.assets.resolve(reference_id)
        elif reference_id.startswith("artifact_"):
            path = self.artifacts.resolve(reference_id)
        else:
            raise ValueError("References must use an asset_ or artifact_ ID.")
        if path.suffix.lower() not in extensions:
            raise ValueError("The referenced file type is not valid for this operation.")
        return path


class JobLauncher:
    def __init__(
        self,
        paths: LocalPaths,
        configuration: ConfigurationStore,
        assets: AssetStore,
        artifacts: ArtifactStore,
        jobs: JobStore,
    ) -> None:
        self.paths = paths
        self.configuration = configuration
        self.assets = assets
        self.artifacts = artifacts
        self.jobs = jobs
        self.references = ReferenceResolver(assets, artifacts)

    def _blender_runtime(self) -> tuple[Path, Path]:
        blender = self.configuration.blender()
        plugin_root = self.configuration.plugin_root()
        if blender is None:
            raise RuntimeError("Blender is not available. Configure it in ViewForge Local.")
        if plugin_root is None:
            raise RuntimeError("The ViewForge plugin runtime is not available.")
        return blender, plugin_root

    def _assert_capacity(self) -> None:
        if self.jobs.active_count() >= 1:
            raise RuntimeError("A local geometry job is already queued or running.")

    def _create(
        self,
        kind: JobKind,
        arguments: dict[str, Any],
        *,
        runtime: Literal["blender", "modeling"],
        staged_inputs: dict[str, tuple[Path, str]] | None = None,
        inline_json_inputs: dict[str, tuple[dict[str, Any], str]] | None = None,
    ) -> JobSummary:
        self._assert_capacity()
        blender: Path | None = None
        plugin_root: Path | None = None
        if runtime == "blender":
            blender, plugin_root = self._blender_runtime()
        elif not self.configuration.modeling_runtime_available():
            raise RuntimeError(
                "The bundled ViewForge modeling runtime is unavailable. Rebuild ViewForge Local."
            )

        self.paths.ensure()
        job_id = f"job_{uuid.uuid4().hex[:24]}"
        job_dir = self.jobs.directory(job_id)
        job_dir.mkdir(parents=False, exist_ok=False)
        prepared_arguments = dict(arguments)
        if staged_inputs or inline_json_inputs:
            inputs_dir = job_dir / "inputs"
            inputs_dir.mkdir()
        if staged_inputs:
            for argument_name, (source, relative_name) in staged_inputs.items():
                destination = (inputs_dir / relative_name).resolve()
                if not is_within(destination, inputs_dir.resolve()):
                    raise ValueError("Invalid staged input name.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                prepared_arguments[argument_name] = str(destination)
        if inline_json_inputs:
            for argument_name, (document, relative_name) in inline_json_inputs.items():
                destination = (inputs_dir / relative_name).resolve()
                if not is_within(destination, inputs_dir.resolve()):
                    raise ValueError("Invalid inline input name.")
                atomic_write_json(destination, document)
                prepared_arguments[argument_name] = str(destination)

        request = JobRequest(
            schema_version=2,
            job_id=job_id,
            kind=kind.value,
            runtime=runtime,
            blender_executable=str(blender) if blender else None,
            plugin_root=str(plugin_root) if plugin_root else None,
            arguments=prepared_arguments,
        )
        request_path = job_dir / "request.json"
        atomic_write_json(
            request_path,
            {
                "schemaVersion": request.schema_version,
                "jobId": request.job_id,
                "kind": request.kind,
                "runtime": request.runtime,
                "blenderExecutable": request.blender_executable,
                "pluginRoot": request.plugin_root,
                "arguments": request.arguments,
            },
        )
        now = utc_now()
        job = StoredJob(
            id=job_id,
            kind=kind.value,
            state=JobState.QUEUED.value,
            created_at=now,
            updated_at=now,
            request_path=str(request_path),
        )
        self.jobs.save(job)
        environment = os.environ.copy()
        environment["VIEWFORGE_LOCAL_STATE_DIR"] = str(self.paths.root)
        command = [sys.executable, "-m", "face3d.local_mcp.worker", "--request", str(request_path)]
        try:
            worker = subprocess.Popen(
                command,
                cwd=job_dir,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as error:
            job.state = JobState.FAILED.value
            job.failure = "The local worker could not start."
            self.jobs.save(job)
            raise RuntimeError(job.failure) from error
        job.worker_pid = worker.pid
        self.jobs.save(job)
        return self.jobs.public(job)

    def build_skeleton(
        self,
        asset_id: str,
        profile: str,
        landmarks_asset_id: str | None = None,
        component_map_asset_id: str | None = None,
        front_annotation_asset_id: str | None = None,
        side_annotation_asset_id: str | None = None,
    ) -> JobSummary:
        source = self.references.resolve(asset_id, {".glb", ".gltf"})
        if profile == "quadruped-v1" and landmarks_asset_id is None:
            raise ValueError("quadruped-v1 requires an explicit 3D landmark JSON asset.")
        arguments: dict[str, Any] = {"profile": profile}
        staged = {"input": (source, f"source{source.suffix.lower()}")}
        optional = {
            "landmarks": (landmarks_asset_id, {".json"}),
            "component_map": (component_map_asset_id, {".json"}),
            "front_annotation": (front_annotation_asset_id, {".png", ".jpg", ".jpeg"}),
            "side_annotation": (side_annotation_asset_id, {".png", ".jpg", ".jpeg"}),
        }
        for key, (reference_id, extensions) in optional.items():
            if reference_id:
                path = self.references.resolve(reference_id, extensions)
                staged[key] = (path, f"{key}{path.suffix.lower()}")
        return self._create(
            JobKind.BUILD_SKELETON,
            arguments,
            runtime="blender",
            staged_inputs=staged,
        )

    def create_animation(
        self,
        input_blend_id: str,
        skeleton_id: str,
        coordinates_asset_id: str,
    ) -> JobSummary:
        staged = {
            "input_blend": (
                self.references.resolve(input_blend_id, {".blend"}),
                "source.blend",
            ),
            "skeleton": (
                self.references.resolve(skeleton_id, {".json"}),
                "skeleton.json",
            ),
            "coordinates": (
                self.references.resolve(coordinates_asset_id, {".json"}),
                "coordinates.json",
            ),
        }
        return self._create(
            JobKind.CREATE_BONE_ANIMATION,
            {},
            runtime="blender",
            staged_inputs=staged,
        )

    def bind_components(
        self,
        input_blend_id: str,
        skeleton_id: str,
        mapping_asset_id: str,
    ) -> JobSummary:
        staged = {
            "input_blend": (
                self.references.resolve(input_blend_id, {".blend"}),
                "source.blend",
            ),
            "skeleton": (
                self.references.resolve(skeleton_id, {".json"}),
                "skeleton.json",
            ),
            "mapping": (
                self.references.resolve(mapping_asset_id, {".json"}),
                "mapping.json",
            ),
        }
        return self._create(
            JobKind.BIND_RIGID_COMPONENTS,
            {},
            runtime="blender",
            staged_inputs=staged,
        )

    def build_declarative_model(
        self,
        *,
        spec_asset_id: str | None = None,
        spec: dict[str, Any] | None = None,
    ) -> JobSummary:
        if (spec_asset_id is None) == (spec is None):
            raise ValueError("Provide exactly one of spec or spec_asset_id.")
        staged = None
        inline = None
        if spec_asset_id is not None:
            source = self.references.resolve(spec_asset_id, {".json"})
            staged = {"spec": (source, "model-spec.json")}
        else:
            inline = {"spec": (spec or {}, "model-spec.json")}
        return self._create(
            JobKind.BUILD_DECLARATIVE_MODEL,
            {},
            runtime="blender",
            staged_inputs=staged,
            inline_json_inputs=inline,
        )

    def render_model_preview(
        self,
        *,
        source_id: str,
        views: list[str] | None,
        resolution: int,
        material_mode: str,
        background: str,
    ) -> JobSummary:
        source = self.references.resolve(source_id, {".blend", ".glb"})
        selected_views = list(views) if views is not None else list(DEFAULT_RENDER_VIEWS)
        if not 1 <= len(selected_views) <= len(RENDER_VIEWS):
            raise ValueError("views must contain between one and seven values.")
        if len(set(selected_views)) != len(selected_views):
            raise ValueError("views must not contain duplicates.")
        unknown_views = sorted(set(selected_views) - RENDER_VIEWS)
        if unknown_views:
            raise ValueError(f"Unsupported render views: {unknown_views}")
        if isinstance(resolution, bool) or not 256 <= resolution <= 1024:
            raise ValueError("resolution must be between 256 and 1024.")
        if material_mode not in RENDER_MATERIAL_MODES:
            raise ValueError("material_mode must be original or neutral.")
        if background not in RENDER_BACKGROUNDS:
            raise ValueError(
                "background must be studio_dark, studio_light, or transparent."
            )
        return self._create(
            JobKind.RENDER_MODEL_PREVIEW,
            {
                "views": selected_views,
                "resolution": int(resolution),
                "material_mode": material_mode,
                "background": background,
            },
            runtime="blender",
            staged_inputs={"input": (source, f"source{source.suffix.lower()}")},
        )

    def generate_pixel_cube(self, side_cm: float, cells_per_edge: int) -> JobSummary:
        if not 0.1 <= side_cm <= 100_000:
            raise ValueError("side_cm must be between 0.1 and 100000.")
        if not 2 <= cells_per_edge <= 512:
            raise ValueError("cells_per_edge must be between 2 and 512.")
        return self._create(
            JobKind.GENERATE_PIXEL_CUBE,
            {"side_cm": float(side_cm), "cells_per_edge": int(cells_per_edge)},
            runtime="modeling",
        )

    def reconstruct_six_view_visual_hull(
        self,
        *,
        front_asset_id: str,
        back_asset_id: str,
        left_asset_id: str,
        right_asset_id: str,
        top_asset_id: str,
        bottom_asset_id: str,
        resolution: int,
        width_m: float,
        depth_m: float,
        height_m: float,
    ) -> JobSummary:
        references = {
            "front": front_asset_id,
            "back": back_asset_id,
            "left": left_asset_id,
            "right": right_asset_id,
            "top": top_asset_id,
            "bottom": bottom_asset_id,
        }
        staged: dict[str, tuple[Path, str]] = {}
        for role, reference_id in references.items():
            source = self.references.resolve(reference_id, IMAGE_EXTENSIONS)
            staged[role] = (source, f"views/{role}{source.suffix.lower()}")
        return self._create(
            JobKind.RECONSTRUCT_SIX_VIEW_VISUAL_HULL,
            {
                "resolution": int(resolution),
                "width_m": float(width_m),
                "depth_m": float(depth_m),
                "height_m": float(height_m),
            },
            runtime="modeling",
            staged_inputs=staged,
        )

    def _face_inputs(
        self,
        front_asset_id: str,
        left45_asset_id: str,
        right45_asset_id: str,
        config_asset_id: str,
    ) -> tuple[dict[str, Any], dict[str, tuple[Path, str]]]:
        config = self.references.resolve(config_asset_id, CONFIG_EXTENSIONS)
        workspace = self.configuration.workspace()
        if workspace is None:
            raise RuntimeError("Configure a local workspace before starting a modeling job.")
        load_workspace_modeling_config(config, workspace)
        references = {
            "front": front_asset_id,
            "left45": left45_asset_id,
            "right45": right45_asset_id,
        }
        staged: dict[str, tuple[Path, str]] = {}
        for role, reference_id in references.items():
            source = self.references.resolve(reference_id, IMAGE_EXTENSIONS)
            staged[role] = (source, f"views/{role}{source.suffix.lower()}")
        return {"config": str(config), "config_sha256": sha256_file(config)}, staged

    def validate_face_multiview(
        self,
        front_asset_id: str,
        left45_asset_id: str,
        right45_asset_id: str,
        config_asset_id: str,
    ) -> JobSummary:
        arguments, staged = self._face_inputs(
            front_asset_id,
            left45_asset_id,
            right45_asset_id,
            config_asset_id,
        )
        return self._create(
            JobKind.VALIDATE_FACE_MULTIVIEW,
            arguments,
            runtime="modeling",
            staged_inputs=staged,
        )

    def reconstruct_face_multiview(
        self,
        front_asset_id: str,
        left45_asset_id: str,
        right45_asset_id: str,
        config_asset_id: str,
    ) -> JobSummary:
        arguments, staged = self._face_inputs(
            front_asset_id,
            left45_asset_id,
            right45_asset_id,
            config_asset_id,
        )
        return self._create(
            JobKind.RECONSTRUCT_FACE_MULTIVIEW,
            arguments,
            runtime="modeling",
            staged_inputs=staged,
        )

    def _source_reconstruction(self, source_job_id: str) -> tuple[StoredJob, JobRequest, Path]:
        source_job = self.jobs.load(source_job_id)
        allowed = {
            JobKind.RECONSTRUCT_FACE_MULTIVIEW.value,
            JobKind.CONTINUE_FACE_RECONSTRUCTION.value,
        }
        if source_job.kind not in allowed:
            raise ValueError("The source job is not a face reconstruction job.")
        source_request = load_request(Path(source_job.request_path))
        source_run = (self.jobs.directory(source_job_id) / "output" / "run").resolve()
        if not source_run.is_dir() or not is_within(source_run, self.paths.jobs.resolve()):
            raise ValueError("The source reconstruction run is unavailable.")
        return source_job, source_request, source_run

    def continue_face_reconstruction(self, source_job_id: str, approve_masks: bool) -> JobSummary:
        if approve_masks is not True:
            raise ValueError("approve_masks must be true after the generated masks were reviewed.")
        source_job, source_request, source_run = self._source_reconstruction(source_job_id)
        if source_job.state != JobState.REVIEW_REQUIRED.value:
            raise ValueError("The source reconstruction is not waiting for mask review.")
        config_sha256 = source_request.arguments.get("config_sha256")
        if not isinstance(config_sha256, str):
            raise ValueError("The source job does not contain an immutable modeling config hash.")
        return self._create(
            JobKind.CONTINUE_FACE_RECONSTRUCTION,
            {
                "source_job_id": source_job_id,
                "source_run": str(source_run),
                "source_input": source_request.arguments.get(
                    "source_input",
                    str(self.jobs.directory(source_job_id) / "inputs" / "views"),
                ),
                "config": source_request.arguments["config"],
                "config_sha256": config_sha256,
                "approve_masks": True,
            },
            runtime="modeling",
        )

    def package_face_reconstruction(self, source_job_id: str) -> JobSummary:
        source_job, source_request, source_run = self._source_reconstruction(source_job_id)
        if source_job.state != JobState.SUCCEEDED.value:
            raise ValueError("Only a succeeded reconstruction can be packaged.")
        config_sha256 = source_request.arguments.get("config_sha256")
        if not isinstance(config_sha256, str):
            raise ValueError("The source job does not contain an immutable modeling config hash.")
        return self._create(
            JobKind.PACKAGE_FACE_RECONSTRUCTION,
            {
                "source_job_id": source_job_id,
                "source_run": str(source_run),
                "config": source_request.arguments["config"],
                "config_sha256": config_sha256,
            },
            runtime="modeling",
        )


def load_request(path: Path) -> JobRequest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = payload.get("schemaVersion")
    if schema_version not in {1, 2}:
        raise ValueError("Unsupported job request schema.")
    return JobRequest(
        schema_version=int(schema_version),
        job_id=payload["jobId"],
        kind=payload["kind"],
        runtime=payload.get("runtime", "blender"),
        blender_executable=payload.get("blenderExecutable"),
        plugin_root=payload.get("pluginRoot"),
        arguments=dict(payload["arguments"]),
    )


def _script(plugin_root: Path, relative: str) -> Path:
    path = (plugin_root / relative).resolve()
    if not path.is_file() or not path.is_relative_to(plugin_root.resolve()):
        raise RuntimeError("Required ViewForge worker script is unavailable.")
    return path


def blender_command(request: JobRequest, output_dir: Path) -> list[str]:
    if not request.blender_executable or not request.plugin_root:
        raise RuntimeError("The Blender runtime is incomplete.")
    blender = Path(request.blender_executable).resolve()
    plugin_root = Path(request.plugin_root).resolve()
    if not blender.is_file():
        raise RuntimeError("Configured Blender executable is unavailable.")
    if request.kind == JobKind.BUILD_SKELETON.value:
        script = _script(
            plugin_root,
            "skills/build-biological-skeleton/scripts/build_biological_skeleton.py",
        )
        command = [
            str(blender),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python",
            str(script),
            "--",
            "--input",
            request.arguments["input"],
            "--output-dir",
            str(output_dir),
            "--profile",
            request.arguments["profile"],
        ]
        for key, option in (
            ("landmarks", "--landmarks"),
            ("component_map", "--component-map"),
            ("front_annotation", "--front-annotation"),
            ("side_annotation", "--side-annotation"),
        ):
            if key in request.arguments:
                command.extend([option, request.arguments[key]])
        return command
    if request.kind == JobKind.CREATE_BONE_ANIMATION.value:
        script = _script(
            plugin_root,
            "skills/animate-biological-skeleton/scripts/build_bone_animation.py",
        )
        return [
            str(blender),
            "--background",
            "--disable-autoexec",
            request.arguments["input_blend"],
            "--python",
            str(script),
            "--",
            "--input-blend",
            request.arguments["input_blend"],
            "--skeleton",
            request.arguments["skeleton"],
            "--coordinates",
            request.arguments["coordinates"],
            "--output-blend",
            str(output_dir / "bone-animation.blend"),
            "--qa",
            str(output_dir / "animation-qa.json"),
        ]
    if request.kind == JobKind.BIND_RIGID_COMPONENTS.value:
        script = _script(
            plugin_root,
            "skills/animate-biological-skeleton/scripts/bind_rigid_components.py",
        )
        return [
            str(blender),
            "--background",
            "--disable-autoexec",
            request.arguments["input_blend"],
            "--python",
            str(script),
            "--",
            "--input-blend",
            request.arguments["input_blend"],
            "--skeleton",
            request.arguments["skeleton"],
            "--mapping",
            request.arguments["mapping"],
            "--output-blend",
            str(output_dir / "rigid-bound-animation.blend"),
            "--qa",
            str(output_dir / "binding-qa.json"),
        ]
    if request.kind == JobKind.BUILD_DECLARATIVE_MODEL.value:
        script = _script(plugin_root, "runtime/blender/build_declarative_model.py")
        return [
            str(blender),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python",
            str(script),
            "--",
            "--spec",
            request.arguments["spec"],
            "--output-blend",
            str(output_dir / "declarative-model.blend"),
            "--output-glb",
            str(output_dir / "declarative-model.glb"),
            "--qa",
            str(output_dir / "modeling-qa.json"),
        ]
    if request.kind == JobKind.RENDER_MODEL_PREVIEW.value:
        script = _script(plugin_root, "runtime/blender/render_model_preview.py")
        command = [
            str(blender),
            "--background",
            "--disable-autoexec",
        ]
        if Path(request.arguments["input"]).suffix.lower() == ".blend":
            command.append(request.arguments["input"])
        else:
            command.append("--factory-startup")
        command.extend(
            [
                "--python",
                str(script),
                "--",
                "--input",
                request.arguments["input"],
                "--output-dir",
                str(output_dir / "renders"),
                "--views",
                ",".join(request.arguments["views"]),
                "--resolution",
                str(request.arguments["resolution"]),
                "--material-mode",
                request.arguments["material_mode"],
                "--background",
                request.arguments["background"],
            ]
        )
        return command
    raise ValueError("Unsupported Blender job kind.")


def _worker_environment(job_dir: Path) -> dict[str, str]:
    allowed = ("HOME", "USER", "LOGNAME", "PATH", "TMPDIR", "LANG", "LC_ALL")
    environment = {key: os.environ[key] for key in allowed if key in os.environ}
    environment.setdefault("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    environment["PYTHONNOUSERSITE"] = "1"
    environment["BLENDER_USER_CONFIG"] = str(job_dir / "blender-config")
    return environment


def _request_modeling_config(request: JobRequest, paths: LocalPaths) -> Any:
    expected_sha256 = request.arguments.get("config_sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise RuntimeError("The modeling job is missing its immutable config hash.")
    workspace = ConfigurationStore(paths).workspace()
    if workspace is None:
        raise RuntimeError("The configured modeling workspace is unavailable.")
    return load_workspace_modeling_config(
        Path(request.arguments["config"]),
        workspace,
        expected_sha256=expected_sha256,
    )


def _modeling_result(request: JobRequest, paths: LocalPaths, output_dir: Path) -> dict[str, Any]:
    kind = JobKind(request.kind)
    if kind == JobKind.GENERATE_PIXEL_CUBE:
        from face3d.pixel_cube import PixelCubeSpec, create_pixel_cube

        return create_pixel_cube(
            output_dir / "pixel-cube",
            PixelCubeSpec(
                side_length_m=float(request.arguments["side_cm"]) / 100,
                cells_per_edge=int(request.arguments["cells_per_edge"]),
            ),
        )
    if kind == JobKind.RECONSTRUCT_SIX_VIEW_VISUAL_HULL:
        from face3d.six_view_visual_hull import SIX_VIEW_ROLES, reconstruct_six_view_visual_hull

        return reconstruct_six_view_visual_hull(
            {role: Path(request.arguments[role]) for role in SIX_VIEW_ROLES},
            output_dir / "six-view-visual-hull",
            resolution=int(request.arguments["resolution"]),
            width_m=float(request.arguments["width_m"]),
            depth_m=float(request.arguments["depth_m"]),
            height_m=float(request.arguments["height_m"]),
        )
    if kind == JobKind.VALIDATE_FACE_MULTIVIEW:
        from face3d.stages.intake import validate_only

        return validate_only(
            Path(request.arguments["front"]).parent,
            _request_modeling_config(request, paths),
            output_dir / "validation",
        )
    if kind == JobKind.RECONSTRUCT_FACE_MULTIVIEW:
        from face3d.pipeline import reconstruct

        return reconstruct(
            Path(request.arguments["front"]).parent,
            output_dir / "run",
            _request_modeling_config(request, paths),
        )
    if kind == JobKind.CONTINUE_FACE_RECONSTRUCTION:
        from face3d.pipeline import reconstruct
        from face3d.stages.intake import confirm_masks

        if request.arguments.get("approve_masks") is not True:
            raise RuntimeError("Mask review approval is required.")
        source_run = Path(request.arguments["source_run"]).resolve()
        source_input = Path(request.arguments["source_input"]).resolve()
        if not source_run.is_dir() or not is_within(source_run, paths.jobs.resolve()):
            raise RuntimeError("The source reconstruction is unavailable.")
        if not source_input.is_dir() or not is_within(source_input, paths.jobs.resolve()):
            raise RuntimeError("The source reconstruction inputs are unavailable.")
        run_dir = output_dir / "run"
        shutil.copytree(source_run, run_dir)
        confirm_masks(run_dir)
        return reconstruct(
            source_input,
            run_dir,
            _request_modeling_config(request, paths),
        )
    if kind == JobKind.PACKAGE_FACE_RECONSTRUCTION:
        from face3d.package import package_run

        source_run = Path(request.arguments["source_run"]).resolve()
        if not source_run.is_dir() or not is_within(source_run, paths.jobs.resolve()):
            raise RuntimeError("The source reconstruction is unavailable.")
        return package_run(
            source_run,
            output_dir / "reconstruction.viewforge3d",
            _request_modeling_config(request, paths),
        )
    raise ValueError("Unsupported modeling job kind.")


def _register_generated(
    job: StoredJob,
    output_dir: Path,
    artifacts: ArtifactStore,
) -> list[str]:
    admitted_extensions = {
        ".bin",
        ".blend",
        ".gif",
        ".glb",
        ".gltf",
        ".jpeg",
        ".jpg",
        ".json",
        ".jsonl",
        ".mp4",
        ".npz",
        ".png",
        ".txt",
        ".viewforge3d",
        ".yaml",
        ".yml",
    }
    generated = [
        path
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in admitted_extensions
    ]
    job.artifact_ids = [artifacts.register(job.id, path).id for path in generated]
    return job.artifact_ids


def run_job(request_path: Path, paths: LocalPaths | None = None) -> int:
    local_paths = paths or LocalPaths()
    local_paths.ensure()
    artifacts = ArtifactStore(local_paths)
    jobs = JobStore(local_paths, artifacts)
    request = load_request(request_path.resolve())
    job = jobs.load(request.job_id)
    job.state = JobState.RUNNING.value
    job.status_message = None
    job.worker_pid = os.getpid()
    jobs.save(job)
    job_dir = jobs.directory(job.id)
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = job_dir / "worker.log"
    try:
        if request.runtime == "blender":
            command = blender_command(request, output_dir)
            with log_path.open("wb") as log:
                process = subprocess.Popen(
                    command,
                    cwd=job_dir,
                    env=_worker_environment(job_dir),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                )
                job.blender_pid = process.pid
                jobs.save(job)
                exit_code = process.wait()
            job.exit_code = exit_code
            if exit_code != 0:
                raise RuntimeError(f"Blender worker exited with code {exit_code}.")
            if request.kind == JobKind.RENDER_MODEL_PREVIEW.value:
                from .rendering import compose_preview_sheet

                render_dir = output_dir / "renders"
                sheet = compose_preview_sheet(
                    render_dir,
                    list(request.arguments["views"]),
                    output_dir / "render-preview-sheet.png",
                )
                manifest_path = render_dir / "render-manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["render"]["contactSheet"] = sheet
                atomic_write_json(manifest_path, manifest)
        else:
            with log_path.open("w", encoding="utf-8") as log:
                log.write(f"ViewForge modeling job {request.kind}\n")
                result = _modeling_result(request, local_paths, output_dir)
                atomic_write_json(output_dir / "result.json", result)
            job.exit_code = 0
        if not _register_generated(job, output_dir, artifacts):
            raise RuntimeError("The local geometry worker produced no registered artifacts.")
        job.state = JobState.SUCCEEDED.value
        job.status_message = "The local geometry job completed."
        job.failure = None
        jobs.save(job)
        return 0
    except Face3DError as error:
        atomic_write_json(output_dir / "error.json", error.as_dict())
        _register_generated(job, output_dir, artifacts)
        job.exit_code = error.exit_code
        if error.code == "mask-review-required":
            job.state = JobState.REVIEW_REQUIRED.value
            job.status_message = (
                "Review the generated silhouette masks, then call "
                "continue_face_reconstruction with approve_masks=true."
            )
            job.failure = None
            jobs.save(job)
            return 0
        job.state = JobState.FAILED.value
        job.status_message = f"{error.code} ({error.stage})"
        job.failure = (
            "The local modeling job failed. Read the sanitized error.json artifact for details."
        )
        jobs.save(job)
        return 1
    except Exception:
        job.state = JobState.FAILED.value
        job.status_message = None
        job.failure = "The local geometry job failed. Check the local worker log."
        jobs.save(job)
        return 1
