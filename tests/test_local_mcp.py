from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import anyio
import pytest
from mcp.types import ImageContent
from PIL import Image

from face3d.local_mcp.jobs import JobRequest, blender_command, load_request, run_job
from face3d.local_mcp.models import AssetKind, JobKind, JobState
from face3d.local_mcp.rendering import compose_preview_sheet
from face3d.local_mcp.server import Runtime, create_server
from face3d.local_mcp.storage import (
    ArtifactStore,
    ConfigurationStore,
    LocalConfiguration,
    LocalPaths,
    StoredJob,
    atomic_write_json,
    utc_now,
)


def configured_runtime(tmp_path: Path) -> tuple[Runtime, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    paths = LocalPaths(tmp_path / "state")
    paths.ensure()
    configuration = ConfigurationStore(paths)
    configuration.save(
        LocalConfiguration(
            workspace_root=str(workspace),
            blender_executable="/Applications/Blender.app/Contents/MacOS/Blender",
            plugin_root=str(
                Path(__file__).resolve().parents[1] / "plugins" / "viewforge-3d-toolkit"
            ),
        )
    )
    return Runtime(paths), workspace


def test_asset_registration_stays_inside_workspace_and_hides_path(tmp_path: Path) -> None:
    runtime, workspace = configured_runtime(tmp_path)
    model = workspace / "character.glb"
    model.write_bytes(b"glTF fixture")

    asset = runtime.assets.register("character.glb")

    assert asset.kind == AssetKind.MODEL
    assert asset.id.startswith("asset_")
    assert str(workspace) not in asset.model_dump_json()
    outside = tmp_path / "outside.glb"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="inside the configured workspace"):
        runtime.assets.register(str(outside))


def test_registered_asset_detects_mutation(tmp_path: Path) -> None:
    runtime, workspace = configured_runtime(tmp_path)
    document = workspace / "coordinates.json"
    document.write_text("{}", encoding="utf-8")
    asset = runtime.assets.register(str(document))
    document.write_text('{"changed": true}', encoding="utf-8")

    with pytest.raises(ValueError, match="changed"):
        runtime.assets.resolve(asset.id)


def test_yaml_modeling_config_can_be_registered(tmp_path: Path) -> None:
    runtime, workspace = configured_runtime(tmp_path)
    config = workspace / "face-v3.yaml"
    source = Path(__file__).resolve().parents[1] / "configs" / "face-v3.yaml"
    config.write_bytes(source.read_bytes())

    asset = runtime.assets.register(config.name)
    status = runtime.inspect_modeling_profile(asset.id)

    assert asset.kind == AssetKind.CONFIG
    assert status.profile == "face-v3"
    assert status.required_views == ["front", "left45", "right45"]
    assert status.assets_ready is False
    assert str(workspace) not in status.model_dump_json()


def test_modeling_config_cannot_reference_assets_outside_workspace(tmp_path: Path) -> None:
    runtime, workspace = configured_runtime(tmp_path)
    outside = tmp_path / "outside.task"
    outside.write_bytes(b"fixture")
    source = Path(__file__).resolve().parents[1] / "configs" / "face-v3.yaml"
    payload = source.read_text(encoding="utf-8").replace(
        "face_landmarker: .local/models/mediapipe/face_landmarker.task",
        f"face_landmarker: {outside}",
    )
    config = workspace / "face-v3.yaml"
    config.write_text(payload, encoding="utf-8")
    asset = runtime.assets.register(config.name)

    with pytest.raises(ValueError, match="outside the configured workspace"):
        runtime.inspect_modeling_profile(asset.id)


def test_json_artifact_redacts_local_paths(tmp_path: Path) -> None:
    runtime, workspace = configured_runtime(tmp_path)
    job_id = "job_fixture"
    output = runtime.paths.jobs / job_id / "output"
    output.mkdir(parents=True)
    qa = output / "qa.json"
    qa.write_text(
        json.dumps(
            {
                "source": {"path": str(workspace / "secret.glb")},
                "output": str(runtime.paths.root / "jobs" / job_id / "result.blend"),
            }
        ),
        encoding="utf-8",
    )
    artifact = ArtifactStore(runtime.paths).register(job_id, qa)

    result = runtime.resolve_json_artifact(artifact.id)
    serialized = result.model_dump_json()

    assert str(workspace) not in serialized
    assert str(runtime.paths.root) not in serialized
    assert "secret.glb" in serialized


def test_mcp_tool_contract_has_focused_annotated_tools(tmp_path: Path) -> None:
    runtime, _ = configured_runtime(tmp_path)
    server = create_server(runtime)

    tools = anyio.run(server.list_tools)
    names = {tool.name for tool in tools}

    assert names == {
        "viewforge_status",
        "register_local_asset",
        "list_local_assets",
        "build_biological_skeleton",
        "create_bone_animation",
        "bind_rigid_components",
        "inspect_modeling_profile",
        "build_declarative_blender_model",
        "render_model_preview",
        "generate_pixel_cube",
        "reconstruct_six_view_visual_hull",
        "validate_face_multiview",
        "reconstruct_face_multiview",
        "continue_face_reconstruction",
        "package_face_reconstruction",
        "get_viewforge_job",
        "list_viewforge_jobs",
        "list_job_artifacts",
        "read_json_artifact",
        "read_image_artifact",
    }
    assert all(tool.annotations is not None for tool in tools)
    by_name = {tool.name: tool for tool in tools}
    assert by_name["viewforge_status"].annotations.read_only_hint is True
    assert by_name["inspect_modeling_profile"].annotations.read_only_hint is True
    assert by_name["build_biological_skeleton"].annotations.read_only_hint is False
    assert by_name["render_model_preview"].annotations.read_only_hint is False
    assert by_name["read_image_artifact"].annotations.read_only_hint is True
    assert by_name["generate_pixel_cube"].annotations.open_world_hint is False
    assert by_name["build_biological_skeleton"].annotations.open_world_hint is False
    schema = by_name["build_declarative_blender_model"].input_schema
    object_schema = schema["$defs"]["DeclarativeModelObject"]
    assert object_schema["properties"]["primitive"]["enum"] == [
        "cube",
        "uv_sphere",
        "ico_sphere",
        "cylinder",
        "cone",
        "torus",
        "mesh",
    ]
    render_schema = by_name["render_model_preview"].input_schema
    assert render_schema["required"] == ["source_id"]
    assert render_schema["properties"]["resolution"]["default"] == 768


def test_blender_commands_disable_embedded_autoexec(tmp_path: Path) -> None:
    plugin_root = Path(__file__).resolve().parents[1] / "plugins" / "viewforge-3d-toolkit"
    request = JobRequest(
        schema_version=1,
        job_id="job_fixture",
        kind=JobKind.CREATE_BONE_ANIMATION.value,
        blender_executable="/Applications/Blender.app/Contents/MacOS/Blender",
        plugin_root=str(plugin_root),
        arguments={
            "input_blend": str(tmp_path / "input.blend"),
            "skeleton": str(tmp_path / "skeleton.json"),
            "coordinates": str(tmp_path / "coordinates.json"),
        },
    )

    command = blender_command(request, tmp_path / "output")

    assert "--background" in command
    assert "--disable-autoexec" in command
    assert command.count("--python") == 1


def test_declarative_blender_command_uses_plugin_owned_script(tmp_path: Path) -> None:
    plugin_root = Path(__file__).resolve().parents[1] / "plugins" / "viewforge-3d-toolkit"
    request = JobRequest(
        schema_version=2,
        job_id="job_fixture",
        kind=JobKind.BUILD_DECLARATIVE_MODEL.value,
        blender_executable="/Applications/Blender.app/Contents/MacOS/Blender",
        plugin_root=str(plugin_root),
        arguments={"spec": str(tmp_path / "model-spec.json")},
    )

    command = blender_command(request, tmp_path / "output")

    assert "--factory-startup" in command
    assert "--disable-autoexec" in command
    assert command.count("--python") == 1
    script = Path(command[command.index("--python") + 1]).resolve()
    assert script.is_relative_to(plugin_root.resolve())
    assert script.name == "build_declarative_model.py"


def test_render_blender_command_uses_plugin_owned_script_and_safe_loading(
    tmp_path: Path,
) -> None:
    plugin_root = Path(__file__).resolve().parents[1] / "plugins" / "viewforge-3d-toolkit"
    arguments = {
        "input": str(tmp_path / "source.glb"),
        "views": ["perspective", "front"],
        "resolution": 512,
        "material_mode": "original",
        "background": "studio_dark",
    }
    glb_request = JobRequest(
        schema_version=2,
        job_id="job_render_glb",
        kind=JobKind.RENDER_MODEL_PREVIEW.value,
        blender_executable=sys.executable,
        plugin_root=str(plugin_root),
        arguments=arguments,
    )

    glb_command = blender_command(glb_request, tmp_path / "glb-output")

    assert "--background" in glb_command
    assert "--disable-autoexec" in glb_command
    assert "--factory-startup" in glb_command
    assert glb_command.count("--python") == 1
    script = Path(glb_command[glb_command.index("--python") + 1]).resolve()
    assert script.is_relative_to(plugin_root.resolve())
    assert script.name == "render_model_preview.py"

    blend_arguments = {**arguments, "input": str(tmp_path / "source.blend")}
    blend_request = JobRequest(
        schema_version=2,
        job_id="job_render_blend",
        kind=JobKind.RENDER_MODEL_PREVIEW.value,
        blender_executable=sys.executable,
        plugin_root=str(plugin_root),
        arguments=blend_arguments,
    )

    blend_command = blender_command(blend_request, tmp_path / "blend-output")

    assert "--factory-startup" not in blend_command
    assert blend_command.index(blend_arguments["input"]) < blend_command.index("--python")


def test_inline_declarative_spec_is_staged_as_immutable_job_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _ = configured_runtime(tmp_path)
    monkeypatch.setattr(runtime.configuration, "blender", lambda: Path(sys.executable))

    class FakeWorker:
        pid = 4321

    monkeypatch.setattr(
        "face3d.local_mcp.jobs.subprocess.Popen",
        lambda *args, **kwargs: FakeWorker(),
    )
    summary = runtime.launcher.build_declarative_model(
        spec={
            "schemaVersion": 1,
            "objects": [{"name": "Seat", "primitive": "cube", "scale": [1, 1, 0.1]}],
        }
    )

    request = load_request(Path(runtime.jobs.load(summary.id).request_path))
    staged = Path(request.arguments["spec"])
    assert staged.is_relative_to(runtime.jobs.directory(summary.id) / "inputs")
    assert json.loads(staged.read_text(encoding="utf-8"))["objects"][0]["name"] == "Seat"


def test_render_preview_stages_source_and_validates_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, workspace = configured_runtime(tmp_path)
    source = workspace / "model.glb"
    source.write_bytes(b"glTF render fixture")
    asset = runtime.assets.register(source.name)
    monkeypatch.setattr(runtime.configuration, "blender", lambda: Path(sys.executable))

    class FakeWorker:
        pid = 4322

    monkeypatch.setattr(
        "face3d.local_mcp.jobs.subprocess.Popen",
        lambda *args, **kwargs: FakeWorker(),
    )
    summary = runtime.launcher.render_model_preview(
        source_id=asset.id,
        views=["perspective", "front"],
        resolution=512,
        material_mode="neutral",
        background="transparent",
    )

    request = load_request(Path(runtime.jobs.load(summary.id).request_path))
    staged = Path(request.arguments["input"])
    assert staged.is_relative_to(runtime.jobs.directory(summary.id) / "inputs")
    assert staged.read_bytes() == source.read_bytes()
    assert request.arguments["views"] == ["perspective", "front"]
    assert request.arguments["resolution"] == 512
    with pytest.raises(ValueError, match="duplicates"):
        runtime.launcher.render_model_preview(
            source_id=asset.id,
            views=["front", "front"],
            resolution=512,
            material_mode="original",
            background="studio_dark",
        )


def test_render_contact_sheet_and_image_artifact_roundtrip(tmp_path: Path) -> None:
    runtime, _ = configured_runtime(tmp_path)
    job_id = "job_render_fixture"
    output = runtime.paths.jobs / job_id / "output"
    render_dir = output / "renders"
    render_dir.mkdir(parents=True)
    views = ["perspective", "front", "right"]
    for index, view in enumerate(views):
        Image.new(
            "RGB",
            (320, 320),
            (40 + index * 40, 80 + index * 20, 120),
        ).save(render_dir / f"render-{view}.png")
    sheet_path = output / "render-preview-sheet.png"

    sheet = compose_preview_sheet(render_dir, views, sheet_path)
    artifact = ArtifactStore(runtime.paths).register(job_id, sheet_path)
    result = runtime.resolve_image_artifact(artifact.id)

    assert sheet["views"] == views
    assert sheet["width"] == 960
    assert sheet_path.is_file()
    assert len(result.content) == 2
    assert isinstance(result.content[1], ImageContent)
    assert base64.b64decode(result.content[1].data) == sheet_path.read_bytes()
    assert str(runtime.paths.root) not in result.model_dump_json()


def test_read_image_artifact_rejects_non_image(tmp_path: Path) -> None:
    runtime, _ = configured_runtime(tmp_path)
    job_id = "job_json_fixture"
    output = runtime.paths.jobs / job_id / "output"
    output.mkdir(parents=True)
    document = output / "qa.json"
    document.write_text("{}", encoding="utf-8")
    artifact = ArtifactStore(runtime.paths).register(job_id, document)

    with pytest.raises(ValueError, match="Only PNG"):
        runtime.resolve_image_artifact(artifact.id)


def test_non_blender_pixel_cube_job_runs_in_modeling_runtime(tmp_path: Path) -> None:
    paths = LocalPaths(tmp_path / "state")
    paths.ensure()
    job_id = "job_pixel_cube_fixture"
    job_dir = paths.jobs / job_id
    job_dir.mkdir()
    request_path = job_dir / "request.json"
    atomic_write_json(
        request_path,
        {
            "schemaVersion": 2,
            "jobId": job_id,
            "kind": JobKind.GENERATE_PIXEL_CUBE.value,
            "runtime": "modeling",
            "blenderExecutable": None,
            "pluginRoot": None,
            "arguments": {"side_cm": 2.0, "cells_per_edge": 4},
        },
    )
    now = utc_now()
    from face3d.local_mcp.storage import JobStore

    artifacts = ArtifactStore(paths)
    jobs = JobStore(paths, artifacts)
    jobs.save(
        StoredJob(
            id=job_id,
            kind=JobKind.GENERATE_PIXEL_CUBE.value,
            state=JobState.QUEUED.value,
            created_at=now,
            updated_at=now,
            request_path=str(request_path),
        )
    )

    assert run_job(request_path, paths) == 0
    completed = jobs.public_by_id(job_id)
    assert completed.state == JobState.SUCCEEDED
    assert {artifact.extension for artifact in completed.artifacts} >= {".glb", ".json"}
