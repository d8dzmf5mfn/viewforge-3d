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
    skeleton_schema = by_name["build_biological_skeleton"].input_schema
    assert skeleton_schema["required"] == ["source_id"]
    assert {"landmarks", "component_map"} <= set(skeleton_schema["properties"])
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


def test_local_edition_adds_only_local_workbench_tools(tmp_path: Path) -> None:
    chat_runtime, _ = configured_runtime(tmp_path)
    local_runtime = Runtime(chat_runtime.paths, edition="local")

    chat_tools = anyio.run(create_server(chat_runtime).list_tools)
    local_tools = anyio.run(create_server(local_runtime).list_tools)
    chat_names = {tool.name for tool in chat_tools}
    local_names = {tool.name for tool in local_tools}

    assert local_names == chat_names | {
        "smooth_model_surface",
        "get_local_artifact_path",
    }
    assert "smooth_model_surface" not in chat_names
    assert "get_local_artifact_path" not in chat_names
    local_by_name = {tool.name: tool for tool in local_tools}
    smooth_schema = local_by_name["smooth_model_surface"].input_schema
    assert smooth_schema["required"] == ["source"]
    assert smooth_schema["properties"]["preserve_volume"]["default"] is True
    assert local_by_name["get_local_artifact_path"].annotations.read_only_hint is True
    status = local_runtime.status()
    assert status.edition == "local"
    assert "topology_preserving_smoothing" in status.capabilities


def test_local_artifact_path_is_not_available_to_chat_edition(tmp_path: Path) -> None:
    chat_runtime, _ = configured_runtime(tmp_path)
    job_id = "job_local_path_fixture"
    output = chat_runtime.paths.jobs / job_id / "output"
    output.mkdir(parents=True)
    model = output / "smoothed-model.glb"
    model.write_bytes(b"glTF local path fixture")
    artifact = ArtifactStore(chat_runtime.paths).register(job_id, model)
    local_runtime = Runtime(chat_runtime.paths, edition="local")

    location = local_runtime.resolve_local_artifact_path(artifact.id)

    assert location.path == str(model.resolve())
    with pytest.raises(RuntimeError, match="local edition"):
        chat_runtime.resolve_local_artifact_path(artifact.id)


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
    assert command[command.index("--python-exit-code") + 1] == "1"
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
    assert command[command.index("--python-exit-code") + 1] == "1"
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
    assert glb_command[glb_command.index("--python-exit-code") + 1] == "1"
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


def test_surface_smooth_blender_command_uses_local_plugin_script(tmp_path: Path) -> None:
    plugin_root = Path(__file__).resolve().parents[1] / "plugins" / "viewforge-3d-local"
    request = JobRequest(
        schema_version=2,
        job_id="job_smooth_glb",
        kind=JobKind.SMOOTH_MODEL_SURFACE.value,
        blender_executable=sys.executable,
        plugin_root=str(plugin_root),
        arguments={
            "input": str(tmp_path / "source.glb"),
            "options": str(tmp_path / "smoothing-options.json"),
        },
    )

    command = blender_command(request, tmp_path / "smooth-output")

    assert "--factory-startup" in command
    assert "--disable-autoexec" in command
    assert command[command.index("--python-exit-code") + 1] == "1"
    script = Path(command[command.index("--python") + 1]).resolve()
    assert script.is_relative_to(plugin_root.resolve())
    assert script.name == "smooth_model_surface.py"


def test_skeleton_blend_command_loads_source_before_worker_script(tmp_path: Path) -> None:
    plugin_root = Path(__file__).resolve().parents[1] / "plugins" / "viewforge-3d-toolkit"
    source = tmp_path / "character.blend"
    request = JobRequest(
        schema_version=2,
        job_id="job_skeleton_blend",
        kind=JobKind.BUILD_SKELETON.value,
        blender_executable=sys.executable,
        plugin_root=str(plugin_root),
        arguments={"input": str(source), "profile": "humanoid-v1"},
    )

    command = blender_command(request, tmp_path / "output")

    assert "--factory-startup" not in command
    assert command.index(str(source)) < command.index("--python")
    assert command[command.index("--python-exit-code") + 1] == "1"


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


def test_skeleton_accepts_generated_blend_artifact_and_inline_landmarks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _ = configured_runtime(tmp_path)
    source_job_id = "job_generated_model"
    source_output = runtime.paths.jobs / source_job_id / "output"
    source_output.mkdir(parents=True)
    source = source_output / "declarative-model.blend"
    source.write_bytes(b"generated blend fixture")
    source_artifact = runtime.artifacts.register(source_job_id, source)
    monkeypatch.setattr(runtime.configuration, "blender", lambda: Path(sys.executable))

    class FakeWorker:
        pid = 4323

    monkeypatch.setattr(
        "face3d.local_mcp.jobs.subprocess.Popen",
        lambda *args, **kwargs: FakeWorker(),
    )
    landmarks = {
        "schemaVersion": 1,
        "profileId": "humanoid-v1",
        "landmarks": {"fixture": [0, 0, 0]},
    }

    summary = runtime.launcher.build_skeleton(
        source_id=source_artifact.id,
        profile="humanoid-v1",
        landmarks=landmarks,
    )

    request = load_request(Path(runtime.jobs.load(summary.id).request_path))
    staged_source = Path(request.arguments["input"])
    staged_landmarks = Path(request.arguments["landmarks"])
    assert staged_source.suffix == ".blend"
    assert staged_source.read_bytes() == source.read_bytes()
    assert json.loads(staged_landmarks.read_text(encoding="utf-8")) == landmarks
    assert staged_source.is_relative_to(runtime.jobs.directory(summary.id) / "inputs")
    assert staged_landmarks.is_relative_to(runtime.jobs.directory(summary.id) / "inputs")


@pytest.mark.parametrize(
    ("kind", "inline_name", "inline_document"),
    [
        (
            "animation",
            "coordinates",
            {"schemaVersion": 1, "frames": []},
        ),
        (
            "binding",
            "mapping",
            {"schemaVersion": 1, "components": {}},
        ),
    ],
)
def test_animation_and_binding_accept_inline_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    inline_name: str,
    inline_document: dict[str, object],
) -> None:
    runtime, workspace = configured_runtime(tmp_path)
    blend = workspace / "bone-only.blend"
    skeleton = workspace / "skeleton.json"
    blend.write_bytes(b"blend fixture")
    skeleton.write_text("{}", encoding="utf-8")
    blend_asset = runtime.assets.register(blend.name)
    skeleton_asset = runtime.assets.register(skeleton.name)
    monkeypatch.setattr(runtime.configuration, "blender", lambda: Path(sys.executable))

    class FakeWorker:
        pid = 4324

    monkeypatch.setattr(
        "face3d.local_mcp.jobs.subprocess.Popen",
        lambda *args, **kwargs: FakeWorker(),
    )
    if kind == "animation":
        summary = runtime.launcher.create_animation(
            input_blend_id=blend_asset.id,
            skeleton_id=skeleton_asset.id,
            coordinates=inline_document,
        )
    else:
        summary = runtime.launcher.bind_components(
            input_blend_id=blend_asset.id,
            skeleton_id=skeleton_asset.id,
            mapping=inline_document,
        )

    request = load_request(Path(runtime.jobs.load(summary.id).request_path))
    staged = Path(request.arguments[inline_name])
    assert json.loads(staged.read_text(encoding="utf-8")) == inline_document
    assert staged.is_relative_to(runtime.jobs.directory(summary.id) / "inputs")


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


def test_surface_smooth_accepts_workspace_path_and_stages_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, workspace = configured_runtime(tmp_path)
    source = workspace / "rough-model.glb"
    source.write_bytes(b"glTF smoothing fixture")
    configuration = runtime.configuration.load()
    runtime.configuration.save(
        LocalConfiguration(
            workspace_root=configuration.workspace_root,
            blender_executable=configuration.blender_executable,
            plugin_root=str(
                Path(__file__).resolve().parents[1] / "plugins" / "viewforge-3d-local"
            ),
        )
    )
    monkeypatch.setattr(runtime.configuration, "blender", lambda: Path(sys.executable))

    class FakeWorker:
        pid = 4325

    monkeypatch.setattr(
        "face3d.local_mcp.jobs.subprocess.Popen",
        lambda *args, **kwargs: FakeWorker(),
    )

    summary = runtime.launcher.smooth_model_surface(
        source=str(source),
        object_names=["Head"],
        vertex_group="PolishMask",
        iterations=4,
        strength=0.25,
        preserve_volume=True,
        preserve_boundaries=True,
        max_displacement_ratio=0.01,
        shade_smooth=True,
    )

    request = load_request(Path(runtime.jobs.load(summary.id).request_path))
    staged_source = Path(request.arguments["input"])
    staged_options = Path(request.arguments["options"])
    options = json.loads(staged_options.read_text(encoding="utf-8"))
    assert staged_source.is_relative_to(runtime.jobs.directory(summary.id) / "inputs")
    assert staged_source.read_bytes() == source.read_bytes()
    assert options["objectNames"] == ["Head"]
    assert options["vertexGroup"] == "PolishMask"
    assert options["iterations"] == 4
    assert options["preserveVolume"] is True
    assert options["maxDisplacementRatio"] == 0.01


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


def test_failed_blender_job_emits_readable_sanitized_error_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _ = configured_runtime(tmp_path)
    job_id = "job_failed_blender_fixture"
    job_dir = runtime.paths.jobs / job_id
    job_dir.mkdir()
    request_path = job_dir / "request.json"
    atomic_write_json(
        request_path,
        {
            "schemaVersion": 2,
            "jobId": job_id,
            "kind": JobKind.BUILD_SKELETON.value,
            "runtime": "blender",
            "blenderExecutable": sys.executable,
            "pluginRoot": str(
                Path(__file__).resolve().parents[1] / "plugins" / "viewforge-3d-toolkit"
            ),
            "arguments": {"input": str(job_dir / "source.blend"), "profile": "humanoid-v1"},
        },
    )
    runtime.jobs.save(
        StoredJob(
            id=job_id,
            kind=JobKind.BUILD_SKELETON.value,
            state=JobState.QUEUED.value,
            created_at=utc_now(),
            updated_at=utc_now(),
            request_path=str(request_path),
        )
    )
    diagnostic_line = f"ValueError: missing neck in {runtime.paths.root}/jobs/private"
    monkeypatch.setattr(
        "face3d.local_mcp.jobs.blender_command",
        lambda request, output_dir: [
            sys.executable,
            "-c",
            f"import sys; print({diagnostic_line!r}); sys.exit(7)",
        ],
    )

    assert run_job(request_path, runtime.paths) == 1
    completed = runtime.jobs.public_by_id(job_id)
    error_artifact = next(item for item in completed.artifacts if item.name == "error.json")
    diagnostic = runtime.resolve_json_artifact(error_artifact.id)
    serialized = diagnostic.model_dump_json()

    assert completed.state == JobState.FAILED
    assert "Read the sanitized error.json" in (completed.failure or "")
    assert diagnostic.document["exitCode"] == 7
    assert "missing neck" in diagnostic.document["workerLogTail"]
    assert str(runtime.paths.root) not in serialized


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
