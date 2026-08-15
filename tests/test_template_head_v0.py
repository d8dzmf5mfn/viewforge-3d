import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from face3d.errors import Face3DError
from face3d.template_head_v0 import (
    RawTemplateHeadV0,
    _unify_compute_mesh,
    prepare_template_head_v0,
)


def _camera(role: str, yaw: float) -> dict[str, object]:
    role_by_yaw = {0.0: "front", -45.0: "left45", 45.0: "right45"}
    return {
        "role": role_by_yaw[yaw] if role == "side" else role,
        "width": 512,
        "height": 512,
        "focal_length_px": 520.0,
        "principal_point_px": [256.0, 256.0],
        "rotation_vector": [float(np.pi), 0.0, 0.0],
        "translation": [0.0, 0.0, 3.0],
        "yaw_deg": yaw,
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
    }


def _fixture_package(
    path: Path,
    *,
    offset_render: bool = False,
    include_skin: bool = True,
) -> None:
    compute = trimesh.creation.icosphere(subdivisions=1, radius=0.8)
    compute_vertices = np.asarray(compute.vertices, dtype=np.float64)
    compute_faces = np.asarray(compute.faces, dtype=np.int64)
    render_vertices = np.concatenate((compute_vertices, compute_vertices[:1]), axis=0)
    render_faces = compute_faces.copy()
    first_face_with_zero = int(np.flatnonzero(np.any(render_faces == 0, axis=1))[0])
    corner = int(np.flatnonzero(render_faces[first_face_with_zero] == 0)[0])
    render_faces[first_face_with_zero, corner] = len(compute_vertices)
    if offset_render:
        render_vertices[-1, 0] += 0.05
    uv = np.column_stack(
        (
            0.5 + np.arctan2(render_vertices[:, 0], render_vertices[:, 2]) / (2 * np.pi),
            0.5
            - np.arcsin(
                np.clip(render_vertices[:, 1] / 0.8, -1.0, 1.0)
            )
            / np.pi,
        )
    ).astype(np.float32)
    uv[-1] = np.asarray([1.1, 0.5], dtype=np.float32)
    render = trimesh.Trimesh(
        vertices=render_vertices,
        faces=render_faces,
        process=False,
        validate=False,
    )
    render.visual = trimesh.visual.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=(0.7, 0.6, 0.5, 1.0)
        ),
    )
    smooth_glb = trimesh.exchange.gltf.export_glb(compute, include_normals=True)
    skin_glb = trimesh.exchange.gltf.export_glb(render, include_normals=True)
    manifest = {
        "schemaVersion": "1.0.0",
        "cameras": [
            _camera("front", 0.0),
            _camera("left45", -45.0),
            _camera("right45", 45.0),
        ],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("models/smooth.glb", smooth_glb)
        if include_skin:
            archive.writestr("models/skin.glb", skin_glb)


def test_prepare_template_head_v0_preserves_compute_and_uv_topology(tmp_path: Path) -> None:
    package = tmp_path / "fixture.face3d"
    _fixture_package(package)
    baseline = tmp_path / "baseline.png"
    Image.new("RGB", (320, 180), (235, 235, 235)).save(baseline)
    output = tmp_path / "template-head-v0"

    result = prepare_template_head_v0(package, baseline, output)

    assert result["ok"] is True
    assert result["state"] == "raw-extracted"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["geometry"]["componentCount"] == 1
    assert manifest["geometry"]["boundaryEdgeCount"] == 0
    assert manifest["geometry"]["nonManifoldEdgeCount"] == 0
    assert manifest["renderTopology"]["maximumPositionDifference"] == 0.0
    assert manifest["uv"]["seamDuplicateVertexCount"] == 1
    assert manifest["uv"]["state"] == "source-uv-requires-normalization"
    assert manifest["readiness"]["stableUvReady"] is False
    assert manifest["route"]["sdfRole"] == "qa-only"
    assert (output / "reference" / "quality-baseline.png").is_file()
    assert set(manifest["artifacts"]["fixedViews"]) == {
        "front",
        "left45",
        "right45",
        "side",
    }

    restored = RawTemplateHeadV0.load(
        output / "template" / "template-head-v0.raw.npz"
    )
    assert len(restored.compute_vertices) + 1 == len(restored.render_to_compute)
    assert np.array_equal(
        restored.render_to_compute[restored.render_faces],
        restored.compute_faces,
    )


def test_prepare_template_head_v0_rejects_render_geometry_drift(tmp_path: Path) -> None:
    package = tmp_path / "mismatch.face3d"
    _fixture_package(package, offset_render=True)
    baseline = tmp_path / "baseline.png"
    Image.new("RGB", (32, 32), (200, 200, 200)).save(baseline)

    with pytest.raises(Face3DError) as captured:
        prepare_template_head_v0(package, baseline, tmp_path / "output")

    assert captured.value.code == "template-render-position-mismatch"


def test_prepare_template_head_v0_generates_stable_uv_when_skin_is_absent(
    tmp_path: Path,
) -> None:
    package = tmp_path / "smooth-only.face3d"
    _fixture_package(package, include_skin=False)
    baseline = tmp_path / "baseline.png"
    Image.new("RGB", (32, 32), (200, 200, 200)).save(baseline)
    output = tmp_path / "output"

    result = prepare_template_head_v0(package, baseline, output)

    manifest = json.loads((output / "manifest.json").read_text())
    assert result["uvMethod"] == "xatlas-0.0.11"
    assert manifest["uv"]["method"] == "xatlas-0.0.11"
    assert manifest["uv"]["state"] == "source-uv-extracted"
    assert manifest["readiness"]["stableUvReady"] is True
    assert "sourceRender" not in manifest["artifacts"]


def test_unify_compute_mesh_records_non_sdf_exact_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    second = first.copy()
    second.apply_translation((0.5, 0.0, 0.0))
    source = trimesh.util.concatenate((first, second))
    expected = trimesh.creation.icosphere(subdivisions=2, radius=1.1)
    captured: dict[str, object] = {}

    def fake_union(
        meshes: list[trimesh.Trimesh],
        *,
        engine: str,
        check_volume: bool,
    ) -> trimesh.Trimesh:
        captured.update(
            {
                "componentCount": len(meshes),
                "engine": engine,
                "checkVolume": check_volume,
            }
        )
        return expected

    monkeypatch.setattr(trimesh.boolean, "union", fake_union)

    unified, preparation = _unify_compute_mesh(source)

    assert unified is expected
    assert captured == {
        "componentCount": 2,
        "engine": "blender",
        "checkVolume": True,
    }
    assert preparation["method"] == "blender-exact-boolean-union"
    assert preparation["sdfUsed"] is False
    assert preparation["input"]["componentCount"] == 2
    assert preparation["output"]["componentCount"] == 1


def test_prepare_template_head_v0_accepts_licensed_direct_glb(tmp_path: Path) -> None:
    package = tmp_path / "source.face3d"
    _fixture_package(package)
    source_glb = tmp_path / "source.glb"
    with zipfile.ZipFile(package) as archive:
        source_glb.write_bytes(archive.read("models/skin.glb"))
    license_path = tmp_path / "LICENSE.txt"
    license_path.write_text("CC BY 3.0 test fixture\n")
    baseline = tmp_path / "baseline.png"
    Image.new("RGB", (32, 32), (200, 200, 200)).save(baseline)
    output = tmp_path / "output"

    result = prepare_template_head_v0(
        None,
        baseline,
        output,
        source_glb=source_glb,
        source_license=license_path,
    )

    manifest = json.loads((output / "manifest.json").read_text())
    assert result["ok"] is True
    assert manifest["source"]["kind"] == "direct-glb"
    assert manifest["source"]["uvPolicy"] == (
        "regenerate-stable-xatlas-from-compute-topology"
    )
    assert manifest["topologyPreparation"]["method"] == (
        "exact-position-uv-seam-weld"
    )
    assert manifest["topologyPreparation"]["weldedDuplicateVertexCount"] == 1
    assert manifest["topologyPreparation"]["maximumPositionDifference"] == 0.0
    assert manifest["uv"]["method"] == "xatlas-0.0.11"
    assert manifest["readiness"]["stableUvReady"] is True
    assert (output / "source" / "original.glb").is_file()
    assert (output / "source" / "LICENSE.txt").is_file()


def test_raw_template_hash_fails_closed(tmp_path: Path) -> None:
    package = tmp_path / "fixture.face3d"
    _fixture_package(package)
    baseline = tmp_path / "baseline.png"
    Image.new("RGB", (32, 32), (200, 200, 200)).save(baseline)
    output = tmp_path / "template"
    prepare_template_head_v0(package, baseline, output)
    source = output / "template" / "template-head-v0.raw.npz"
    with np.load(source, allow_pickle=False) as payload:
        values = {name: payload[name] for name in payload.files}
    values["compute_vertices"] = values["compute_vertices"].copy()
    values["compute_vertices"][0, 0] += 0.01
    encoded = io.BytesIO()
    np.savez_compressed(encoded, **values)
    source.write_bytes(encoded.getvalue())

    with pytest.raises(Face3DError) as captured:
        RawTemplateHeadV0.load(source)

    assert captured.value.code == "template-geometry-hash-mismatch"
