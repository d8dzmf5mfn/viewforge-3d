from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh

from face3d.phone_run import ReferenceImage, build_iphone17_unskinned_run
from face3d.phone_v1 import (
    FIXED_VIEWS,
    PhoneDimensions,
    build_phone_scene,
    build_template_phone_v0,
    primary_surface_metrics,
)


def test_iphone17_primary_surface_preserves_topology_and_metric_dimensions() -> None:
    template = build_template_phone_v0()
    fitted = template.fit_dimensions(PhoneDimensions.iphone17())

    assert np.allclose(fitted.compute_mesh.extents, [71.5, 149.6, 7.95])
    assert np.array_equal(fitted.compute_faces, template.compute_faces)
    assert np.array_equal(fitted.render_to_compute, template.render_to_compute)
    assert fitted.uv_sha256 == template.uv_sha256
    assert primary_surface_metrics(fitted)["passed"] is True


def test_thin_attached_features_keep_ordered_depth_layers() -> None:
    dimensions = PhoneDimensions.iphone17()
    fitted = build_template_phone_v0().fit_dimensions(dimensions)
    scene = build_phone_scene(fitted, dimensions)

    assert np.allclose(scene.geometry["DisplayGlass"].extents, [69.45, 147.61, 0.16])
    assert np.allclose(scene.geometry["DynamicIsland"].extents, [20.75, 5.12, 0.08])
    assert np.allclose(scene.geometry["ActionButton"].extents, [0.46, 8.4, 2.2])
    assert np.allclose(scene.geometry["UsbCReferenceInset"].extents, [12.45, 0.12, 2.2])
    attached = [mesh for name, mesh in scene.geometry.items() if name != "PhonePrimarySurface"]
    assert all(mesh.is_watertight for mesh in attached)
    assert all(mesh.is_winding_consistent for mesh in attached)


def _write_reference(path: Path, color: tuple[int, int, int]) -> None:
    image = np.empty((1024, 1100, 3), dtype=np.uint8)
    image[:] = color
    assert cv2.imwrite(str(path), image)


def test_build_iphone17_unskinned_run_is_traceable_and_immutable(tmp_path: Path) -> None:
    hero = tmp_path / "hero.jpg"
    side = tmp_path / "side.jpg"
    _write_reference(hero, (210, 190, 225))
    _write_reference(side, (190, 180, 210))
    output = tmp_path / "iphone17-run"
    references = (
        ReferenceImage("hero", "https://example.test/hero.jpg", hero, "shape-reference"),
        ReferenceImage("side", "https://example.test/side.jpg", side, "side-reference"),
    )

    manifest = build_iphone17_unskinned_run(
        output,
        references=references,
        render_size=128,
        generated_at=datetime(2026, 8, 15, tzinfo=UTC),
    )

    assert manifest["artifactType"] == "geometry-preview"
    assert manifest["external3dImported"] is False
    assert manifest["skinApplied"] is False
    assert manifest["imageTexturesApplied"] is False
    assert manifest["acceptance"]["userSignoff"] is False
    assert (output / manifest["model"]).stat().st_size > 10_000
    quality = json.loads((output / manifest["quality"]).read_text())
    assert quality["passed"] is True
    assert quality["glbRoundtrip"]["noImageTextures"] is True
    scene = trimesh.load(output / manifest["model"], force="scene", process=False)
    assert len(scene.geometry) == 16
    assert len(list((output / "qa").glob("fixed-view-*.png"))) == len(FIXED_VIEWS)
    with pytest.raises(FileExistsError, match="immutable run destination"):
        build_iphone17_unskinned_run(output, references=references, render_size=128)
