from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import trimesh

from face3d.errors import Face3DError
from face3d.models import CameraRecord, ViewRole
from face3d.template_head_anatomy import (
    EYELID_CONTOUR_LANDMARKS,
    LandmarkSurfaceMap,
    _cleanup_boolean_result,
    _ordered_simple_cycle,
    _relative_triangle_area_metrics,
    _stabilize_boolean_topology,
    _top_curvature_spike_ratio,
    derive_eye_specs,
    eyelid_contour_symmetry,
    fitting_eyelid_contour,
    rebind_template_head_v0_eyelids,
    remap_landmarks_to_anatomy,
)
from face3d.template_head_v0 import _repair_degenerate_xatlas_charts
from face3d.unified_head import UnifiedHeadAsset


def _surface_map() -> LandmarkSurfaceMap:
    count = 478
    normalized = np.full((count, 3), 0.5, dtype=np.float64)
    points = np.zeros((count, 3), dtype=np.float64)
    valid = np.ones(count, dtype=bool)
    # Anatomical right (observer left) and anatomical left (observer right).
    points[33] = (-0.95, 1.64, 1.84)
    points[133] = (-0.43, 1.66, 1.84)
    points[159] = (-0.69, 1.68, 1.93)
    points[145] = (-0.69, 1.59, 1.89)
    points[263] = (0.74, 1.65, 1.82)
    points[362] = (0.21, 1.67, 1.84)
    points[386] = (0.48, 1.69, 1.94)
    points[374] = (0.48, 1.60, 1.90)
    return LandmarkSurfaceMap(
        normalized=normalized,
        points=points,
        valid=valid,
        triangle=np.zeros(count, dtype=np.uint32),
        barycentric=np.tile(np.asarray([1.0, 0.0, 0.0]), (count, 1)),
        reprojection_error_px=np.zeros(count, dtype=np.float64),
    )


def test_derive_eye_specs_uses_anatomical_sides_and_shared_full_radius() -> None:
    specs = derive_eye_specs(_surface_map())

    assert specs["left"].center[0] > 0
    assert specs["right"].center[0] < 0
    assert specs["left"].eyeball_radius == specs["right"].eyeball_radius
    assert specs["left"].eyeball_radius > 0.25
    assert specs["left"].socket_radius / specs["left"].eyeball_radius == pytest.approx(1.029)
    assert specs["left"].slot_axes[1] < specs["left"].slot_axes[0] * 0.25
    assert specs["left"].slot_axes[2] > specs["left"].eyeball_radius * 2.5


def test_full_eyelid_fitting_contours_are_balanced_without_moving_geometry() -> None:
    source = _surface_map()
    vertices: list[np.ndarray] = []
    for side, center, radii in (
        ("right", (-1.0, 1.6, 1.9), (0.50, 0.10, 0.08)),
        ("left", (1.0, 1.6, 1.9), (0.48, 0.10, 0.08)),
    ):
        angles = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
        contour = np.column_stack(
            (
                center[0] + np.cos(angles) * radii[0],
                center[1] + np.sin(angles) * radii[1],
                center[2] + np.sin(angles) * radii[2],
            )
        )
        vertices.extend(contour)
        indices = np.asarray(EYELID_CONTOUR_LANDMARKS[side], dtype=np.int64)
        source.points[indices] = contour[
            np.linspace(0, len(contour), len(indices), endpoint=False, dtype=np.int64)
        ]
    vertices_array = np.asarray(vertices, dtype=np.float64)
    before = vertices_array.copy()

    rings: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, object]] = {}
    for side in ("right", "left"):
        rings[side], metrics[side] = fitting_eyelid_contour(
            vertices_array,
            source,
            side,
        )

    symmetry = eyelid_contour_symmetry(metrics)
    assert symmetry["passed"] is True
    assert symmetry["horizontalSpanRatio"] >= 0.90
    assert symmetry["verticalSpanRatio"] >= 0.80
    assert len(rings["left"]) >= 16
    assert len(rings["right"]) >= 16
    assert np.array_equal(vertices_array, before)


def test_eyelid_contour_symmetry_fails_closed() -> None:
    metrics = {
        "left": {"boundSpan": [0.5, 0.1, 0.1]},
        "right": {"boundSpan": [0.3, 0.1, 0.1]},
    }
    with pytest.raises(Face3DError) as captured:
        eyelid_contour_symmetry(metrics)

    assert captured.value.code == "template-eyelid-contour-asymmetry"


def test_ordered_simple_cycle_returns_each_vertex_once() -> None:
    cycle = _ordered_simple_cycle(np.asarray([[3, 7], [1, 3], [9, 1], [7, 9]], dtype=np.int64))

    assert cycle[0] == 1
    assert set(cycle.tolist()) == {1, 3, 7, 9}
    assert len(cycle) == 4


def test_ordered_simple_cycle_rejects_branch() -> None:
    with pytest.raises(ValueError):
        _ordered_simple_cycle(np.asarray([[0, 1], [1, 2], [2, 0], [1, 3]], dtype=np.int64))


def test_boolean_cleanup_discards_only_tiny_zero_volume_artifact() -> None:
    main = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    artifact = trimesh.Trimesh(
        vertices=np.asarray([[3.0, 0.0, 0.0], [3.0001, 0.0, 0.0], [3.0, 0.0001, 0.0]]),
        faces=np.asarray([[0, 1, 2]]),
        process=False,
    )

    cleaned, metrics = _cleanup_boolean_result(trimesh.util.concatenate((main, artifact)))

    assert cleaned.is_volume
    assert cleaned.is_watertight
    assert metrics.discarded_component_count == 1
    assert metrics.discarded_face_count == 1


def test_boolean_cleanup_rejects_second_material_component() -> None:
    main = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    second = trimesh.creation.icosphere(subdivisions=1, radius=0.4)
    second.apply_translation((3.0, 0.0, 0.0))

    with pytest.raises(Face3DError) as captured:
        _cleanup_boolean_result(trimesh.util.concatenate((main, second)))

    assert captured.value.code == "template-eye-boolean-artifacts"


def test_topology_stabilization_removes_sliver_faces_deterministically() -> None:
    source = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    vertices = np.asarray(source.vertices, dtype=np.float64).copy()
    faces = np.asarray(source.faces, dtype=np.int64)
    first, second = faces[0, :2]
    vertices[second] = vertices[first] + (vertices[second] - vertices[first]) * 1e-5
    source = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)

    first_result, first_metrics = _stabilize_boolean_topology(
        source,
        target_face_count=1200,
        maximum_surface_p99_diagonal=0.02,
        maximum_surface_distance_diagonal=0.04,
    )
    second_result, second_metrics = _stabilize_boolean_topology(
        source,
        target_face_count=1200,
        maximum_surface_p99_diagonal=0.02,
        maximum_surface_distance_diagonal=0.04,
    )

    before = _relative_triangle_area_metrics(vertices, faces)
    after = _relative_triangle_area_metrics(first_result.vertices, first_result.faces)
    assert before["minimumRelativeArea"] < 1e-4
    assert after["minimumRelativeArea"] >= 5e-4
    assert first_metrics.self_intersection_pair_count_after == 0
    assert first_metrics.float32_self_intersection_pair_count == 0
    assert first_result.is_watertight
    assert first_result.is_winding_consistent
    assert np.array_equal(first_result.vertices, second_result.vertices)
    assert np.array_equal(first_result.faces, second_result.faces)
    assert first_metrics == second_metrics


def test_top_curvature_gate_accepts_rounded_cranium() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    assert _top_curvature_spike_ratio(sphere) <= 4.0


def test_degenerate_xatlas_charts_are_repaired_without_geometry_drift() -> None:
    compute_faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    render_to_compute = np.arange(4, dtype=np.int64)
    render_faces = compute_faces.copy()
    uv = np.asarray(
        [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.5, 0.5]],
        dtype=np.float64,
    )

    mapping, faces, repaired_uv, metrics = _repair_degenerate_xatlas_charts(
        compute_faces,
        render_to_compute,
        render_faces,
        uv,
        maximum_repair_fraction=0.6,
    )
    triangles = repaired_uv[faces]
    doubled_area = np.abs(
        (triangles[:, 1, 0] - triangles[:, 0, 0]) * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 1, 1] - triangles[:, 0, 1]) * (triangles[:, 2, 0] - triangles[:, 0, 0])
    )

    assert metrics["repairedFaceCount"] == 1
    assert np.all(doubled_area > 1e-12)
    assert np.array_equal(mapping[faces], compute_faces)
    assert np.all((repaired_uv >= 0.0) & (repaired_uv <= 1.0))


def test_landmarks_are_rebound_to_post_surgery_faces_and_iris_is_excluded() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=2.0)
    source = _surface_map()
    camera = CameraRecord(
        role=ViewRole.FRONT,
        width=1024,
        height=1024,
        focal_length_px=900.0,
        principal_point_px=(512.0, 512.0),
        rotation_vector=(0.0, 0.0, 0.0),
        translation=(0.0, 0.0, 6.0),
        pitch_deg=0.0,
        yaw_deg=0.0,
        roll_deg=0.0,
    )
    rings = {
        "right": np.asarray([0, 1, 2, 3], dtype=np.int64),
        "left": np.asarray([4, 5, 6, 7], dtype=np.int64),
    }

    rebound, offset = remap_landmarks_to_anatomy(source, mesh, camera, rings)
    valid = np.flatnonzero(rebound.valid)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    reconstructed = np.einsum(
        "nvc,nv->nc",
        vertices[faces[rebound.triangle[valid]]],
        rebound.barycentric[valid],
    )

    assert not np.any(rebound.valid[468:478])
    assert np.all(rebound.triangle[valid] < len(faces))
    assert np.allclose(reconstructed, rebound.points[valid], atol=1e-7)
    assert np.isfinite(offset[valid]).all()


def test_eyelid_rebind_preserves_current_template_geometry_and_uv(tmp_path: Path) -> None:
    source = Path("assets/template-head-v0")
    if not source.is_dir():
        pytest.skip("optional local TemplateHeadV0 assets are not installed")
    copied = tmp_path / "template-head-v0"
    shutil.copytree(source, copied)
    unified_path = copied / "anatomy" / "template-head-v0.unified.npz"
    before = UnifiedHeadAsset.load(unified_path)

    result = rebind_template_head_v0_eyelids(copied)
    after = UnifiedHeadAsset.load(unified_path)

    assert result["geometryChanged"] is False
    assert result["uvChanged"] is False
    assert result["contourSymmetry"]["passed"] is True
    assert np.array_equal(before.skin_vertices, after.skin_vertices)
    assert np.array_equal(before.skin_faces, after.skin_faces)
    assert np.array_equal(before.render_to_skin, after.render_to_skin)
    assert np.array_equal(before.render_faces, after.render_faces)
    assert np.array_equal(before.uv, after.uv)
    assert after.anatomy["eyes"]["contourSymmetry"]["passed"] is True
    assert after.anatomy["eyes"]["left"]["interfaceRingVertexCount"] == 22
    assert after.anatomy["eyes"]["right"]["interfaceRingVertexCount"] == 65
