from pathlib import Path

import cv2
import numpy as np
import trimesh

import face3d.multiview_pixels as multiview_pixels
from face3d.models import CameraRecord, ViewRole
from face3d.multiview_pixels import (
    foreground_mask_from_background,
    photometric_feature_relief,
    reconstruct_multiview_pixel_cells,
    reconstruct_multiview_pixel_surface,
)
from face3d.render import render_flat_mesh


def _camera(role: ViewRole, yaw_degrees: float) -> CameraRecord:
    rotation = trimesh.transformations.rotation_matrix(np.pi, (1, 0, 0))[:3, :3]
    rotation = (
        rotation
        @ trimesh.transformations.rotation_matrix(np.radians(yaw_degrees), (0, 1, 0))[:3, :3]
    )
    rotation_vector, _ = cv2.Rodrigues(rotation)
    return CameraRecord(
        role=role,
        width=256,
        height=256,
        focal_length_px=380,
        principal_point_px=(128, 128),
        rotation_vector=tuple(float(value) for value in rotation_vector.reshape(3)),
        translation=(0, 0, 4.0),
        yaw_deg=yaw_degrees,
        pitch_deg=0,
        roll_deg=0,
    )


def _hidden_reference() -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
    vertices = np.asarray(mesh.vertices).copy()
    vertices[:, 0] *= 0.70
    vertices[:, 1] *= 1.02
    vertices[:, 2] *= 0.74
    nose = np.exp(-((vertices[:, 0] / 0.16) ** 2 + ((vertices[:, 1] - 0.02) / 0.25) ** 2))
    vertices[:, 2] -= 0.24 * nose
    mesh.vertices = vertices
    return mesh


def _rgb24(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.uint32)
    return (values[..., 0] << 16) | (values[..., 1] << 8) | values[..., 2]


def test_three_images_reconstruct_traceable_closed_surface(tmp_path: Path) -> None:
    hidden_reference = _hidden_reference()
    cameras = {
        ViewRole.FRONT: _camera(ViewRole.FRONT, 0),
        ViewRole.LEFT45: _camera(ViewRole.LEFT45, -45),
        ViewRole.RIGHT45: _camera(ViewRole.RIGHT45, 45),
    }
    images: dict[ViewRole, np.ndarray] = {}
    masks: dict[ViewRole, np.ndarray] = {}
    for role, camera in cameras.items():
        path = tmp_path / f"{role.value}.png"
        render_flat_mesh(hidden_reference, camera, path, width=256, height=256)
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert bgr is not None
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images[role] = image
        masks[role] = foreground_mask_from_background(image)

    feature_map, relief_map = photometric_feature_relief(
        images[ViewRole.FRONT],
        masks[ViewRole.FRONT],
        maximum_offset=0.01,
    )
    reconstruction = reconstruct_multiview_pixel_surface(
        images,
        masks,
        cameras,
        resolution=52,
        target_triangles=20_000,
        feature_map=feature_map,
        relief_map=relief_map,
    )

    assert 1_000 < len(reconstruction.positions) < 150_000
    assert np.all(reconstruction.source_bits == 7)
    assert np.all(np.isfinite(reconstruction.positions))
    assert reconstruction.smooth_mesh.is_watertight
    assert reconstruction.metrics["representation"] == ("three-view-2d-silhouette-depth-envelope")
    assert reconstruction.metrics["boundaryEdges"] == 0
    all_input_codes = np.unique(
        np.concatenate([_rgb24(image).reshape(-1) for image in images.values()])
    )
    assert np.all(np.isin(reconstruction.pixel_codes, all_input_codes))
    assert np.count_nonzero(reconstruction.feature_class) > 0


def test_pixel_cells_stop_before_continuous_mesh(
    tmp_path: Path, monkeypatch: object
) -> None:
    hidden_reference = _hidden_reference()
    cameras = {
        ViewRole.FRONT: _camera(ViewRole.FRONT, 0),
        ViewRole.LEFT45: _camera(ViewRole.LEFT45, -45),
        ViewRole.RIGHT45: _camera(ViewRole.RIGHT45, 45),
    }
    images: dict[ViewRole, np.ndarray] = {}
    masks: dict[ViewRole, np.ndarray] = {}
    for role, camera in cameras.items():
        path = tmp_path / f"pixel-only-{role.value}.png"
        render_flat_mesh(hidden_reference, camera, path, width=256, height=256)
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert bgr is not None
        images[role] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        masks[role] = foreground_mask_from_background(images[role])

    def forbidden_smooth_mesh(*args: object, **kwargs: object) -> trimesh.Trimesh:
        raise AssertionError("continuous mesh must not run during the Pixel stage")

    monkeypatch.setattr(multiview_pixels, "_marching_cubes_mesh", forbidden_smooth_mesh)
    cells = reconstruct_multiview_pixel_cells(
        images,
        masks,
        cameras,
        resolution=40,
        inferred_roles=frozenset({ViewRole.LEFT45, ViewRole.RIGHT45}),
    )

    assert len(cells.positions) > 500
    assert np.all(cells.source_bits == 15)
    assert cells.metrics["smoothingApplied"] is False
    assert cells.metrics["continuousMeshGenerated"] is False
    assert cells.metrics["isolatedVoxelCount"] == 0
    assert cells.metrics["traceabilityComplete"] is True
