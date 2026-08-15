from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import trimesh

from face3d.models import CameraRecord


def render_flat_mesh(
    mesh: trimesh.Trimesh,
    camera: CameraRecord,
    destination: Path,
    *,
    width: int = 720,
    height: int = 720,
    use_mesh_face_colors: bool = False,
) -> None:
    rotation, _ = cv2.Rodrigues(np.asarray(camera.rotation_vector, dtype=np.float64))
    camera_vertices = np.asarray(mesh.vertices) @ rotation.T + np.asarray(camera.translation)
    scale_x = width / camera.width
    scale_y = height / camera.height
    focal_x = camera.focal_length_px * scale_x
    focal_y = camera.focal_length_px * scale_y
    principal = np.asarray(
        [camera.principal_point_px[0] * scale_x, camera.principal_point_px[1] * scale_y]
    )
    z = camera_vertices[:, 2]
    pixels = np.zeros((len(camera_vertices), 2), dtype=np.float64)
    valid = z > 1e-6
    pixels[valid, 0] = camera_vertices[valid, 0] / z[valid] * focal_x + principal[0]
    pixels[valid, 1] = camera_vertices[valid, 1] / z[valid] * focal_y + principal[1]
    all_faces = np.asarray(mesh.faces)
    valid_faces = np.all(valid[all_faces], axis=1)
    valid_face_indices = np.flatnonzero(valid_faces)
    faces = all_faces[valid_faces]
    depths = np.mean(z[faces], axis=1)
    order = np.argsort(depths)[::-1]
    background = np.zeros((height, width, 3), dtype=np.uint8)
    background[:] = (23, 26, 30)
    # Match the Web viewer's smooth shading. Using geometric face normals here
    # made the regular Marching Cubes tessellation look like depth banding even
    # when the underlying surface was continuous.
    normals = np.mean(np.asarray(mesh.vertex_normals)[faces], axis=1)
    normal_length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(normal_length, 1e-12)
    light = np.asarray([-0.35, 0.65, 0.68])
    light /= np.linalg.norm(light)
    intensity = np.clip(0.36 + 0.64 * np.abs(normals @ light), 0.0, 1.0)
    base = np.asarray([188, 181, 170], dtype=np.float64)  # BGR neutral gray
    per_face_base: np.ndarray | None = None
    if use_mesh_face_colors:
        colors = np.asarray(getattr(mesh.visual, "face_colors", np.empty((0, 4))))
        if colors.ndim == 2 and len(colors) == len(all_faces) and colors.shape[1] >= 3:
            per_face_base = colors[valid_face_indices, :3][:, ::-1].astype(np.float64)
    triangles = np.rint(pixels[faces]).astype(np.int32)
    for face_index in order:
        triangle = triangles[face_index]
        if (
            triangle[:, 0].max() < 0
            or triangle[:, 1].max() < 0
            or triangle[:, 0].min() >= width
            or triangle[:, 1].min() >= height
        ):
            continue
        face_base = base if per_face_base is None else per_face_base[face_index]
        color = tuple(int(value) for value in np.clip(face_base * intensity[face_index], 0, 255))
        cv2.fillConvexPoly(background, triangle, color, lineType=cv2.LINE_AA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), background)
