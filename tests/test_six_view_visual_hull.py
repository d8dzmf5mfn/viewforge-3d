from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from face3d.six_view_visual_hull import SIX_VIEW_ROLES, reconstruct_six_view_visual_hull


def _silhouette(path: Path, *, horizontal: bool) -> None:
    image = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(image)
    box = (18, 35, 109, 92) if horizontal else (35, 18, 92, 109)
    draw.rounded_rectangle(box, radius=14, fill="black")
    image.save(path)


def test_six_view_visual_hull_writes_preview_glb_and_provenance(tmp_path: Path) -> None:
    views: dict[str, Path] = {}
    for role in SIX_VIEW_ROLES:
        path = tmp_path / f"{role}.png"
        _silhouette(path, horizontal=role in {"top", "bottom"})
        views[role] = path

    result = reconstruct_six_view_visual_hull(
        views,
        tmp_path / "output",
        resolution=48,
        width_m=0.8,
        depth_m=0.6,
        height_m=1.2,
    )

    assert result["ok"] is True
    assert result["state"] == "preview"
    assert result["previewOnly"] is True
    model_path = Path(result["model"])
    assert model_path.is_file()
    mesh = trimesh.load(model_path, force="mesh", process=False)
    assert len(mesh.vertices) > 8
    assert len(mesh.faces) > 12
    assert np.all(np.isfinite(mesh.vertices))
