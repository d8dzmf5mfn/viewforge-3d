import cv2
import numpy as np

from face3d.height_mesh import build_height_field_mesh


def test_height_field_mesh_is_closed_and_feature_locked() -> None:
    height, width = 100, 84
    y, x = np.mgrid[:height, :width]
    mask = (
        ((x - width / 2) / (width * 0.42)) ** 2 + ((y - height / 2) / (height * 0.46)) ** 2 <= 1
    ).astype(np.uint8) * 255
    depth = 0.22 * np.exp(
        -(((x - width / 2) / (width * 0.2)) ** 2) - (((y - height * 0.48) / (height * 0.28)) ** 2)
    )
    depth += 0.012 * np.sin(x * 0.9)
    feature = np.zeros_like(mask)
    cv2.circle(feature, (width // 2, height // 2), 8, 2, thickness=cv2.FILLED)
    result = build_height_field_mesh(
        depth.astype(np.float32),
        mask,
        feature,
        target_triangles=12_000,
        minimum_triangles=8_000,
        maximum_triangles=16_000,
        pixel_step=1 / width,
        taubin_iterations=16,
        taubin_lambda=0.42,
        taubin_nu=-0.1,
        hausdorff_voxels_max=1.5,
    )
    assert result.metrics["watertight"]
    assert result.metrics["edgeManifold"]
    assert result.metrics["windingConsistent"]
    assert result.metrics["boundaryEdges"] == 0
    assert result.metrics["featureDriftVoxels"] == 0
    assert result.metrics["normalVarianceReduction"] >= 0.3
    assert 8_000 <= result.metrics["triangles"] <= 16_000
