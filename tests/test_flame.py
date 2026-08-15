from pathlib import Path

import numpy as np

from face3d.stages.flame import FlameModel, FlameRegionMasks


def test_loads_numpy_flame_contract(tmp_path: Path) -> None:
    vertices = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    shapedirs = np.zeros((3, 3, 2), dtype=np.float64)
    shapedirs[2, 1, 0] = 0.5
    model_path = tmp_path / "model.npz"
    np.savez(model_path, v_template=vertices, faces=faces, shapedirs=shapedirs)
    landmarks_path = tmp_path / "landmarks.npz"
    np.savez(
        landmarks_path,
        lmk_face_idx=np.zeros(68, dtype=np.int64),
        lmk_b_coords=np.tile(np.asarray([[1 / 3, 1 / 3, 1 / 3]]), (68, 1)),
    )
    model = FlameModel.load(model_path, landmarks_path, 2)
    shaped = model.shaped_vertices(np.asarray([1.0, 0.0]))
    assert shaped[2, 1] == 1.5
    assert model.landmark_vertices(shaped, 0).shape == (68, 3)


def test_region_masks_accept_combined_byte_keys(tmp_path: Path) -> None:
    vertices = np.asarray(
        [
            [-2.0, 0.0, 0.0],
            [-1.5, 0.2, 0.0],
            [-1.0, 0.4, 0.0],
            [-0.5, 0.6, 0.0],
            [0.5, 0.6, 0.0],
            [1.0, 0.4, 0.0],
            [1.5, 0.2, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    path = tmp_path / "masks.npy"
    np.save(
        path,
        {
            b"ears": np.arange(8),
            b"eye_region": np.arange(8),
            b"neck": np.asarray([0, 7]),
        },
        allow_pickle=True,
    )
    masks = FlameRegionMasks.load(path, vertices)
    assert len(masks.left_ear) == 4
    assert len(masks.right_ear) == 4
    assert len(masks.left_eyelid) == 4
    assert len(masks.right_eyelid) == 4
