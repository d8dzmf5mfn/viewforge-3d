from __future__ import annotations

import numpy as np

from face3d.multiview_pixels import _surface_mask
from face3d.pixel_coarsen import coarsen_pixel_grid


def test_factor_two_increases_unit_volume_and_reduces_surface_count() -> None:
    occupancy = np.ones((4, 4, 4), dtype=bool)
    source_surface_indices = np.argwhere(_surface_mask(occupancy))
    source_rgb = np.full((len(source_surface_indices), 3), 220, dtype=np.uint8)
    coarse = coarsen_pixel_grid(
        occupancy,
        np.zeros(3, dtype=np.float64),
        0.1,
        source_surface_indices,
        source_rgb,
        factor=2,
    )

    assert coarse.occupancy.shape == (2, 2, 2)
    assert len(coarse.surface_indices) == 8
    assert len(source_surface_indices) == 56
    assert coarse.unit_size == 0.2
    assert np.allclose(coarse.origin, 0.05)
    assert coarse.metrics["unitVolumeFactor"] == 8.0
    assert coarse.metrics["surfaceConnectionPerformed"] is False
    assert coarse.metrics["geometricSmoothingApplied"] is False


def test_coarse_unit_uses_darkest_source_sample_to_preserve_features() -> None:
    occupancy = np.ones((4, 4, 4), dtype=bool)
    source_surface_indices = np.argwhere(_surface_mask(occupancy))
    source_rgb = np.full((len(source_surface_indices), 3), 230, dtype=np.uint8)
    source_target = np.flatnonzero(np.all(source_surface_indices == np.asarray([0, 0, 0]), axis=1))[
        0
    ]
    source_rgb[source_target] = (12, 18, 28)
    coarse = coarsen_pixel_grid(
        occupancy,
        np.zeros(3, dtype=np.float64),
        0.1,
        source_surface_indices,
        source_rgb,
        factor=2,
    )

    target = np.flatnonzero(np.all(coarse.surface_indices == np.asarray([0, 0, 0]), axis=1))[0]
    assert coarse.representative_source_indices[target] == source_target
    assert np.array_equal(
        source_rgb[coarse.representative_source_indices[target]],
        np.asarray([12, 18, 28], dtype=np.uint8),
    )
