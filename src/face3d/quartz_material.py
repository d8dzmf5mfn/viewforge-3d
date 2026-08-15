from __future__ import annotations

from typing import Any

import numpy as np


def quartz_color_volume(
    occupancy: np.ndarray,
    solid_indices: np.ndarray,
    solid_rgb: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map source luminance to a deterministic 3D milky-quartz mineral field."""
    source = np.asarray(solid_rgb, dtype=np.float64) / 255.0
    solid_indices = np.asarray(solid_indices, dtype=np.int32)
    luminance = source @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    coordinates = solid_indices.astype(np.float64)
    primary = np.sin(
        coordinates[:, 0] * 0.217
        + coordinates[:, 1] * 0.083
        + coordinates[:, 2] * 0.173
        + 0.72 * np.sin(coordinates[:, 1] * 0.041 + coordinates[:, 2] * 0.097)
    )
    secondary = np.sin(
        coordinates[:, 0] * 0.613 - coordinates[:, 1] * 0.127 + coordinates[:, 2] * 0.337
    )
    veins = np.exp(-((np.abs(primary) / 0.105) ** 2))
    crystal_grain = primary * secondary
    tone = 0.43 + 0.53 * np.power(np.clip(luminance, 0.0, 1.0), 0.72)
    tone = np.clip(tone - veins * 0.085 + crystal_grain * 0.028, 0.34, 0.985)
    quartz = np.column_stack(
        (
            tone * 0.94 + 0.018,
            tone * 0.975 + 0.014,
            tone + 0.010,
        )
    )
    quartz = np.rint(np.clip(quartz, 0.0, 1.0) * 255.0).astype(np.uint8)
    volume = np.full((*occupancy.shape, 3), (225, 234, 242), dtype=np.uint8)
    volume[tuple(solid_indices.T)] = quartz
    correlation = 1.0
    if float(np.std(luminance)) > 1e-12 and float(np.std(tone)) > 1e-12:
        correlation = float(np.corrcoef(luminance, tone)[0, 1])
    return volume, {
        "profile": "polished-milky-quartz",
        "materialClass": "natural-stone/quartz",
        "mapping": "source luminance plus deterministic three-dimensional mineral veins",
        "sourceHuePreserved": False,
        "sourceLuminanceStructurePreserved": True,
        "sourceToQuartzToneCorrelation": correlation,
        "quartzRgbMinimum": quartz.min(axis=0).astype(int).tolist(),
        "quartzRgbMaximum": quartz.max(axis=0).astype(int).tolist(),
        "quartzRgbMean": quartz.mean(axis=0).astype(float).tolist(),
        "randomSeedUsed": False,
        "textureCoordinatesInWorldGrid": True,
    }
