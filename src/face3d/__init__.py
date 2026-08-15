"""Traceable 3D Pixel assets with a three-view face compatibility pipeline."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("face3d")
except PackageNotFoundError:  # pragma: no cover - source checkout
    __version__ = "0.1.0"

__all__ = ["__version__"]
