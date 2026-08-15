"""ViewForge 3D implementation with a legacy ``face3d`` import path."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("viewforge3d")
except PackageNotFoundError:  # pragma: no cover - source checkout
    __version__ = "0.1.0"

__all__ = ["__version__"]
