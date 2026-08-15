from pathlib import Path

import pytest

from face3d.errors import Face3DError
from face3d.models import ViewRole
from face3d.stages.intake import discover_views


def test_discovers_fixed_roles(tmp_path: Path) -> None:
    for index, role in enumerate(ViewRole):
        (tmp_path / f"{role.value}.jpg").write_bytes(bytes([index + 1]))
    views = discover_views(tmp_path)
    assert list(views) == list(ViewRole)


def test_duplicate_content_fails_closed(tmp_path: Path) -> None:
    for role in ViewRole:
        (tmp_path / f"{role.value}.png").write_bytes(b"same")
    with pytest.raises(Face3DError, match="完全相同") as raised:
        discover_views(tmp_path)
    assert raised.value.code == "duplicate-view"
