from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from face3d.comparison import build_original_model_comparison


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_comparison_preserves_all_source_files(tmp_path: Path) -> None:
    sources: list[Path] = []
    for index, color in enumerate(("white", "lavender", "black", "purple")):
        source = tmp_path / f"source-{index}.png"
        Image.new("RGB", (320 + index, 480 - index), color).save(source)
        sources.append(source)
    before = {source: _sha256(source) for source in sources}
    output = tmp_path / "comparison.png"

    build_original_model_comparison(
        original_front_back=sources[0],
        original_side=sources[1],
        model_front=sources[2],
        model_back=sources[3],
        destination=output,
    )

    assert output.is_file()
    assert Image.open(output).size == (1920, 1440)
    assert {source: _sha256(source) for source in sources} == before
