from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from .storage import sha256_file

MAX_SHEET_TILE_SIZE = 512
SHEET_COLUMNS = 3
SHEET_LABEL_HEIGHT = 34
SHEET_BACKGROUND = (18, 20, 25)
SHEET_LABEL_COLOR = (235, 237, 242)


def compose_preview_sheet(
    render_dir: Path,
    views: list[str],
    destination: Path,
) -> dict[str, Any]:
    if not views:
        raise ValueError("At least one rendered view is required.")

    sources = [render_dir / f"render-{view}.png" for view in views]
    if any(not source.is_file() for source in sources):
        raise FileNotFoundError("One or more rendered preview images are missing.")

    with Image.open(sources[0]) as first:
        tile_size = min(MAX_SHEET_TILE_SIZE, max(first.width, first.height))
    columns = min(SHEET_COLUMNS, len(sources))
    rows = math.ceil(len(sources) / columns)
    canvas = Image.new(
        "RGB",
        (columns * tile_size, rows * (tile_size + SHEET_LABEL_HEIGHT)),
        SHEET_BACKGROUND,
    )
    draw = ImageDraw.Draw(canvas)

    for index, (view, source) in enumerate(zip(views, sources, strict=True)):
        column = index % columns
        row = index // columns
        x = column * tile_size
        y = row * (tile_size + SHEET_LABEL_HEIGHT)
        with Image.open(source) as image:
            contained = ImageOps.contain(
                image.convert("RGBA"),
                (tile_size, tile_size),
                method=Image.Resampling.LANCZOS,
            )
            tile = Image.new("RGBA", (tile_size, tile_size), (*SHEET_BACKGROUND, 255))
            offset = (
                (tile_size - contained.width) // 2,
                (tile_size - contained.height) // 2,
            )
            tile.alpha_composite(contained, offset)
            canvas.paste(tile.convert("RGB"), (x, y))
        draw.text(
            (x + 12, y + tile_size + 9),
            view.replace("_", " ").title(),
            fill=SHEET_LABEL_COLOR,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)
    return {
        "name": destination.name,
        "sha256": sha256_file(destination),
        "width": canvas.width,
        "height": canvas.height,
        "views": list(views),
    }
