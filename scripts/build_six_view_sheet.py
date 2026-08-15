"""Build a labelled 3x2 six-view review sheet from fixed Blender renders."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

VIEWS = (
    ("left90.png", "左 90°"),
    ("left45.png", "左 45°"),
    ("front.png", "正面"),
    ("right45.png", "右 45°"),
    ("right90.png", "右 90°"),
    ("back.png", "背面"),
)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    images = [Image.open(args.input / name).convert("RGB") for name, _ in VIEWS]
    tile_width = max(image.width for image in images)
    tile_height = max(image.height for image in images)
    header_height = 82
    label_height = 46
    sheet = Image.new(
        "RGB",
        (tile_width * 3, header_height + (tile_height + label_height) * 2),
        (9, 12, 16),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((28, 18), args.title, fill=(235, 241, 247), font=font(34))

    for index, ((_, label), image) in enumerate(zip(VIEWS, images, strict=True)):
        row, column = divmod(index, 3)
        x = column * tile_width
        y = header_height + row * (tile_height + label_height)
        sheet.paste(image, (x, y))
        draw.rectangle(
            (x, y + tile_height, x + tile_width, y + tile_height + label_height),
            fill=(16, 21, 27),
        )
        draw.text(
            (x + 18, y + tile_height + 7),
            label,
            fill=(209, 221, 232),
            font=font(24),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, format="PNG", optimize=True)


if __name__ == "__main__":
    main()
