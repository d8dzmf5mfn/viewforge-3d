"""Compose fixed-view hard-material candidates rendered from the same V24 geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/System/Library/Fonts/STHeiti Medium.ttc"


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size=size)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (2880, 940), (14, 18, 23))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (48, 28), "V24 硬质材质候选｜同几何、同灯光、同机位", font=font(40), fill=(238, 242, 247)
    )
    draw.text(
        (50, 82),
        "“硬感”来自低 Subsurface、较清晰的高光与受控 Coat；不代表模型具有真实机械硬度。",
        font=font(23),
        fill=(158, 171, 186),
    )

    options = (
        (
            "hard-satin",
            "硬质缎面｜推荐",
            "Base .55/.58/.62\nRoughness .36  IOR 1.50\n"
            "Specular .42  Coat .16/.20\nMetal 0  Subsurface 0",
            (87, 190, 255),
        ),
        (
            "matte-ceramic",
            "哑光陶瓷",
            "Base .72/.68/.62\nRoughness .30  IOR 1.52\n"
            "Specular .48  Coat .28/.14\nMetal 0  Subsurface 0",
            (255, 184, 95),
        ),
        (
            "dense-stone",
            "致密石材",
            "Base .34/.38/.42\nRoughness .46  IOR 1.48\n"
            "Specular .36  Coat .04/.28\nMetal 0  Subsurface 0",
            (173, 181, 192),
        ),
        (
            "jade-hard",
            "硬质青玉感",
            "Base .18/.42/.28\nRoughness .32  IOR 1.54\n"
            "Specular .44  Coat .12/.18\nMetal 0  Subsurface 0",
            (92, 221, 162),
        ),
    )
    views = (("left90.png", "侧"), ("front.png", "正"), ("right45.png", "45°"))
    for index, (folder, title, params, color) in enumerate(options):
        x0 = 40 + index * 710
        draw.rounded_rectangle(
            (x0, 130, x0 + 680, 900),
            radius=24,
            fill=(24, 30, 37),
            outline=color,
            width=4 if index == 0 else 2,
        )
        draw.text((x0 + 24, 150), title, font=font(29), fill=(238, 242, 247))
        for view_index, (filename, label) in enumerate(views):
            image = Image.open(args.root / folder / filename).convert("RGB")
            image.thumbnail((210, 210), Image.Resampling.LANCZOS)
            x = x0 + 15 + view_index * 220
            canvas.paste(image, (x, 205))
            draw.text((x + 88, 416), label, font=font(19), fill=(158, 171, 186))
        draw.multiline_text(
            (x0 + 28, 475),
            params,
            font=font(24),
            fill=(205, 214, 224),
            spacing=15,
        )
        if index == 0:
            draw.rounded_rectangle((x0 + 24, 718, x0 + 656, 858), radius=18, fill=(25, 49, 66))
            draw.text((x0 + 42, 738), "首选原因", font=font(24), fill=color)
            draw.multiline_text(
                (x0 + 42, 778),
                "高光比当前参考更清楚，但不会像亮釉那样\n放大每一道条带；五官仍容易读。",
                font=font(21),
                fill=(205, 214, 224),
                spacing=8,
            )
        else:
            notes = {
                "matte-ceramic": "亮斑最强；平滑不足时会放大表面缺陷。",
                "dense-stone": "最稳重，反光最弱；硬但不“亮”。",
                "jade-hard": "颜色明确；仍保持不透明，避免蜡感。",
            }
            draw.rounded_rectangle((x0 + 24, 738, x0 + 656, 840), radius=18, fill=(30, 37, 45))
            draw.multiline_text(
                (x0 + 42, 764),
                notes[folder],
                font=font(21),
                fill=(185, 195, 206),
                spacing=8,
            )

    canvas.save(args.output, format="PNG", optimize=True)


if __name__ == "__main__":
    main()
