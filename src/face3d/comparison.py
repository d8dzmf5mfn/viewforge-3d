from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _fit_panel(image: Image.Image, size: tuple[int, int], background: str) -> Image.Image:
    contained = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, background)
    panel.paste(
        contained,
        ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2),
    )
    return panel


def build_original_model_comparison(
    *,
    original_front_back: Path,
    original_side: Path,
    model_front: Path,
    model_back: Path,
    destination: Path,
) -> None:
    sources = (original_front_back, original_side, model_front, model_back)
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    canvas = Image.new("RGB", (1920, 1440), "#12151c")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=38)
    label_font = ImageFont.load_default(size=25)
    draw.text(
        (72, 42),
        "iPhone 17: official 2D evidence vs geometry preview",
        fill="white",
        font=title_font,
    )
    draw.text(
        (72, 102),
        "Sources are copied into this board without in-place modification.",
        fill="#aeb8ca",
        font=label_font,
    )

    panel_size = (820, 520)
    positions = ((70, 205), (1030, 205), (70, 835), (1030, 835))
    labels = (
        "Apple official front/back perspective",
        "TemplatePhoneV0 front orbit",
        "Apple official side/color lineup",
        "TemplatePhoneV0 back orbit",
    )
    ordered_sources = (original_front_back, model_front, original_side, model_back)
    backgrounds = ("#f5f5f5", "#1b1717", "#f5f5f5", "#1b1717")
    for source, position, label, background in zip(
        ordered_sources,
        positions,
        labels,
        backgrounds,
        strict=True,
    ):
        with Image.open(source) as image:
            panel = _fit_panel(image, panel_size, background)
        canvas.paste(panel, position)
        draw.rectangle(
            (
                position[0] - 1,
                position[1] - 1,
                position[0] + panel_size[0],
                position[1] + panel_size[1],
            ),
            outline="#3b4352",
            width=2,
        )
        draw.text((position[0], position[1] - 40), label, fill="#e5e9f2", font=label_font)

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)
