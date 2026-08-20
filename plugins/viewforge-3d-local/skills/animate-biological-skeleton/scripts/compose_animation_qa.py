#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from _animation_common import atomic_write_json, load_json, sha256_file
from PIL import Image, ImageDraw


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Composite model and skeleton layers into a GIF and contact sheet."
    )
    parser.add_argument("--render-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--columns", default=4, type=int)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    render_dir = arguments.render_dir.expanduser().resolve()
    output_dir = arguments.output_dir.expanduser().resolve()
    metrics_path = render_dir / "render-metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = load_json(metrics_path)
    records = metrics.get("frames")
    if not isinstance(records, list) or not records:
        raise ValueError("render metrics contain no frames")
    combined_dir = output_dir / "frames"
    gif_path = output_dir / "animation-preview.gif"
    sheet_path = output_dir / "keyframes-contact-sheet.png"
    composition_path = output_dir / "composition.json"
    planned = [
        *(combined_dir / f"frame-{int(record['frame']):04d}.png" for record in records),
        gif_path,
        sheet_path,
        composition_path,
    ]
    collisions = [str(path) for path in planned if path.exists()]
    if collisions:
        raise FileExistsError(f"immutable composition outputs already exist: {collisions[:4]}")
    combined_dir.mkdir(parents=True, exist_ok=True)

    combined_paths: list[Path] = []
    for record in records:
        frame = int(record["frame"])
        destination = combined_dir / f"frame-{frame:04d}.png"
        if metrics["mode"] == "rigid-bound":
            source_path = render_dir / record["source"]["path"]
            bones_path = render_dir / record["bones"]["path"]
            with (
                Image.open(source_path).convert("RGBA") as source,
                Image.open(bones_path).convert("RGBA") as bones,
            ):
                Image.alpha_composite(source, bones).convert("RGB").save(
                    destination, format="PNG", optimize=True
                )
        else:
            source_path = render_dir / record["combined"]["path"]
            with Image.open(source_path).convert("RGB") as image:
                image.save(destination, format="PNG", optimize=True)
        combined_paths.append(destination)

    images = [Image.open(path).convert("RGB") for path in combined_paths]
    contiguous = all(
        int(records[index]["frame"]) == int(records[index - 1]["frame"]) + 1
        for index in range(1, len(records))
    )
    duration_ms = max(1, round(1000 / int(metrics["fps"]))) if contiguous else 300
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )

    columns = min(max(1, arguments.columns), len(images))
    rows = math.ceil(len(images) / columns)
    tile_width, tile_height = images[0].size
    label_height = 34
    sheet = Image.new("RGB", (tile_width * columns, (tile_height + label_height) * rows), "#070b12")
    draw = ImageDraw.Draw(sheet)
    for index, (record, image) in enumerate(zip(records, images, strict=True)):
        column = index % columns
        row = index // columns
        x = column * tile_width
        y = row * (tile_height + label_height)
        sheet.paste(image, (x, y))
        draw.text(
            (x + 14, y + tile_height + 8),
            f"Frame {int(record['frame'])}",
            fill="#d8e3f0",
        )
    sheet.save(sheet_path, format="PNG", optimize=True)
    for image in images:
        image.close()

    payload = {
        "schemaVersion": 1,
        "kind": "model-plus-complete-skeleton-animation-preview",
        "state": "pendingUserSignoff",
        "sourceMetrics": str(metrics_path),
        "sourceMetricsSha256": sha256_file(metrics_path),
        "mode": metrics["mode"],
        "frameCount": len(records),
        "frames": [
            {
                "frame": int(record["frame"]),
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
            }
            for record, path in zip(records, combined_paths, strict=True)
        ],
        "gif": {"path": str(gif_path), "sha256": sha256_file(gif_path)},
        "contactSheet": {
            "path": str(sheet_path),
            "sha256": sha256_file(sheet_path),
        },
        "completeSkeletonVisible": True,
        "skin": False,
    }
    atomic_write_json(composition_path, payload)
    print(
        json.dumps(
            {
                "frames": len(records),
                "gif": str(gif_path),
                "contactSheet": str(sheet_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
