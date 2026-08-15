#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

VIEWS = ("left90", "left45", "front", "right45", "right90", "back")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Composite model and bone QA render layers.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    input_dir = arguments.input.expanduser().resolve()
    output_dir = arguments.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"{name}.png" for name in VIEWS]
    outputs.append(output_dir / "composite-metrics.json")
    collisions = [str(path) for path in outputs if path.exists()]
    if collisions:
        raise FileExistsError(f"immutable composite outputs already exist: {collisions}")

    records: dict[str, dict[str, str]] = {}
    for name in VIEWS:
        source_path = input_dir / "source" / f"{name}.png"
        bone_path = input_dir / "bones" / f"{name}.png"
        if not source_path.is_file() or not bone_path.is_file():
            raise FileNotFoundError(f"missing QA render layers for {name}")
        source = Image.open(source_path).convert("RGBA")
        bones = Image.open(bone_path).convert("RGBA")
        if source.size != bones.size:
            raise ValueError(f"layer dimensions differ for {name}")
        destination = output_dir / f"{name}.png"
        Image.alpha_composite(source, bones).convert("RGB").save(
            destination, format="PNG", optimize=True
        )
        records[name] = {
            "sourceSha256": sha256_file(source_path),
            "boneSha256": sha256_file(bone_path),
            "output": destination.name,
            "outputSha256": sha256_file(destination),
        }

    (output_dir / "composite-metrics.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "operation": "alpha composite bone layer over source layer",
                "renders": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
