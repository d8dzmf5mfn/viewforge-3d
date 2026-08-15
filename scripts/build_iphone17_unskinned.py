#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from face3d.phone_run import ReferenceImage, build_iphone17_unskinned_run

HERO_URL = (
    "https://www.apple.com/newsroom/images/2025/09/apple-debuts-iphone-17/article/"
    "Apple-iPhone-17-hero-250909_inline.jpg.large_2x.jpg"
)
SIDE_URL = (
    "https://www.apple.com/newsroom/images/2025/09/apple-debuts-iphone-17/article/"
    "Apple-iPhone-17-color-lineup-250909_big.jpg.large_2x.jpg"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an immutable, unskinned iPhone 17 geometry-preview run."
    )
    parser.add_argument("--hero", type=Path, required=True)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    manifest = build_iphone17_unskinned_run(
        arguments.output,
        references=(
            ReferenceImage(
                label="official-front-back-perspective",
                source_url=HERO_URL,
                path=arguments.hero,
                evidence_role="shape-and-attached-feature-reference",
            ),
            ReferenceImage(
                label="official-side-and-color-lineup",
                source_url=SIDE_URL,
                path=arguments.side,
                evidence_role="side-profile-and-contoured-edge-reference",
            ),
        ),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
