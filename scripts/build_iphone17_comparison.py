#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from face3d.comparison import build_original_model_comparison


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Place original iPhone evidence and existing model renders side by side."
    )
    parser.add_argument("--original-front-back", type=Path, required=True)
    parser.add_argument("--original-side", type=Path, required=True)
    parser.add_argument("--model-front", type=Path, required=True)
    parser.add_argument("--model-back", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    build_original_model_comparison(
        original_front_back=arguments.original_front_back,
        original_side=arguments.original_side,
        model_front=arguments.model_front,
        model_back=arguments.model_back,
        destination=arguments.output,
    )


if __name__ == "__main__":
    main()
