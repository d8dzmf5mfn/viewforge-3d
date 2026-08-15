from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

from face3d import __version__
from face3d.config import load_config
from face3d.io import atomic_write_json, sha256_file
from face3d.package import package_run
from face3d.report import build_report, write_notices
from face3d.stages.intake import confirm_masks, run_intake
from face3d.stages.pixel_direct import run_pixel_direct

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = PROJECT_ROOT / "configs" / "face-v1.yaml"
INPUT = PROJECT_ROOT / "assets" / "demo" / "generated-open-eye-v1"
SOURCE = "OpenAI-generated synthetic fixture created for this repository"
LICENSE = "Synthetic project test fixture; no real person's identity"


def _rss_bytes() -> int:
    measured = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return measured if sys.platform == "darwin" else measured * 1024


def create(output: Path) -> dict[str, object]:
    if not INPUT.is_dir():
        raise FileNotFoundError(f"missing committed 2D demo views: {INPUT}")
    started = time.perf_counter()
    config = load_config(CONFIG)
    with tempfile.TemporaryDirectory(prefix="face3d-demo-") as temporary:
        run = Path(temporary)
        # This confirmation is allowed only for the deterministic repository QA
        # fixture. Production reconstruction still stops for human mask review.
        run_intake(INPUT, run, config, stop_for_mask_review=False)
        confirm_masks(run)
        run_intake(INPUT, run, config, stop_for_mask_review=False)
        reconstruction = run_pixel_direct(run, config)
        runtime = {
            "elapsedSeconds": time.perf_counter() - started,
            "lastInvocationElapsedSeconds": time.perf_counter() - started,
            "peakRssBytes": _rss_bytes(),
            "deterministic": config.deterministic,
            "resumedStages": [],
        }
        manifest, report = build_report(run, config, runtime)
        manifest["provenance"]["demoAsset"] = {
            "name": "Generated Open-Eye Face v1",
            "license": LICENSE,
            "source": SOURCE,
            "role": "generated-2d-input-fixture-only",
            "views": {path.stem: sha256_file(path) for path in sorted(INPUT.glob("*.png"))},
        }
        manifest["demo"] = {
            "kind": "deterministic-synthetic-benchmark",
            "inputBoundary": "reconstruction reads only the three committed PNG views",
            "visualReviewStatus": report["summary"]["visualReviewStatus"],
        }
        atomic_write_json(run / "manifest.json", manifest)
        write_notices(run)
        packaged = package_run(run, output, config)
    return {
        **packaged,
        "face3dVersion": __version__,
        "reconstruction": reconstruction,
        "automatedGatesPassed": report["summary"]["automatedGatesPassed"],
        "visualReviewStatus": report["summary"]["visualReviewStatus"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the viewer demo through the production 2D-to-3D pipeline"
    )
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(create(arguments.output), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
