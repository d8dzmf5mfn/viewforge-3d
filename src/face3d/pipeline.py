from __future__ import annotations

import resource
import sys
import time
from pathlib import Path
from typing import Any

from face3d.assets import require_assets
from face3d.config import Face3DConfig
from face3d.io import package_code_hash, sha256_file
from face3d.models import REQUIRED_VIEWS
from face3d.report import build_report, write_notices
from face3d.stages.hybrid_v2 import run_hybrid_v2
from face3d.stages.intake import discover_views, run_intake
from face3d.stages.pixel_direct import run_pixel_direct
from face3d.stages.template_v3 import run_template_v3
from face3d.state import RunState


def _rss_bytes() -> int:
    # macOS reports bytes; Linux reports KiB.
    measured = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return measured if sys.platform == "darwin" else measured * 1024


def _files(run_dir: Path, relatives: list[str]) -> list[Path]:
    return [run_dir / relative for relative in relatives]


def reconstruct(input_dir: Path, run_dir: Path, config: Face3DConfig) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    assets = require_assets(config)
    code_hash = package_code_hash()
    input_views = discover_views(input_dir)
    input_hashes = {role.value: sha256_file(path) for role, path in input_views.items()}
    model_hashes = {
        name: details["sha256"]
        for name, details in assets["models"].items()
        if details["sha256"] is not None
    }
    common = {
        "input": input_hashes,
        "config": sha256_file(config.source_path),
        "models": model_hashes,
        "code": code_hash,
    }
    state = RunState(run_dir)

    intake_signature = state.signature({**common, "stage": "intake"})
    intake_artifacts = _files(
        run_dir,
        [
            "working/intake.json",
            *[f"references/{role.value}.png" for role in REQUIRED_VIEWS],
            *[f"working/landmarks/{role.value}.npz" for role in REQUIRED_VIEWS],
            *[f"working/masks/{role.value}.png" for role in REQUIRED_VIEWS],
            "working/masks/confirmed.json",
            *[f"overlays/landmarks-{role.value}.png" for role in REQUIRED_VIEWS],
            *[f"overlays/silhouette-{role.value}.png" for role in REQUIRED_VIEWS],
        ],
    )
    intake_reused = state.reusable("intake", intake_signature)
    if not intake_reused:
        stage_started = time.perf_counter()
        intake = run_intake(input_dir, run_dir, config)
        state.complete(
            "intake",
            intake_signature,
            intake_artifacts,
            {
                "views": len(intake["views"]),
                "stageElapsedSeconds": time.perf_counter() - stage_started,
                "peakRssBytes": _rss_bytes(),
            },
        )

    reconstruction_stage = config.mode
    pixel_signature = state.signature(
        {**common, "stage": reconstruction_stage, "upstream": intake_signature}
    )
    if config.is_v3:
        pixel_relatives = [
            "working/fit.npz",
            "working/cameras.json",
            "working/fit-metrics.json",
            "working/unified-head.npz",
            "working/skin-metrics.json",
            "working/mesh-metrics.json",
            "working/sdf-metrics.json",
            "qa/anatomy.json",
            "models/fitted-head.glb",
            "models/head.glb",
            "textures/head-albedo.jpg",
            "textures/head-confidence.png",
            "textures/head-source.png",
            "projection/skin-projection.npz",
            "projection/schema.json",
            *[f"overlays/fit-silhouette-{role.value}.png" for role in REQUIRED_VIEWS],
        ]
    elif config.is_v2:
        pixel_relatives = [
            "working/fit.npz",
            "working/cameras.json",
            "working/fit-metrics.json",
            "working/unified-head.npz",
            "working/sdf-v2.f16",
            "working/sdf-v2.json",
            "working/sdf-metrics.json",
            "working/smooth-mesh.npz",
            "working/mesh-metrics.json",
            "working/skin-metrics.json",
            "qa/anatomy.json",
            "pixels/pixels.bin",
            "pixels/schema.json",
            "models/voxels.glb",
            "models/head.glb",
            "textures/head-albedo.jpg",
            "textures/head-confidence.png",
            "textures/head-source.png",
            *[f"overlays/fit-silhouette-{role.value}.png" for role in REQUIRED_VIEWS],
        ]
    else:
        pixel_relatives = [
            "working/pixel-field.npz",
            "working/cameras.json",
            "working/fit-metrics.json",
            "working/sdf-metrics.json",
            "working/smooth-mesh.npz",
            "working/mesh-metrics.json",
            "working/skin-metrics.json",
            "pixels/pixels.bin",
            "pixels/schema.json",
            "models/voxels.glb",
            "models/raw-isosurface.glb",
            "models/smooth.glb",
            *[f"overlays/fit-silhouette-{role.value}.png" for role in REQUIRED_VIEWS],
        ]
        if not config.output.geometry_only:
            pixel_relatives.extend(
                [
                    "models/skin.glb",
                    "textures/skin-atlas.jpg",
                    "textures/skin-confidence.png",
                    "textures/skin-normal.png",
                    "textures/skin-metallic-roughness.png",
                ]
            )
    pixel_artifacts = _files(run_dir, pixel_relatives)
    pixel_reused = state.reusable(reconstruction_stage, pixel_signature)
    if not pixel_reused:
        stage_started = time.perf_counter()
        if config.is_v3:
            metrics = run_template_v3(run_dir, config)
        elif config.is_v2:
            metrics = run_hybrid_v2(run_dir, config)
        else:
            metrics = run_pixel_direct(run_dir, config)
        state.complete(
            reconstruction_stage,
            pixel_signature,
            pixel_artifacts,
            {
                **metrics,
                "stageElapsedSeconds": time.perf_counter() - stage_started,
                "peakRssBytes": _rss_bytes(),
            },
        )

    stage_metrics = [state.metrics(stage) for stage in ("intake", reconstruction_stage)]
    runtime = {
        "elapsedSeconds": sum(
            float(metrics.get("stageElapsedSeconds", 0.0)) for metrics in stage_metrics
        ),
        "lastInvocationElapsedSeconds": time.perf_counter() - started,
        "peakRssBytes": max(
            [_rss_bytes(), *[int(metrics.get("peakRssBytes", 0)) for metrics in stage_metrics]]
        ),
        "deterministic": config.deterministic,
        "resumedStages": [
            stage
            for stage, reused in (("intake", intake_reused), (reconstruction_stage, pixel_reused))
            if reused
        ],
    }
    manifest, report = build_report(run_dir, config, runtime)
    write_notices(run_dir)
    return {
        "ok": True,
        "run": str(run_dir),
        "manifest": str(run_dir / "manifest.json"),
        "automatedGatesPassed": report["summary"]["automatedGatesPassed"],
        "userSignoffRequired": True,
        "runtime": manifest["runtime"],
    }
