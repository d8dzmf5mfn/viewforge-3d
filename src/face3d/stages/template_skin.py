from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from face3d.config import Face3DConfig
from face3d.errors import fail
from face3d.io import atomic_write_bytes, atomic_write_json, sha256_file
from face3d.models import CameraRecord
from face3d.skin_v2 import build_skin_v2
from face3d.unified_head import UnifiedHeadAsset, geometry_hash


def _load_fitted_head(run_dir: Path) -> tuple[UnifiedHeadAsset, dict[str, Any]]:
    fit_metrics_path = run_dir / "working" / "fit-metrics.json"
    fitted_path = run_dir / "working" / "unified-head.npz"
    cameras_path = run_dir / "working" / "cameras.json"
    missing = [
        str(path)
        for path in (fit_metrics_path, fitted_path, cameras_path)
        if not path.is_file()
    ]
    if missing:
        fail(
            "template-skin-upstream-missing",
            "TemplateHeadV0 皮肤投影缺少拟合产物",
            stage="template-skin",
            details={"missing": missing},
        )
    fit_metrics = json.loads(fit_metrics_path.read_text())
    if not fit_metrics.get("passed"):
        fail(
            "template-skin-fit-not-accepted",
            "三视图拟合未通过，禁止在失败几何上生成看似合理的人皮",
            stage="template-skin",
            details={"fitMetrics": str(fit_metrics_path)},
        )
    head = UnifiedHeadAsset.load(fitted_path)
    actual_hash = geometry_hash(head.render_vertices, head.render_faces)
    expected_hash = fit_metrics.get("fittedGeometrySha256")
    if actual_hash != expected_hash or actual_hash != head.geometry_sha256:
        fail(
            "template-skin-geometry-hash-mismatch",
            "拟合头模、皮肤输入和记录的几何哈希不一致",
            stage="template-skin",
            details={
                "fit": expected_hash,
                "asset": head.geometry_sha256,
                "actual": actual_hash,
            },
        )
    return head, fit_metrics


def run_template_skin(run_dir: Path, config: Face3DConfig) -> dict[str, Any]:
    if not config.is_v3:
        fail(
            "config-invalid",
            "TemplateHeadV0 皮肤投影只接受 face-v3 配置",
            stage="template-skin",
        )
    run_dir = run_dir.expanduser().resolve()
    head, fit_metrics = _load_fitted_head(run_dir)
    cameras_payload = json.loads((run_dir / "working" / "cameras.json").read_text())
    cameras = [CameraRecord.model_validate(value) for value in cameras_payload["cameras"]]
    if len(cameras) != 3:
        fail(
            "template-skin-camera-count-invalid",
            "TemplateHeadV0 皮肤投影必须使用三台已拟合相机",
            stage="template-skin",
            details={"cameraCount": len(cameras)},
        )

    result = build_skin_v2(
        run_dir,
        head,
        cameras,
        config,
        observed_confidence_threshold=0.02,
    )
    projection = result.projection
    trace_output = io.BytesIO()
    np.savez_compressed(
        trace_output,
        source_role=projection.source_role.astype(np.uint8),
        source_uv=projection.source_uv.astype(np.uint16),
        camera_depth=projection.depth.astype(np.float32),
        confidence=projection.confidence.astype(np.float32),
        source_bits=projection.source_bits.astype(np.uint8),
        per_view_weights=projection.per_view_weights.astype(np.float16),
    )
    trace_path = run_dir / "projection" / "skin-projection.npz"
    atomic_write_bytes(trace_path, trace_output.getvalue())
    schema_path = run_dir / "projection" / "schema.json"
    atomic_write_json(
        schema_path,
        {
            "schemaVersion": 1,
            "recordCount": int(len(projection.source_role)),
            "recordDomain": "TemplateHeadV0 compute vertices",
            "viewOrder": [camera.role.value for camera in cameras],
            "fields": {
                "source_role": "uint8; view index or 255 for template-inferred",
                "source_uv": "uint16[N,2]; selected source-image pixel coordinate",
                "camera_depth": "float32[N]; selected camera-space depth",
                "confidence": "float32[N]; fused projection confidence",
                "source_bits": "uint8[N]; bit mask of contributing views",
                "per_view_weights": "float16[N,3]; normalized view weights",
            },
        },
    )

    metrics = {
        **result.metrics,
        "schemaVersion": 3,
        "profile": "face-v3",
        "surfaceSource": "fitted-template-head-v0",
        "geometryRecreated": False,
        "neutralAndSkinSharePositionIndex": True,
        "projectionTrace": "projection/skin-projection.npz",
        "projectionTraceSha256": sha256_file(trace_path),
        "projectionSchema": "projection/schema.json",
        "projectionSchemaSha256": sha256_file(schema_path),
        "traceRecordCount": int(len(projection.source_role)),
        "sourceViewOrder": [camera.role.value for camera in cameras],
        "fitMetrics": "working/fit-metrics.json",
        "fitGeometryHash": fit_metrics["fittedGeometrySha256"],
        "sdfUsed": False,
    }
    atomic_write_json(run_dir / "working" / "skin-metrics.json", metrics)
    if not metrics["passed"]:
        fail(
            "template-skin-gate-failed",
            "同一几何上的三视图皮肤投影未通过门禁",
            stage="template-skin",
            details={
                "metrics": str(run_dir / "working" / "skin-metrics.json"),
                "observedVertexFraction": metrics["observedVertexFraction"],
                "seamDeltaE00Median": metrics["seamDeltaE00Median"],
                "seamDeltaE00P95": metrics["seamDeltaE00P95"],
            },
        )
    return metrics
