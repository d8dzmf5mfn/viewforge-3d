from __future__ import annotations

import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from face3d.config import load_config
from face3d.io import atomic_write_bytes, atomic_write_json, sha256_file
from face3d.package import package_run
from face3d.unified_head import UnifiedHeadAsset, geometry_hash

ROOT = Path(__file__).resolve().parent.parent


def create(output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="face-v3-contract-") as temporary:
        run = Path(temporary)
        for directory in (
            "models",
            "projection",
            "qa",
            "references",
            "overlays",
            "textures",
        ):
            (run / directory).mkdir(parents=True, exist_ok=True)

        template_root = ROOT / "assets" / "template-head-v0" / "anatomy"
        head = UnifiedHeadAsset.load(template_root / "template-head-v0.unified.npz")
        source_anatomy = json.loads(
            (template_root / "anatomy.json").read_text(encoding="utf-8")
        )
        vertices = np.asarray(head.skin_vertices, dtype=np.float64)
        faces = np.asarray(head.skin_faces, dtype=np.int64)
        digest = geometry_hash(vertices, faces)
        source_geometry = source_anatomy["geometry"]
        source_eyes = source_anatomy["eyes"]
        assert digest == source_geometry["computeGeometrySha256"]
        anatomy = {
            "schemaVersion": 3,
            "templateId": "TemplateHeadV0",
            "geometry": {
                "componentCount": source_geometry["componentCount"],
                "boundaryEdgeCount": source_geometry["boundaryEdgeCount"],
                "nonManifoldEdgeCount": source_geometry["nonManifoldEdgeCount"],
                "selfIntersectionPairCount": source_geometry["topologyStabilization"][
                    "float32SelfIntersectionPairCount"
                ],
                "topCurvatureSpikeRatio": source_geometry["topCurvatureSpikeRatio"],
                "geometryHash": digest,
                "geometryHashesMatch": True,
                "passed": True,
            },
            "ears": {
                **source_anatomy["ears"],
            },
            "eyes": {
                "completeEyeballNodes": source_eyes["completeEyeballNodes"],
                "intersectionCount": source_eyes["intersectionCount"],
                "radiusDifferenceRatio": source_eyes["radiusDifferenceRatio"],
                "left": {
                    "contactGapP99R": source_eyes["left"]["contactGapP99R"],
                    "passed": source_eyes["left"]["intersectionCount"] == 0,
                },
                "right": {
                    "contactGapP99R": source_eyes["right"]["contactGapP99R"],
                    "passed": source_eyes["right"]["intersectionCount"] == 0,
                },
            },
            "passed": True,
        }
        atlas = Image.new("RGB", (256, 256), (164, 113, 92))
        atlas.save(run / "textures" / "head-albedo.jpg", quality=92)
        head.export_head_glb(run / "models" / "head.glb", atlas)
        Image.new("RGB", (256, 256), (32, 123, 255)).save(
            run / "textures" / "head-source.png"
        )
        Image.new("RGB", (256, 256), (230, 230, 230)).save(
            run / "textures" / "head-confidence.png"
        )

        projection_buffer = io.BytesIO()
        np.savez_compressed(
            projection_buffer,
            source_role=np.zeros(len(vertices), dtype=np.uint8),
            source_uv=np.zeros((len(vertices), 2), dtype=np.uint16),
            camera_depth=np.ones(len(vertices), dtype=np.float32),
            confidence=np.full(len(vertices), 0.9, dtype=np.float32),
            source_bits=np.ones(len(vertices), dtype=np.uint8),
            per_view_weights=np.tile(
                np.asarray([[1.0, 0.0, 0.0]], dtype=np.float16),
                (len(vertices), 1),
            ),
        )
        projection_path = run / "projection" / "skin-projection.npz"
        atomic_write_bytes(projection_path, projection_buffer.getvalue())
        projection_schema = run / "projection" / "schema.json"
        atomic_write_json(
            projection_schema,
            {
                "schemaVersion": 1,
                "recordCount": len(vertices),
                "recordDomain": "TemplateHeadV0 compute vertices",
                "viewOrder": ["front", "left45", "right45"],
            },
        )

        for role in ("front", "left45", "right45"):
            preview = template_root / "qa" / f"fixed-view-{role}.png"
            for relative in (
                f"references/{role}.png",
                f"overlays/landmarks-{role}.png",
                f"overlays/silhouette-{role}.png",
                f"qa/fixed-view-{role}.png",
                f"qa/fixed-view-skin-{role}.png",
                f"qa/registration-{role}.png",
            ):
                shutil.copyfile(preview, run / relative)
        side_preview = template_root / "qa" / "fixed-view-side.png"
        shutil.copyfile(side_preview, run / "qa" / "fixed-view-side.png")
        shutil.copyfile(side_preview, run / "qa" / "fixed-view-skin-side.png")
        atomic_write_json(run / "qa" / "anatomy.json", anatomy)
        report = {
            "schemaVersion": "3.0.0",
            "gates": [],
            "summary": {
                "automatedGatesPassed": True,
                "userSignoffRequired": True,
                "browserAuditRequired": True,
                "visualBaselineReviewed": False,
                "visualReviewStatus": "pendingUserSignoff",
                "finalAcceptance": False,
            },
        }
        atomic_write_json(run / "qa" / "report.json", report)
        per_view = {
            role: {
                "landmarkNME": 0.01,
                "landmarkErrorPx": 1.0,
                "silhouetteIoU": 0.96,
                "passed": True,
            }
            for role in ("front", "left45", "right45")
        }
        head_hash = sha256_file(run / "models" / "head.glb")
        skin_metrics = {
            "representation": "canonical-uv-zbuffer-multifrequency-projection",
            "model": "models/head.glb",
            "atlas": "textures/head-albedo.jpg",
            "confidenceMap": "textures/head-confidence.png",
            "sourceMap": "textures/head-source.png",
            "atlasResolution": [256, 256],
            "observedVertexFraction": 0.9,
            "atlasObservedFraction": 0.9,
            "meanProjectionConfidence": 0.9,
            "seamDeltaE00Median": 1.0,
            "seamDeltaE00P95": 2.0,
            "geometryHash": digest,
            "neutralGeometryHash": digest,
            "skinGeometryHash": digest,
            "maximumVertexDifference": 0.0,
            "neutralAndSkinSharePositionIndex": True,
            "geometryRecreated": False,
            "modelSha256": head_hash,
            "atlasSha256": sha256_file(run / "textures" / "head-albedo.jpg"),
            "confidenceSha256": sha256_file(run / "textures" / "head-confidence.png"),
            "sourceSha256": sha256_file(run / "textures" / "head-source.png"),
            "inferredRegions": ["rearCranium"],
            "passed": True,
        }
        manifest = {
            "schemaVersion": "3.0.0",
            "subjectProfile": "face-v3",
            "mode": "template-head-v0",
            "templateId": "TemplateHeadV0",
            "provenance": {
                "inputs": {
                    role: {
                        "sourceSha256": "11" * 32,
                        "normalizedSha256": "11" * 32,
                        "file": f"references/{role}.png",
                    }
                    for role in per_view
                },
                "models": {},
                "configSha256": "33" * 32,
                "codeSha256": "44" * 32,
                "viewforge3dVersion": "0.1.0",
            },
            "cameras": [],
            "fit": {
                "fittedGeometrySha256": digest,
                "perView": per_view,
                "orientation": {
                    "flippedTriangleCount": 0,
                    "minimumSignedAreaRatio": 1.0,
                },
                "passed": True,
            },
            "mesh": {
                "vertices": len(vertices),
                "triangles": len(faces),
                "watertight": True,
                "edgeManifold": True,
                "boundaryEdges": 0,
                "degenerateTriangles": 0,
                "normalVarianceReduction": 0.0,
                "hausdorffVoxels": 0.0,
                "model": "models/head.glb",
                "geometryHash": digest,
                **anatomy["geometry"],
            },
            "skin": skin_metrics,
            "projection": {
                "mapping": "final-template vertex to source-view pixel/depth/confidence",
                "recordCount": len(vertices),
                "binary": "projection/skin-projection.npz",
                "binarySha256": sha256_file(projection_path),
                "schema": "projection/schema.json",
                "schemaSha256": sha256_file(projection_schema),
                "sourceViewOrder": ["front", "left45", "right45"],
            },
            "anatomy": anatomy,
            "sdf": {
                "role": "qa-only",
                "surfaceGenerated": False,
                "gridAllocated": False,
                "finite": True,
                "outsideSignConsistency": 1.0,
                "insideSignConsistency": 1.0,
                "passed": True,
            },
            "confidence": {
                "mean": 0.9,
                "lowConfidenceRegions": ["rearCranium"],
                "templateInferredRegions": ["rearCranium"],
            },
            "runtime": {},
        }
        atomic_write_json(run / "manifest.json", manifest)
        package_run(run, output, load_config(ROOT / "configs" / "face-v3.yaml"))


if __name__ == "__main__":
    import sys

    create(Path(sys.argv[1]))
