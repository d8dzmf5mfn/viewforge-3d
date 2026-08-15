from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from face3d.config import load_config
from face3d.glb import export_pixel_instances
from face3d.io import atomic_write_json, sha256_file
from face3d.package import package_run
from face3d.pixel_binary import write_pixel_records_v2
from face3d.unified_head import EyeballAsset, UnifiedHeadAsset, geometry_hash

ROOT = Path(__file__).resolve().parent.parent


def create(output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="face-v2-contract-") as temporary:
        run = Path(temporary)
        for directory in ("models", "pixels", "qa", "references", "overlays", "textures"):
            (run / directory).mkdir(parents=True, exist_ok=True)
        skin = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        vertices = np.asarray(skin.vertices, dtype=np.float64)
        vertices[:, 1] *= 1.18
        vertices[:, 2] *= 1.08
        faces = np.asarray(skin.faces, dtype=np.int64)
        mapping = np.arange(len(vertices), dtype=np.int64)
        uv = np.column_stack(
            (
                0.5 + np.arctan2(vertices[:, 0], vertices[:, 2]) / (2 * np.pi),
                0.5
                - np.arcsin(np.clip(vertices[:, 1] / 1.18, -1, 1)) / np.pi,
            )
        ).astype(np.float32)
        digest = geometry_hash(vertices, faces)
        left_eye = EyeballAsset(
            np.asarray([-0.32, 0.16, 0.88]),
            0.14,
            np.asarray([0.0, 0.0, 1.0]),
        )
        right_eye = EyeballAsset(
            np.asarray([0.32, 0.16, 0.88]),
            0.14,
            np.asarray([0.0, 0.0, 1.0]),
        )
        anatomy = {
            "schemaVersion": "2.0.0",
            "unifiedHead": {
                "connectedComponents": 1,
                "boundaryEdges": 0,
                "nonManifoldEdges": 0,
                "degenerateTriangles": 0,
                "topCurvatureSpikeRatio": 1.1,
                "geometryHash": digest,
            },
            "ears": {
                "source": "contract-fixture",
                "carrierPresent": False,
                "rootSharedWithScalp": True,
            },
            "eyes": {
                "completeEyeballNodes": 2,
                "contactGapP99R": 0.015,
                "radiusDifferenceRatio": 0.0,
                "irisReprojectionErrorPx": 1.0,
                "penetrationCount": 0,
            },
        }
        head = UnifiedHeadAsset(
            vertices,
            faces,
            mapping,
            faces,
            uv,
            {"left_ear": np.asarray([0]), "right_ear": np.asarray([1])},
            left_eye,
            right_eye,
            digest,
            anatomy,
        )
        atlas = Image.new("RGB", (256, 256), (164, 113, 92))
        head.export_head_glb(run / "models" / "head.glb", atlas)
        positions = np.asarray([[0.0, 0.0, 1.08]], dtype=np.float32)
        export_pixel_instances(
            positions,
            np.asarray([[0.05, 0.05, 0.01]], dtype=np.float32),
            np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            np.asarray([0xA4715C], dtype=np.uint32),
            np.asarray([[32, 32]], dtype=np.uint16),
            np.asarray([2.0], dtype=np.float32),
            np.asarray([0], dtype=np.uint8),
            np.asarray([0.9], dtype=np.float32),
            np.asarray([1], dtype=np.uint8),
            run / "models" / "voxels.glb",
            contract="v2",
        )
        binary = write_pixel_records_v2(
            run / "pixels" / "pixels.bin",
            run / "pixels" / "schema.json",
            source_uv=np.asarray([[32, 32]], dtype=np.uint16),
            view_role=np.asarray([0], dtype=np.uint8),
            target_node=np.asarray([0], dtype=np.uint8),
            target_triangle=np.asarray([0], dtype=np.uint32),
            barycentric=np.asarray([[1 / 3, 1 / 3, 1 / 3]], dtype=np.float32),
            positions=positions,
            depth=np.asarray([2.0], dtype=np.float32),
            feature_class=np.asarray([0], dtype=np.uint8),
            confidence=np.asarray([0.9], dtype=np.float32),
            source_bits=np.asarray([1], dtype=np.uint8),
            pixel_codes=np.asarray([0xA4715C], dtype=np.uint32),
            grid_size=(384, 384),
            crop=(0, 0, 64, 64),
            source_sha256="22" * 32,
        )
        source = Image.new("RGB", (256, 256), (32, 123, 255))
        confidence = Image.new("RGB", (256, 256), (230, 230, 230))
        source.save(run / "textures" / "head-source.png")
        confidence.save(run / "textures" / "head-confidence.png")
        for role in ("front", "left45", "right45"):
            reference = Image.new("RGB", (128, 128), (124, 91, 78))
            for relative in (
                f"references/{role}.png",
                f"overlays/landmarks-{role}.png",
                f"overlays/silhouette-{role}.png",
                f"qa/fixed-view-{role}.png",
                f"qa/registration-{role}.png",
            ):
                reference.save(run / relative)
        atomic_write_json(run / "qa" / "anatomy.json", anatomy)
        report = {
            "schemaVersion": "2.0.0",
            "gates": [],
            "summary": {
                "automatedGatesPassed": True,
                "userSignoffRequired": True,
                "browserAuditRequired": True,
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
        manifest = {
            "schemaVersion": "2.0.0",
            "subjectProfile": "face-v2",
            "mode": "pixel-flame-hybrid",
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
            "pixel": {
                "mapping": "contract-fixture",
                "gridSize": [384, 384, 384],
                "instanceCount": 1,
                "complexPixelCount": 0,
                "simpleInterpolatedPixelCount": 1,
                "binary": "pixels/pixels.bin",
                "binarySha256": binary["sha256"],
                "schema": "pixels/schema.json",
                "schemaSha256": binary["schemaSha256"],
            },
            "voxel": {
                "representation": "contract-fixture",
                "resolution": [384, 384, 384],
                "voxelSize": 0.01,
                "instanceCount": 1,
                "surfaceCellCoverage": 1.0,
                "maximumSurfaceDistanceVoxels": 0.0,
                "maximumInstances": 200000,
                "model": "models/voxels.glb",
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
                "nodes": ["HeadSkin", "Eyeball.L", "Eyeball.R"],
            },
            "skin": {
                "representation": "contract-fixture",
                "model": "models/head.glb",
                "confidenceMap": "textures/head-confidence.png",
                "sourceMap": "textures/head-source.png",
                "atlasResolution": [256, 256],
                "observedVertexFraction": 0.9,
                "atlasObservedFraction": 0.9,
                "seamDeltaE00Median": 1.0,
                "seamDeltaE00P95": 2.0,
                "geometryHash": digest,
                "neutralGeometryHash": digest,
                "skinGeometryHash": digest,
                "maximumVertexDifference": 0.0,
                "modelSha256": head_hash,
                "inferredRegions": ["rearCranium"],
            },
            "fit": {"perView": per_view},
            "anatomy": anatomy,
            "confidence": {
                "mean": 0.9,
                "lowConfidenceRegions": ["rearCranium"],
                "templateInferredRegions": ["rearCranium"],
            },
            "runtime": {},
        }
        atomic_write_json(run / "manifest.json", manifest)
        package_run(run, output, load_config(ROOT / "configs" / "face-v2.yaml"))


if __name__ == "__main__":
    import sys

    create(Path(sys.argv[1]))
