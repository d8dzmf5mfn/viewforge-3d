from __future__ import annotations

import json
import os
import shutil
import struct
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

from face3d.io import (
    atomic_write_bytes,
    atomic_write_json,
    package_code_hash,
    sha256_file,
)
from face3d.phone_v1 import (
    FIXED_VIEWS,
    PhoneDimensions,
    build_phone_scene,
    build_template_phone_v0,
    export_phone_glb,
    primary_surface_metrics,
    render_phone_scene,
)
from face3d.template_head_anatomy import _self_intersection_pairs
from face3d.template_head_v0 import _edge_and_component_metrics

APPLE_SPECS_URL = "https://www.apple.com/iphone-17/specs/"
APPLE_DIMENSIONAL_DRAWING_URL = (
    "https://developer.apple.com/download/files/accessories/dimensional-drawings/iphone-17.pdf"
)
APPLE_NEWSROOM_URL = "https://www.apple.com/newsroom/2025/09/apple-debuts-iphone-17/"


@dataclass(frozen=True, slots=True)
class ReferenceImage:
    label: str
    source_url: str
    path: Path
    evidence_role: str


def _component_quality(scene: trimesh.Scene) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, geometry in sorted(scene.geometry.items()):
        if name == "PhonePrimarySurface":
            continue
        mesh = geometry.copy()
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        topology = _edge_and_component_metrics(vertices, faces)
        self_intersections = _self_intersection_pairs(mesh)
        passed = bool(
            topology["componentCount"] == 1
            and topology["boundaryEdgeCount"] == 0
            and topology["nonManifoldEdgeCount"] == 0
            and topology["degenerateFaceCount"] == 0
            and topology["duplicateFaceCount"] == 0
            and topology["duplicateVertexCount"] == 0
            and topology["watertight"]
            and topology["windingConsistent"]
            and mesh.volume > 0.0
            and len(self_intersections) == 0
        )
        records[name] = {
            "vertices": int(len(vertices)),
            "triangles": int(len(faces)),
            "extentsMm": np.asarray(mesh.extents, dtype=float).tolist(),
            "volumeMm3": float(mesh.volume),
            "connectedComponents": int(topology["componentCount"]),
            "boundaryEdges": int(topology["boundaryEdgeCount"]),
            "nonManifoldEdges": int(topology["nonManifoldEdgeCount"]),
            "degenerateFaces": int(topology["degenerateFaceCount"]),
            "duplicateFaces": int(topology["duplicateFaceCount"]),
            "duplicateVertices": int(topology["duplicateVertexCount"]),
            "watertight": bool(topology["watertight"]),
            "windingConsistent": bool(topology["windingConsistent"]),
            "selfIntersectionPairs": int(len(self_intersections)),
            "passed": passed,
        }
    return {
        "components": records,
        "componentCount": len(records),
        "passed": bool(records and all(record["passed"] for record in records.values())),
        "assemblyPolicy": (
            "Attached feature components are individually closed; the primary surface is "
            "checked separately on its welded compute topology. Intentional attachment "
            "overlaps are retained in this geometry preview."
        ),
    }


def _glb_json(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if len(payload) < 20:
        raise ValueError("GLB is too short")
    magic, version, declared_length = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(payload):
        raise ValueError("invalid GLB header")
    offset = 12
    while offset + 8 <= len(payload):
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk = payload[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            value = json.loads(chunk.decode("utf-8").rstrip(" \t\r\n\0"))
            if not isinstance(value, dict):
                raise TypeError("GLB JSON root must be an object")
            return value
    raise ValueError("GLB JSON chunk is missing")


def _roundtrip_quality(path: Path, expected_geometry_names: set[str]) -> dict[str, Any]:
    gltf = _glb_json(path)
    scene = trimesh.load(path, force="scene", process=False)
    if not isinstance(scene, trimesh.Scene):
        raise TypeError("GLB roundtrip did not produce a scene")
    geometry_names = set(scene.geometry)
    primary = scene.geometry.get("PhonePrimarySurface")
    primary_extents = (
        np.asarray(primary.extents, dtype=float).tolist() if primary is not None else None
    )
    dimensions = PhoneDimensions.iphone17()
    primary_dimensions_passed = bool(
        primary_extents is not None
        and np.allclose(
            primary_extents,
            [dimensions.width_mm, dimensions.height_mm, dimensions.depth_mm],
            atol=1e-4,
        )
    )
    passed = bool(
        geometry_names == expected_geometry_names
        and primary_dimensions_passed
        and not gltf.get("images")
        and not gltf.get("textures")
    )
    return {
        "format": "glb-2.0",
        "fileSha256": sha256_file(path),
        "fileBytes": path.stat().st_size,
        "geometryCount": len(geometry_names),
        "geometryNames": sorted(geometry_names),
        "expectedGeometryNames": sorted(expected_geometry_names),
        "primaryExtentsMm": primary_extents,
        "primaryDimensionsPassed": primary_dimensions_passed,
        "imageTextureCount": len(gltf.get("images", [])),
        "textureCount": len(gltf.get("textures", [])),
        "noImageTextures": not gltf.get("images") and not gltf.get("textures"),
        "sceneExtentsMm": np.asarray(scene.extents, dtype=float).tolist(),
        "passed": passed,
    }


def _copy_reference_images(
    staging: Path,
    references: tuple[ReferenceImage, ...],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, reference in enumerate(references, start=1):
        source = reference.path.resolve()
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"reference image is unreadable: {source}")
        height, width = image.shape[:2]
        minimum_side = min(width, height)
        if minimum_side < 1024:
            raise ValueError(
                f"reference image minimum side must be at least 1024 px: {source} "
                f"({width}x{height})"
            )
        suffix = source.suffix.lower() or ".jpg"
        destination = staging / "evidence" / f"reference-{index:02d}{suffix}"
        atomic_write_bytes(destination, source.read_bytes())
        records.append(
            {
                "label": reference.label,
                "sourceUrl": reference.source_url,
                "localPath": destination.relative_to(staging).as_posix(),
                "sha256": sha256_file(destination),
                "widthPx": width,
                "heightPx": height,
                "minimumSidePx": minimum_side,
                "minimumSideGatePx": 1024,
                "minimumSideGatePassed": True,
                "evidenceRole": reference.evidence_role,
                "usedForTexture": False,
            }
        )
    return records


def _file_checksums(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }


def build_iphone17_unskinned_run(
    destination: Path,
    *,
    references: tuple[ReferenceImage, ...],
    render_size: int = 1024,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"immutable run destination already exists: {destination}")
    if not references:
        raise ValueError("at least one admitted official reference image is required")
    if render_size < 128:
        raise ValueError("render_size must be at least 128")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        reference_records = _copy_reference_images(staging, references)
        template = build_template_phone_v0()
        dimensions = PhoneDimensions.iphone17()
        fitted = template.fit_dimensions(dimensions)
        template_path = staging / "working" / "template-phone-v0.npz"
        fitted_path = staging / "working" / "iphone17-fitted.npz"
        template.save(template_path)
        fitted.save(fitted_path)

        model_path = staging / "models" / "iphone17-unskinned.glb"
        export_phone_glb(fitted, dimensions, model_path)
        scene = build_phone_scene(fitted, dimensions)
        fixed_views: dict[str, str] = {}
        for view in FIXED_VIEWS:
            relative = Path("qa") / f"fixed-view-{view}.png"
            render_phone_scene(
                scene,
                staging / relative,
                view=view,
                width=render_size,
                height=render_size,
            )
            fixed_views[view] = relative.as_posix()

        primary_quality = primary_surface_metrics(fitted)
        component_quality = _component_quality(scene)
        roundtrip_quality = _roundtrip_quality(model_path, set(scene.geometry))
        quality = {
            "schemaVersion": 1,
            "primarySurface": primary_quality,
            "components": component_quality,
            "glbRoundtrip": roundtrip_quality,
            "fixedViews": fixed_views,
            "fixedViewSizePx": [render_size, render_size],
            "skinApplied": False,
            "imageTexturesApplied": False,
            "sdf": {"role": "qa-only", "surfaceGenerated": False},
            "passed": bool(
                primary_quality["passed"]
                and component_quality["passed"]
                and roundtrip_quality["passed"]
                and len(fixed_views) == len(FIXED_VIEWS)
            ),
        }
        atomic_write_json(staging / "qa" / "geometry-quality.json", quality)
        if not quality["passed"]:
            raise RuntimeError("iPhone 17 geometry quality gate failed")

        evidence_ledger = {
            "schemaVersion": 1,
            "subject": "Apple iPhone 17 base model",
            "sources": [
                {
                    "url": APPLE_SPECS_URL,
                    "authority": "Apple product specifications",
                    "usedFor": ["width", "height", "depth", "display-size"],
                },
                {
                    "url": APPLE_DIMENSIONAL_DRAWING_URL,
                    "authority": "Apple accessory dimensional drawing",
                    "usedFor": ["camera-layout", "front-sensor-layout", "keepout-dimensions"],
                    "redistribution": "source URL only; drawing not packaged",
                },
                {
                    "url": APPLE_NEWSROOM_URL,
                    "authority": "Apple Newsroom",
                    "usedFor": ["dual-camera appearance", "contoured-edge appearance"],
                },
            ],
            "images": reference_records,
            "classification": {
                "authoritative": [
                    "width_mm",
                    "height_mm",
                    "depth_mm",
                    "dynamic_island_width_mm",
                    "dynamic_island_height_mm",
                ],
                "bounded2dFit": [
                    "corner_radius_mm",
                    "front_glass_width_mm",
                    "front_glass_height_mm",
                    "camera_plateau_width_mm",
                    "camera_plateau_height_mm",
                    "camera_plateau_raise_mm",
                    "camera_glass_raise_mm",
                    "rear_camera_outer_diameter_mm",
                ],
                "appearanceOnly": ["side-button-shape", "rear-microphone", "usb-c-recess"],
                "unobservedInterior": "not modeled",
            },
        }
        atomic_write_json(staging / "evidence" / "ledger.json", evidence_ledger)

        timestamp = generated_at or datetime.now(UTC)
        manifest = {
            "schemaVersion": "1.0.0",
            "runId": destination.name,
            "generatedAt": timestamp.astimezone(UTC).isoformat(),
            "artifactType": "geometry-preview",
            "state": "geometry-preview",
            "subjectProfile": {
                "profileId": "iPhone17",
                "targetClass": "smartphone",
                "realPerson": False,
                "topologyFamily": "TemplatePhoneV0",
            },
            "model": "models/iphone17-unskinned.glb",
            "dimensionsMm": dimensions.as_json(),
            "surfaceLineage": (
                "Apple official 2D/dimensional evidence -> project-authored TemplatePhoneV0 -> "
                "bounded dimension fit -> attached-feature geometry preview -> QA"
            ),
            "origin": "project-authored-no-external-3d-input",
            "external3dImported": False,
            "skinApplied": False,
            "imageTexturesApplied": False,
            "materials": {
                "purpose": "neutral geometry visualization only",
                "imageTextures": False,
                "color": "lavender reference color factors",
            },
            "primarySurface": {
                "templateId": "TemplatePhoneV0",
                "geometrySha256": fitted.geometry_sha256,
                "uvSha256": fitted.uv_sha256,
                "topologyChangedDuringFit": False,
            },
            "attachedFeatures": {
                "integration": "separate closed components with intentional attachment overlap",
                "acceptedAsFinal": False,
            },
            "evidence": "evidence/ledger.json",
            "quality": "qa/geometry-quality.json",
            "checksums": "checksums.json",
            "codeSha256": package_code_hash(),
            "acceptance": {
                "geometryQaPassed": True,
                "userSignoffRequired": True,
                "userSignoff": False,
                "deliveryState": "preview-not-final",
            },
            "stages": {
                "intake": "passed",
                "subjectProfile": "passed",
                "templateFit": "passed",
                "attachedFeatures": "preview-passed-component-qa",
                "skin": "skipped-by-user-request",
                "geometryQa": "passed",
                "userAcceptance": "pending",
            },
        }
        atomic_write_json(staging / "manifest.json", manifest)
        atomic_write_json(staging / "checksums.json", _file_checksums(staging))
        os.replace(staging, destination)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
