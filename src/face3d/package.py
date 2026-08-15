from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from face3d.config import Face3DConfig
from face3d.errors import fail
from face3d.io import sha256_file
from face3d.models import REQUIRED_VIEWS
from face3d.report import write_notices

FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
FAST_LOAD_PREFIXES = ("models/", "references/")
FAST_LOAD_FILES = {"pixels/pixels.bin"}


def _archive_payload(source: Path, relative: str) -> bytes:
    image_folders = ("overlays/", "references/")
    if not relative.startswith(image_folders) or source.suffix.lower() != ".png":
        return source.read_bytes()
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        # The original normalized input remains in the run directory with its
        # SHA-256 in the manifest. The package only needs a review-sized copy;
        # keeping multi-megapixel RGB PNGs made local ZIP parsing miss Gate F.
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        image = image.quantize(
            colors=256,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True, compress_level=9)
        return output.getvalue()


def required_entries(schema_version: str = "1.0.0") -> list[str]:
    if schema_version == "3.0.0":
        entries = [
            "manifest.json",
            "models/head.glb",
            "textures/head-albedo.jpg",
            "textures/head-confidence.png",
            "textures/head-source.png",
            "projection/skin-projection.npz",
            "projection/schema.json",
            "qa/anatomy.json",
            "qa/report.json",
            "qa/fixed-view-side.png",
            "qa/fixed-view-skin-side.png",
            "THIRD_PARTY_NOTICES.md",
        ]
    elif schema_version == "2.0.0":
        entries = [
            "manifest.json",
            "models/voxels.glb",
            "models/head.glb",
            "textures/head-confidence.png",
            "textures/head-source.png",
            "pixels/pixels.bin",
            "pixels/schema.json",
            "qa/anatomy.json",
            "qa/report.json",
            "THIRD_PARTY_NOTICES.md",
        ]
    else:
        entries = [
            "manifest.json",
            "models/voxels.glb",
            "models/smooth.glb",
            "models/skin.glb",
            "textures/skin-atlas.jpg",
            "textures/skin-confidence.png",
            "pixels/pixels.bin",
            "pixels/schema.json",
            "qa/report.json",
            "THIRD_PARTY_NOTICES.md",
        ]
    for role in REQUIRED_VIEWS:
        entries.extend(
            (
                f"references/{role.value}.png",
                f"overlays/landmarks-{role.value}.png",
                f"overlays/silhouette-{role.value}.png",
                f"qa/fixed-view-{role.value}.png",
            )
        )
        if schema_version in {"2.0.0", "3.0.0"}:
            entries.append(f"qa/registration-{role.value}.png")
        if schema_version == "3.0.0":
            entries.append(f"qa/fixed-view-skin-{role.value}.png")
    return entries


def _validate_run(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        fail("package-invalid", "结果目录缺少 manifest.json", stage="package")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("schemaVersion")
    if schema_version not in {"1.0.0", "2.0.0", "3.0.0"}:
        fail("package-invalid", "manifest schemaVersion 不受支持", stage="package")
    missing = [
        entry
        for entry in required_entries(schema_version)
        if not (run_dir / entry).is_file()
    ]
    if missing:
        fail(
            "package-invalid",
            "结果目录缺少必需文件",
            stage="package",
            details={"missing": missing},
        )
    expected_mode = {
        "1.0.0": "pixel-direct",
        "2.0.0": "pixel-flame-hybrid",
        "3.0.0": "template-head-v0",
    }[schema_version]
    if manifest.get("mode") != expected_mode:
        fail("package-invalid", f"manifest mode 不是 {expected_mode}", stage="package")
    if schema_version in {"2.0.0", "3.0.0"}:
        mesh = manifest.get("mesh", {})
        skin = manifest.get("skin", {})
        geometry_hash = mesh.get("geometryHash")
        if (
            not isinstance(geometry_hash, str)
            or skin.get("skinGeometryHash") != geometry_hash
            or skin.get("neutralGeometryHash") != geometry_hash
            or skin.get("maximumVertexDifference") != 0.0
        ):
            fail(
                "package-invalid",
                "统一头模的灰模与人皮几何契约不一致",
                stage="package",
            )
        expected_head_hash = skin.get("modelSha256")
        if (
            not isinstance(expected_head_hash, str)
            or sha256_file(run_dir / "models" / "head.glb") != expected_head_hash
        ):
            fail("package-invalid", "head.glb 哈希不匹配", stage="package")
    trace = manifest.get("projection" if schema_version == "3.0.0" else "pixel")
    if not isinstance(trace, dict):
        label = "projection" if schema_version == "3.0.0" else "pixel"
        fail("package-invalid", f"manifest 缺少 {label} 追溯契约", stage="package")
    for path_key, hash_key in (
        ("binary", "binarySha256"),
        ("schema", "schemaSha256"),
    ):
        relative = trace.get(path_key)
        expected = trace.get(hash_key)
        if not isinstance(relative, str) or not isinstance(expected, str):
            fail("package-invalid", "manifest 缺少像素追溯哈希", stage="package")
        source = run_dir / relative
        if not source.is_file() or sha256_file(source) != expected:
            fail(
                "package-invalid",
                f"追溯文件哈希不匹配: {relative}",
                stage="package",
            )
    if schema_version == "3.0.0":
        skin = manifest["skin"]
        sdf = manifest.get("sdf", {})
        if (
            "pixel" in manifest
            or "voxel" in manifest
            or sdf.get("role") != "qa-only"
            or sdf.get("surfaceGenerated") is not False
            or sdf.get("gridAllocated") is not False
        ):
            fail(
                "package-invalid",
                "Face v3 禁止 voxel/pixel 最终表面，SDF 必须保持 QA-only",
                stage="package",
            )
        if (
            trace.get("binary") != "projection/skin-projection.npz"
            or trace.get("schema") != "projection/schema.json"
        ):
            fail("package-invalid", "Face v3 皮肤投影追溯路径无效", stage="package")
        try:
            projection_schema = json.loads(
                (run_dir / trace["schema"]).read_text(encoding="utf-8")
            )
            if projection_schema.get("recordCount") != trace.get("recordCount"):
                raise ValueError("projection schema recordCount mismatch")
            with np.load(run_dir / trace["binary"], allow_pickle=False) as payload:
                record_count = int(trace.get("recordCount", -1))
                expected_shapes = {
                    "source_role": (record_count,),
                    "source_uv": (record_count, 2),
                    "camera_depth": (record_count,),
                    "confidence": (record_count,),
                    "source_bits": (record_count,),
                    "per_view_weights": (record_count, 3),
                }
                if any(
                    name not in payload or payload[name].shape != shape
                    for name, shape in expected_shapes.items()
                ):
                    raise ValueError("projection arrays do not match recordCount")
                if not all(
                    np.all(np.isfinite(payload[name]))
                    for name in ("camera_depth", "confidence", "per_view_weights")
                ):
                    raise ValueError("projection arrays contain non-finite values")
        except Exception as exc:
            fail(
                "package-invalid",
                f"Face v3 皮肤投影追溯内容无效: {exc}",
                stage="package",
            )
        for relative, hash_key in (
            ("textures/head-albedo.jpg", "atlasSha256"),
            ("textures/head-confidence.png", "confidenceSha256"),
            ("textures/head-source.png", "sourceSha256"),
        ):
            if sha256_file(run_dir / relative) != skin.get(hash_key):
                fail(
                    "package-invalid",
                    f"皮肤投影纹理哈希不匹配: {relative}",
                    stage="package",
                )
    return manifest


def package_run(run_dir: Path, output: Path, config: Face3DConfig) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    write_notices(run_dir)
    manifest = _validate_run(run_dir)
    expected_schema = "3.0.0" if config.is_v3 else ("2.0.0" if config.is_v2 else "1.0.0")
    if manifest["schemaVersion"] != expected_schema:
        fail(
            "package-invalid",
            "打包配置与运行结果 schemaVersion 不一致",
            stage="package",
            details={"config": expected_schema, "run": manifest["schemaVersion"]},
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for relative in sorted(required_entries(manifest["schemaVersion"])):
                source = run_dir / relative
                compression = (
                    zipfile.ZIP_STORED
                    if relative.startswith(FAST_LOAD_PREFIXES) or relative in FAST_LOAD_FILES
                    else zipfile.ZIP_DEFLATED
                )
                info = zipfile.ZipInfo(relative, FIXED_ZIP_TIME)
                info.compress_type = compression
                info.external_attr = 0o100644 << 16
                archive.writestr(
                    info,
                    _archive_payload(source, relative),
                    compress_type=compression,
                    compresslevel=9,
                )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    size_mb = output.stat().st_size / 1024**2
    if size_mb > config.acceptance.package_size_mb_max:
        output.unlink(missing_ok=True)
        fail(
            "package-size-exceeded",
            "结果包超过大小门禁，已删除未通过的包",
            stage="package",
            details={"measuredMB": size_mb, "limitMB": config.acceptance.package_size_mb_max},
        )
    return {
        "ok": True,
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }
