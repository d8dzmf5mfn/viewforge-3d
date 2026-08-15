from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from face3d.io import atomic_write_bytes, atomic_write_json, sha256_file

MAGIC = b"P2D3"
VERSION = 1
VERSION_V2 = 2
HEADER = struct.Struct("<4sHHIHH4I32s")
RECORD = struct.Struct("<HHHHIfffffBBxx")
RECORD_V2 = struct.Struct("<HHBBBBIffffffffI")


def encode_pixels(
    *,
    model_uv: np.ndarray,
    source_uv: np.ndarray,
    pixel_codes: np.ndarray,
    positions: np.ndarray,
    thickness: np.ndarray,
    confidence: np.ndarray,
    source_bits: np.ndarray,
    feature_class: np.ndarray,
    grid_size: tuple[int, int],
    crop: tuple[int, int, int, int],
    source_sha256: str,
) -> bytes:
    model_uv = np.asarray(model_uv, dtype=np.uint16).reshape(-1, 2)
    source_uv = np.asarray(source_uv, dtype=np.uint16).reshape(-1, 2)
    pixel_codes = np.asarray(pixel_codes, dtype=np.uint32).reshape(-1)
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    thickness = np.asarray(thickness, dtype=np.float32).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    source_bits = np.asarray(source_bits, dtype=np.uint8).reshape(-1)
    feature_class = np.asarray(feature_class, dtype=np.uint8).reshape(-1)
    count = len(model_uv)
    values = (
        source_uv,
        pixel_codes,
        positions,
        thickness,
        confidence,
        source_bits,
        feature_class,
    )
    if any(len(value) != count for value in values):
        raise ValueError("pixel record attribute lengths must match")
    if len(source_sha256) != 64:
        raise ValueError("source_sha256 must be a 64-character SHA-256 hex digest")
    grid_width, grid_height = grid_size
    if max(grid_width, grid_height, *crop) > np.iinfo(np.uint16).max:
        raise ValueError("pixel coordinates exceed v1 uint16 limits")

    encoded = bytearray(
        HEADER.pack(
            MAGIC,
            VERSION,
            RECORD.size,
            count,
            grid_width,
            grid_height,
            *crop,
            bytes.fromhex(source_sha256),
        )
    )
    for index in range(count):
        encoded.extend(
            RECORD.pack(
                int(model_uv[index, 0]),
                int(model_uv[index, 1]),
                int(source_uv[index, 0]),
                int(source_uv[index, 1]),
                int(pixel_codes[index]),
                float(positions[index, 0]),
                float(positions[index, 1]),
                float(positions[index, 2]),
                float(thickness[index]),
                float(confidence[index]),
                int(source_bits[index]),
                int(feature_class[index]),
            )
        )
    return bytes(encoded)


def write_pixel_records(
    destination: Path,
    schema_destination: Path,
    *,
    source_bits_legend: dict[str, int] | None = None,
    **values: Any,
) -> dict[str, Any]:
    payload = encode_pixels(**values)
    atomic_write_bytes(destination, payload)
    grid_size = values["grid_size"]
    crop = values["crop"]
    count = int(len(values["model_uv"]))
    schema = {
        "schemaVersion": "1.0.0",
        "format": "viewforge3d-pixel-direct",
        "endianness": "little",
        "magic": MAGIC.decode("ascii"),
        "headerBytes": HEADER.size,
        "recordBytes": RECORD.size,
        "recordCount": count,
        "grid": {"width": int(grid_size[0]), "height": int(grid_size[1])},
        "sourceCropXYWH": [int(value) for value in crop],
        "sourceImageSha256": values["source_sha256"],
        "pixelCode": "RGB24: (R << 16) | (G << 8) | B",
        "recordLayout": [
            {"name": "modelU", "offset": 0, "type": "uint16"},
            {"name": "modelV", "offset": 2, "type": "uint16"},
            {"name": "sourceU", "offset": 4, "type": "uint16"},
            {"name": "sourceV", "offset": 6, "type": "uint16"},
            {"name": "pixelCode", "offset": 8, "type": "uint32"},
            {"name": "x", "offset": 12, "type": "float32"},
            {"name": "y", "offset": 16, "type": "float32"},
            {"name": "z", "offset": 20, "type": "float32"},
            {"name": "thickness", "offset": 24, "type": "float32"},
            {"name": "confidence", "offset": 28, "type": "float32"},
            {"name": "sourceBits", "offset": 32, "type": "uint8"},
            {"name": "featureClass", "offset": 33, "type": "uint8"},
        ],
        "featureClasses": {
            "0": "simpleInterpolated",
            "1": "eyesRefined",
            "2": "noseRefined",
            "3": "mouthRefined",
            "4": "earsRefined",
            "5": "jawRefined",
        },
        "sourceBits": source_bits_legend
        or {"front": 1, "left45": 2, "right45": 4, "templateInferred": 8},
    }
    atomic_write_json(schema_destination, schema)
    return {
        "records": count,
        "bytes": len(payload),
        "sha256": sha256_file(destination),
        "schemaSha256": sha256_file(schema_destination),
    }


def encode_pixels_v2(
    *,
    source_uv: np.ndarray,
    view_role: np.ndarray,
    target_node: np.ndarray,
    target_triangle: np.ndarray,
    barycentric: np.ndarray,
    positions: np.ndarray,
    depth: np.ndarray,
    feature_class: np.ndarray,
    confidence: np.ndarray,
    source_bits: np.ndarray,
    pixel_codes: np.ndarray,
    grid_size: tuple[int, int],
    crop: tuple[int, int, int, int],
    source_sha256: str,
) -> bytes:
    source_uv = np.asarray(source_uv, dtype=np.uint16).reshape(-1, 2)
    count = len(source_uv)
    arrays = {
        "view_role": np.asarray(view_role, dtype=np.uint8).reshape(-1),
        "target_node": np.asarray(target_node, dtype=np.uint8).reshape(-1),
        "target_triangle": np.asarray(target_triangle, dtype=np.uint32).reshape(-1),
        "barycentric": np.asarray(barycentric, dtype=np.float32).reshape(-1, 3),
        "positions": np.asarray(positions, dtype=np.float32).reshape(-1, 3),
        "depth": np.asarray(depth, dtype=np.float32).reshape(-1),
        "feature_class": np.asarray(feature_class, dtype=np.uint8).reshape(-1),
        "confidence": np.asarray(confidence, dtype=np.float32).reshape(-1),
        "source_bits": np.asarray(source_bits, dtype=np.uint8).reshape(-1),
        "pixel_codes": np.asarray(pixel_codes, dtype=np.uint32).reshape(-1),
    }
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("Face v2 pixel record attribute lengths must match")
    if len(source_sha256) != 64:
        raise ValueError("source_sha256 must be a 64-character SHA-256 hex digest")
    grid_width, grid_height = grid_size
    if max(grid_width, grid_height, *crop) > np.iinfo(np.uint16).max:
        raise ValueError("pixel coordinates exceed v2 uint16 limits")
    if not np.all(np.isfinite(arrays["positions"])) or not np.all(
        np.isfinite(arrays["barycentric"])
    ):
        raise ValueError("Face v2 pixel positions and barycentrics must be finite")
    barycentric_sum = arrays["barycentric"].sum(axis=1)
    if np.any(arrays["barycentric"] < -1e-5) or np.any(np.abs(barycentric_sum - 1.0) > 1e-4):
        raise ValueError("Face v2 barycentric coordinates are invalid")

    encoded = bytearray(
        HEADER.pack(
            MAGIC,
            VERSION_V2,
            RECORD_V2.size,
            count,
            grid_width,
            grid_height,
            *crop,
            bytes.fromhex(source_sha256),
        )
    )
    for index in range(count):
        encoded.extend(
            RECORD_V2.pack(
                int(source_uv[index, 0]),
                int(source_uv[index, 1]),
                int(arrays["view_role"][index]),
                int(arrays["target_node"][index]),
                int(arrays["feature_class"][index]),
                int(arrays["source_bits"][index]),
                int(arrays["target_triangle"][index]),
                *[float(value) for value in arrays["barycentric"][index]],
                *[float(value) for value in arrays["positions"][index]],
                float(arrays["depth"][index]),
                float(arrays["confidence"][index]),
                int(arrays["pixel_codes"][index]),
            )
        )
    return bytes(encoded)


def write_pixel_records_v2(
    destination: Path,
    schema_destination: Path,
    **values: Any,
) -> dict[str, Any]:
    payload = encode_pixels_v2(**values)
    atomic_write_bytes(destination, payload)
    count = int(len(values["source_uv"]))
    schema = {
        "schemaVersion": "2.0.0",
        "format": "viewforge3d-pixel-flame-hybrid",
        "endianness": "little",
        "magic": MAGIC.decode("ascii"),
        "headerBytes": HEADER.size,
        "recordBytes": RECORD_V2.size,
        "recordCount": count,
        "grid": {
            "width": int(values["grid_size"][0]),
            "height": int(values["grid_size"][1]),
        },
        "sourceCropXYWH": [int(value) for value in values["crop"]],
        "sourceSetSha256": values["source_sha256"],
        "targetNodes": {"0": "HeadSkin", "1": "Eyeball.L", "2": "Eyeball.R"},
        "viewRoles": {"0": "front", "1": "left45", "2": "right45", "255": "template"},
        "recordLayout": [
            {"name": "sourceU", "offset": 0, "type": "uint16"},
            {"name": "sourceV", "offset": 2, "type": "uint16"},
            {"name": "viewRole", "offset": 4, "type": "uint8"},
            {"name": "targetNode", "offset": 5, "type": "uint8"},
            {"name": "featureClass", "offset": 6, "type": "uint8"},
            {"name": "sourceBits", "offset": 7, "type": "uint8"},
            {"name": "targetTriangle", "offset": 8, "type": "uint32"},
            {"name": "barycentric", "offset": 12, "type": "float32[3]"},
            {"name": "positionXYZ", "offset": 24, "type": "float32[3]"},
            {"name": "cameraDepth", "offset": 36, "type": "float32"},
            {"name": "confidence", "offset": 40, "type": "float32"},
            {"name": "pixelCode", "offset": 44, "type": "uint32"},
        ],
        "featureClasses": {
            "0": "genericSurface",
            "1": "eyes",
            "2": "nose",
            "3": "mouth",
            "4": "ears",
            "5": "jaw",
        },
        "sourceBits": {"front": 1, "left45": 2, "right45": 4, "templateInferred": 8},
    }
    atomic_write_json(schema_destination, schema)
    return {
        "records": count,
        "bytes": len(payload),
        "sha256": sha256_file(destination),
        "schemaSha256": sha256_file(schema_destination),
    }


def decode_header(payload: bytes) -> dict[str, Any]:
    if len(payload) < HEADER.size:
        raise ValueError("pixel payload is smaller than the v1 header")
    magic, version, record_size, count, width, height, x, y, w, h, digest = HEADER.unpack_from(
        payload
    )
    supported = {VERSION: RECORD.size, VERSION_V2: RECORD_V2.size}
    if magic != MAGIC or version not in supported or record_size != supported[version]:
        raise ValueError("unsupported pixel payload")
    expected = HEADER.size + count * record_size
    if len(payload) != expected:
        raise ValueError("pixel payload length does not match its header")
    return {
        "version": version,
        "recordSize": record_size,
        "recordCount": count,
        "gridSize": [width, height],
        "crop": [x, y, w, h],
        "sourceSha256": digest.hex(),
    }
