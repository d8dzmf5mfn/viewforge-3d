from pathlib import Path

import numpy as np

from face3d.pixel_binary import (
    HEADER,
    RECORD,
    RECORD_V2,
    decode_header,
    write_pixel_records,
    write_pixel_records_v2,
)


def test_pixel_binary_preserves_source_code_and_coordinates(tmp_path: Path) -> None:
    destination = tmp_path / "pixels.bin"
    schema = tmp_path / "schema.json"
    source_hash = "12" * 32
    result = write_pixel_records(
        destination,
        schema,
        model_uv=np.asarray([[2, 3], [4, 5]], dtype=np.uint16),
        source_uv=np.asarray([[120, 240], [121, 240]], dtype=np.uint16),
        pixel_codes=np.asarray([0x123456, 0xABCDEF], dtype=np.uint32),
        positions=np.asarray([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=np.float32),
        thickness=np.asarray([0.01, 0.01], dtype=np.float32),
        confidence=np.asarray([0.9, 0.8], dtype=np.float32),
        source_bits=np.asarray([7, 7], dtype=np.uint8),
        feature_class=np.asarray([0, 2], dtype=np.uint8),
        grid_size=(320, 280),
        crop=(20, 30, 700, 820),
        source_sha256=source_hash,
    )
    payload = destination.read_bytes()
    assert decode_header(payload) == {
        "version": 1,
        "recordSize": RECORD.size,
        "recordCount": 2,
        "gridSize": [320, 280],
        "crop": [20, 30, 700, 820],
        "sourceSha256": source_hash,
    }
    first = RECORD.unpack_from(payload, HEADER.size)
    assert first[:5] == (2, 3, 120, 240, 0x123456)
    assert first[-2:] == (7, 0)
    assert result["records"] == 2
    assert result["bytes"] == HEADER.size + 2 * RECORD.size


def test_pixel_binary_v2_traces_target_triangle_and_barycentric(tmp_path: Path) -> None:
    destination = tmp_path / "pixels-v2.bin"
    schema = tmp_path / "schema-v2.json"
    source_hash = "ab" * 32
    result = write_pixel_records_v2(
        destination,
        schema,
        source_uv=np.asarray([[120, 240], [360, 480]], dtype=np.uint16),
        view_role=np.asarray([0, 255], dtype=np.uint8),
        target_node=np.asarray([0, 2], dtype=np.uint8),
        target_triangle=np.asarray([42, 9], dtype=np.uint32),
        barycentric=np.asarray([[0.2, 0.3, 0.5], [1 / 3, 1 / 3, 1 / 3]]),
        positions=np.asarray([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=np.float32),
        depth=np.asarray([2.0, 0.0], dtype=np.float32),
        feature_class=np.asarray([1, 1], dtype=np.uint8),
        confidence=np.asarray([0.9, 0.2], dtype=np.float32),
        source_bits=np.asarray([1, 8], dtype=np.uint8),
        pixel_codes=np.asarray([0x123456, 0xEEEEEE], dtype=np.uint32),
        grid_size=(384, 384),
        crop=(0, 0, 1024, 1024),
        source_sha256=source_hash,
    )
    payload = destination.read_bytes()
    assert decode_header(payload) == {
        "version": 2,
        "recordSize": RECORD_V2.size,
        "recordCount": 2,
        "gridSize": [384, 384],
        "crop": [0, 0, 1024, 1024],
        "sourceSha256": source_hash,
    }
    first = RECORD_V2.unpack_from(payload, HEADER.size)
    assert first[:7] == (120, 240, 0, 0, 1, 1, 42)
    assert result["bytes"] == HEADER.size + 2 * RECORD_V2.size
