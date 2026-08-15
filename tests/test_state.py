from pathlib import Path

from face3d.state import RunState


def test_stage_reuse_requires_matching_artifact_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "working" / "artifact.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"first")
    state = RunState(tmp_path)
    signature = state.signature({"stage": "example", "input": "a"})
    state.complete("example", signature, [artifact], {"value": 1})
    assert RunState(tmp_path).reusable("example", signature)
    artifact.write_bytes(b"changed")
    assert not RunState(tmp_path).reusable("example", signature)
