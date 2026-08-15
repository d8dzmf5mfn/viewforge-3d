from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from face3d.io import atomic_write_json, read_json, sha256_file, sha256_json

STATE_SCHEMA_VERSION = 1


@dataclass(slots=True)
class StageState:
    name: str
    signature: str
    artifacts: list[str]
    metrics: dict[str, Any]


class RunState:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir.resolve()
        self.path = self.run_dir / "state.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.data = read_json(self.path)
        else:
            self.data = {"schemaVersion": STATE_SCHEMA_VERSION, "stages": {}}
        if self.data.get("schemaVersion") != STATE_SCHEMA_VERSION:
            self.data = {"schemaVersion": STATE_SCHEMA_VERSION, "stages": {}}

    def signature(self, inputs: dict[str, Any]) -> str:
        return sha256_json(inputs)

    def reusable(self, stage: str, signature: str) -> bool:
        record = self.data.get("stages", {}).get(stage)
        if not isinstance(record, dict) or record.get("signature") != signature:
            return False
        artifact_hashes = record.get("artifactHashes")
        if not isinstance(artifact_hashes, dict):
            return False
        for relative, expected_hash in artifact_hashes.items():
            path = self.run_dir / relative
            if not path.is_file() or sha256_file(path) != expected_hash:
                return False
        return True

    def metrics(self, stage: str) -> dict[str, Any]:
        record = self.data.get("stages", {}).get(stage, {})
        value = record.get("metrics", {}) if isinstance(record, dict) else {}
        return value if isinstance(value, dict) else {}

    def complete(
        self,
        stage: str,
        signature: str,
        artifacts: Iterable[Path],
        metrics: dict[str, Any] | None = None,
    ) -> None:
        hashes: dict[str, str] = {}
        for artifact in artifacts:
            resolved = artifact.resolve()
            relative = resolved.relative_to(self.run_dir).as_posix()
            hashes[relative] = sha256_file(resolved)
        stages = self.data.setdefault("stages", {})
        stages[stage] = {
            "signature": signature,
            "completedAt": datetime.now(UTC).isoformat(),
            "artifactHashes": hashes,
            "metrics": metrics or {},
        }
        atomic_write_json(self.path, self.data)
