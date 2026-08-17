from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import (
    ArtifactSummary,
    AssetKind,
    AssetSummary,
    JobKind,
    JobState,
    JobSummary,
)

SERVER_VERSION = "0.3.1"
ALLOWED_EXTENSIONS: dict[str, AssetKind] = {
    ".glb": AssetKind.MODEL,
    ".gltf": AssetKind.MODEL,
    ".blend": AssetKind.BLEND,
    ".yaml": AssetKind.CONFIG,
    ".yml": AssetKind.CONFIG,
    ".json": AssetKind.JSON,
    ".png": AssetKind.IMAGE,
    ".jpg": AssetKind.IMAGE,
    ".jpeg": AssetKind.IMAGE,
    ".mov": AssetKind.VIDEO,
    ".mp4": AssetKind.VIDEO,
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class LocalConfiguration:
    schema_version: int = 1
    workspace_root: str | None = None
    blender_executable: str | None = None
    plugin_root: str | None = None
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8765


class LocalPaths:
    def __init__(self, state_root: Path | None = None) -> None:
        configured = os.environ.get("VIEWFORGE_LOCAL_STATE_DIR")
        if state_root is not None:
            root = state_root
        elif configured:
            root = Path(configured).expanduser()
        else:
            root = Path.home() / "Library" / "Application Support" / "ViewForge Local"
        self.root = root.resolve()
        self.config = self.root / "config.json"
        self.assets_index = self.root / "assets" / "index.json"
        self.artifacts_index = self.root / "artifacts" / "index.json"
        self.jobs = self.root / "jobs"

    def ensure(self) -> None:
        for directory in (
            self.root,
            self.assets_index.parent,
            self.artifacts_index.parent,
            self.jobs,
        ):
            directory.mkdir(parents=True, exist_ok=True)


class ConfigurationStore:
    def __init__(self, paths: LocalPaths) -> None:
        self.paths = paths

    def load(self) -> LocalConfiguration:
        payload = read_json(self.paths.config, {})
        if not isinstance(payload, dict):
            payload = {}
        workspace = os.environ.get("VIEWFORGE_WORKSPACE_ROOT") or payload.get("workspaceRoot")
        blender = os.environ.get("VIEWFORGE_BLENDER_PATH") or payload.get("blenderExecutable")
        plugin = os.environ.get("VIEWFORGE_PLUGIN_ROOT") or payload.get("pluginRoot")
        host = os.environ.get("VIEWFORGE_MCP_HOST") or payload.get("mcpHost") or "127.0.0.1"
        raw_port = os.environ.get("VIEWFORGE_MCP_PORT") or payload.get("mcpPort") or 8765
        try:
            port = int(raw_port)
        except (TypeError, ValueError):
            port = 8765
        return LocalConfiguration(
            workspace_root=str(workspace) if workspace else None,
            blender_executable=str(blender) if blender else None,
            plugin_root=str(plugin) if plugin else None,
            mcp_host=str(host),
            mcp_port=port,
        )

    def save(self, configuration: LocalConfiguration) -> None:
        self.paths.ensure()
        atomic_write_json(
            self.paths.config,
            {
                "schemaVersion": configuration.schema_version,
                "workspaceRoot": configuration.workspace_root,
                "blenderExecutable": configuration.blender_executable,
                "pluginRoot": configuration.plugin_root,
                "mcpHost": configuration.mcp_host,
                "mcpPort": configuration.mcp_port,
            },
        )

    def workspace(self) -> Path | None:
        raw = self.load().workspace_root
        if not raw:
            return None
        candidate = Path(raw).expanduser().resolve()
        return candidate if candidate.is_dir() else None

    def blender(self) -> Path | None:
        configured = self.load().blender_executable
        candidates = [
            Path(configured).expanduser() if configured else None,
            Path("/Applications/Blender.app/Contents/MacOS/Blender"),
            Path.home() / "Applications" / "Blender.app" / "Contents" / "MacOS" / "Blender",
        ]
        for candidate in candidates:
            if candidate and candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve()
        return None

    def plugin_root(self) -> Path | None:
        configured = self.load().plugin_root
        candidates: list[Path] = []
        if configured:
            candidates.append(Path(configured).expanduser())
        repository_root = Path(__file__).resolve().parents[3]
        candidates.append(repository_root / "plugins" / "viewforge-3d-toolkit")
        for candidate in candidates:
            manifest = candidate / ".codex-plugin" / "plugin.json"
            if manifest.is_file():
                return candidate.resolve()
        return None

    def modeling_runtime_available(self) -> bool:
        required_modules = (
            "cv2",
            "mediapipe",
            "numpy",
            "scipy",
            "skimage",
            "trimesh",
            "face3d.pipeline",
            "face3d.six_view_visual_hull",
        )
        return all(importlib.util.find_spec(name) is not None for name in required_modules)


@dataclass
class StoredAsset:
    id: str
    path: str
    name: str
    kind: str
    extension: str
    size_bytes: int
    sha256: str
    registered_at: str

    def public(self) -> AssetSummary:
        return AssetSummary(
            id=self.id,
            name=self.name,
            kind=AssetKind(self.kind),
            extension=self.extension,
            size_bytes=self.size_bytes,
            sha256=self.sha256,
        )


@dataclass
class StoredArtifact:
    id: str
    job_id: str
    path: str
    name: str
    extension: str
    size_bytes: int
    sha256: str
    created_at: str

    def public(self) -> ArtifactSummary:
        return ArtifactSummary(
            id=self.id,
            job_id=self.job_id,
            name=self.name,
            extension=self.extension,
            size_bytes=self.size_bytes,
            sha256=self.sha256,
        )


@dataclass
class StoredJob:
    id: str
    kind: str
    state: str
    created_at: str
    updated_at: str
    request_path: str
    artifact_ids: list[str] = field(default_factory=list)
    status_message: str | None = None
    failure: str | None = None
    worker_pid: int | None = None
    blender_pid: int | None = None
    exit_code: int | None = None


class AssetStore:
    def __init__(self, paths: LocalPaths, configuration: ConfigurationStore) -> None:
        self.paths = paths
        self.configuration = configuration

    def _records(self) -> dict[str, StoredAsset]:
        payload = read_json(self.paths.assets_index, {"schemaVersion": 1, "assets": []})
        records = payload.get("assets", []) if isinstance(payload, dict) else []
        return {record["id"]: StoredAsset(**record) for record in records}

    def _write(self, records: dict[str, StoredAsset]) -> None:
        atomic_write_json(
            self.paths.assets_index,
            {"schemaVersion": 1, "assets": [asdict(records[key]) for key in sorted(records)]},
        )

    def register(self, raw_path: str) -> AssetSummary:
        workspace = self.configuration.workspace()
        if workspace is None:
            raise ValueError(
                "Select a local workspace in ViewForge Local before registering assets."
            )
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = workspace / candidate
        path = candidate.resolve()
        if not path.is_file() or not is_within(path, workspace):
            raise ValueError("The asset must be an existing file inside the configured workspace.")
        extension = path.suffix.lower()
        kind = ALLOWED_EXTENSIONS.get(extension)
        if kind is None:
            raise ValueError("Unsupported local asset type.")
        digest = sha256_file(path)
        record = StoredAsset(
            id=f"asset_{digest[:24]}",
            path=str(path),
            name=path.name,
            kind=kind.value,
            extension=extension,
            size_bytes=path.stat().st_size,
            sha256=digest,
            registered_at=utc_now(),
        )
        records = self._records()
        records[record.id] = record
        self._write(records)
        return record.public()

    def resolve(self, asset_id: str) -> Path:
        record = self._records().get(asset_id)
        if record is None:
            raise ValueError("Unknown asset ID.")
        path = Path(record.path).resolve()
        workspace = self.configuration.workspace()
        if workspace is None or not path.is_file() or not is_within(path, workspace):
            raise ValueError("The registered asset is no longer available in the workspace.")
        if sha256_file(path) != record.sha256:
            raise ValueError("The registered asset changed; register it again before use.")
        return path

    def list(self) -> list[AssetSummary]:
        return [record.public() for record in self._records().values()]


class ArtifactStore:
    def __init__(self, paths: LocalPaths) -> None:
        self.paths = paths

    def _records(self) -> dict[str, StoredArtifact]:
        payload = read_json(self.paths.artifacts_index, {"schemaVersion": 1, "artifacts": []})
        records = payload.get("artifacts", []) if isinstance(payload, dict) else []
        return {record["id"]: StoredArtifact(**record) for record in records}

    def _write(self, records: dict[str, StoredArtifact]) -> None:
        atomic_write_json(
            self.paths.artifacts_index,
            {
                "schemaVersion": 1,
                "artifacts": [asdict(records[key]) for key in sorted(records)],
            },
        )

    def register(self, job_id: str, path: Path) -> ArtifactSummary:
        resolved = path.resolve()
        job_root = (self.paths.jobs / job_id).resolve()
        if not resolved.is_file() or not is_within(resolved, job_root):
            raise ValueError("Job artifact is outside the immutable job directory.")
        digest = sha256_file(resolved)
        identity = hashlib.sha256(f"{job_id}:{digest}".encode()).hexdigest()
        record = StoredArtifact(
            id=f"artifact_{identity[:24]}",
            job_id=job_id,
            path=str(resolved),
            name=resolved.name,
            extension=resolved.suffix.lower(),
            size_bytes=resolved.stat().st_size,
            sha256=digest,
            created_at=utc_now(),
        )
        records = self._records()
        records[record.id] = record
        self._write(records)
        return record.public()

    def resolve(self, artifact_id: str) -> Path:
        record = self._records().get(artifact_id)
        if record is None:
            raise ValueError("Unknown artifact ID.")
        path = Path(record.path).resolve()
        if not path.is_file() or not is_within(path, self.paths.jobs.resolve()):
            raise ValueError("The artifact is no longer available.")
        if sha256_file(path) != record.sha256:
            raise ValueError("The artifact failed its integrity check.")
        return path

    def get(self, artifact_id: str) -> ArtifactSummary:
        record = self._records().get(artifact_id)
        if record is None:
            raise ValueError("Unknown artifact ID.")
        return record.public()

    def for_job(self, job_id: str) -> list[ArtifactSummary]:
        return [record.public() for record in self._records().values() if record.job_id == job_id]


class JobStore:
    def __init__(self, paths: LocalPaths, artifacts: ArtifactStore) -> None:
        self.paths = paths
        self.artifacts = artifacts

    def directory(self, job_id: str) -> Path:
        return self.paths.jobs / job_id

    def status_path(self, job_id: str) -> Path:
        return self.directory(job_id) / "status.json"

    def save(self, job: StoredJob) -> None:
        job.updated_at = utc_now()
        atomic_write_json(self.status_path(job.id), {"schemaVersion": 1, **asdict(job)})

    def load(self, job_id: str) -> StoredJob:
        payload = read_json(self.status_path(job_id), None)
        if not isinstance(payload, dict):
            raise ValueError("Unknown job ID.")
        payload.pop("schemaVersion", None)
        return StoredJob(**payload)

    def list(self) -> list[StoredJob]:
        if not self.paths.jobs.is_dir():
            return []
        jobs: list[StoredJob] = []
        for status in self.paths.jobs.glob("job_*/status.json"):
            try:
                payload = read_json(status, None)
                if isinstance(payload, dict):
                    payload.pop("schemaVersion", None)
                    jobs.append(StoredJob(**payload))
            except (OSError, TypeError, ValueError):
                continue
        return sorted(jobs, key=lambda item: item.created_at, reverse=True)

    def active_count(self) -> int:
        active = {JobState.QUEUED.value, JobState.RUNNING.value}
        return sum(job.state in active for job in self.list())

    def public(self, job: StoredJob) -> JobSummary:
        return JobSummary(
            id=job.id,
            kind=JobKind(job.kind),
            state=JobState(job.state),
            created_at=job.created_at,
            updated_at=job.updated_at,
            artifacts=self.artifacts.for_job(job.id),
            status_message=job.status_message,
            failure=job.failure,
        )

    def public_by_id(self, job_id: str) -> JobSummary:
        return self.public(self.load(job_id))

    def public_list(self) -> list[JobSummary]:
        return [self.public(job) for job in self.list()]
