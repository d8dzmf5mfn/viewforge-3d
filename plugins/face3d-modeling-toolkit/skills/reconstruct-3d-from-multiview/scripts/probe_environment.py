#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from _common import atomic_write_json, sha256_file, utc_now

MODULES = (
    "numpy",
    "scipy",
    "cv2",
    "mediapipe",
    "torch",
    "open3d",
    "skimage",
    "trimesh",
    "PIL",
)


def command_version(executable: Path | str | None, *arguments: str) -> str | None:
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [str(executable), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (completed.stdout or completed.stderr).strip()
    return text.splitlines()[0] if text else None


def available_modules(python: Path) -> dict[str, bool]:
    program = (
        "import importlib.util,json;"
        f"names={list(MODULES)!r};"
        "print(json.dumps({name: importlib.util.find_spec(name) is not None for name in names}))"
    )
    try:
        completed = subprocess.run(
            [str(python), "-c", program],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        result = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {name: False for name in MODULES}
    return {name: bool(result.get(name, False)) for name in MODULES}


def blender_record() -> dict[str, Any]:
    application = Path("/Applications/Blender.app")
    executable = application / "Contents/MacOS/Blender"
    record: dict[str, Any] = {
        "path": str(executable) if executable.is_file() else shutil.which("blender"),
        "version": None,
        "guiPreferredForComplexBoolean": platform.system() == "Darwin",
    }
    info = application / "Contents/Info.plist"
    if info.is_file():
        try:
            with info.open("rb") as handle:
                record["version"] = plistlib.load(handle).get("CFBundleShortVersionString")
        except (OSError, plistlib.InvalidFileException):
            pass
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe a local multiview reconstruction environment without installing anything."
        )
    )
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-face3d", action="store_true")
    parser.add_argument("--maximum-hash-file-mb", type=float, default=128.0)
    arguments = parser.parse_args()

    project = arguments.project.expanduser().resolve()
    virtual_python = project / ".venv/bin/python"
    python = virtual_python if virtual_python.is_file() else Path(sys.executable)
    modules = available_modules(python)
    executable_paths = {
        "uv": shutil.which("uv"),
        "node": shutil.which("node"),
        "npm": shutil.which("npm"),
    }
    markers = {
        "pyproject": project / "pyproject.toml",
        "uvLock": project / "uv.lock",
        "face3dSource": project / "src/face3d",
        "face3dCli": project / ".venv/bin/face3d",
        "faceV3Config": project / "configs/face-v3.yaml",
        "templateHeadV0": (
            project / "assets/template-head-v0/anatomy/template-head-v0.unified.npz"
        ),
        "viewerPackage": project / "viewer/package.json",
    }
    maximum_hash_bytes = int(arguments.maximum_hash_file_mb * 1024 * 1024)
    files: dict[str, Any] = {}
    for name, path in markers.items():
        record: dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
            "isFile": path.is_file(),
        }
        if path.is_file():
            record["bytes"] = path.stat().st_size
            if path.stat().st_size <= maximum_hash_bytes:
                record["sha256"] = sha256_file(path)
            else:
                record["sha256"] = None
                record["hashSkipped"] = "file-exceeds-probe-limit"
        files[name] = record

    generic_modules = all(modules[name] for name in ("numpy", "scipy", "trimesh", "PIL"))
    face_modules = all(
        modules[name]
        for name in ("numpy", "scipy", "cv2", "mediapipe", "torch", "open3d", "trimesh", "PIL")
    )
    face_markers = all(
        markers[name].exists()
        for name in ("face3dSource", "face3dCli", "faceV3Config", "templateHeadV0")
    )
    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "probedAt": utc_now(),
        "project": str(project),
        "platform": {
            "system": platform.system(),
            "release": platform.mac_ver()[0] or platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "path": str(python),
            "version": command_version(python, "--version"),
            "virtualEnvironmentUsed": virtual_python.is_file(),
            "modules": modules,
        },
        "tools": {
            "uv": {
                "path": executable_paths["uv"],
                "version": command_version(executable_paths["uv"], "--version"),
            },
            "node": {
                "path": executable_paths["node"],
                "version": command_version(executable_paths["node"], "--version"),
            },
            "npm": {
                "path": executable_paths["npm"],
                "version": command_version(executable_paths["npm"], "--version"),
            },
            "blender": blender_record(),
        },
        "files": files,
        "capabilities": {
            "genericTemplateExperiment": generic_modules,
            "localThreeJsViewer": bool(
                executable_paths["node"]
                and executable_paths["npm"]
                and markers["viewerPackage"].is_file()
            ),
            "currentFace3dAdapter": face_modules and face_markers,
        },
        "resourcePolicy": {
            "networkAccessed": False,
            "downloadsPerformed": False,
            "downloadApprovalBytes": 1024**3,
        },
    }
    payload["ready"] = bool(payload["capabilities"]["genericTemplateExperiment"])
    if arguments.output is not None:
        atomic_write_json(arguments.output.expanduser().resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if arguments.require_face3d and not payload["capabilities"]["currentFace3dAdapter"]:
        return 2
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
