#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

EXCLUDED_PARTS = {".DS_Store", "__pycache__", ".pytest_cache", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
FIXED_TIMESTAMP = (2020, 1, 1, 0, 0, 0)
FORBIDDEN_PACKAGE_SUFFIXES = {
    ".face3d",
    ".viewforge3d",
    ".fbx",
    ".glb",
    ".gltf",
    ".heic",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp4",
    ".npz",
    ".obj",
    ".pdf",
    ".png",
    ".webp",
    ".zip",
}
FORBIDDEN_TEXT_PATTERNS = {
    "macOS user path": re.compile(rb"/Users/[A-Za-z0-9._-]+"),
    "external volume path": re.compile(rb"/Volumes/[A-Za-z0-9._-]+"),
    "Windows user path": re.compile(rb"[A-Za-z]:\\\\Users\\\\[A-Za-z0-9._-]+"),
    "email address": re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "private key": re.compile(rb"BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY"),
    "provider token": re.compile(
        rb"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        rb"AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|"
        rb"xox[baprs]-[0-9A-Za-z-]{10,}|sk-(?:proj-)?[0-9A-Za-z_-]{20,}"
    ),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic ViewForge 3D plugin ZIP.")
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _publishable_payload(source: Path) -> bytes:
    if source.is_symlink():
        raise ValueError(f"plugin package does not permit symlinks: {source}")
    if source.suffix.lower() in FORBIDDEN_PACKAGE_SUFFIXES:
        raise ValueError(f"plugin package does not permit binary artifact: {source.name}")
    payload = source.read_bytes()
    for label, pattern in FORBIDDEN_TEXT_PATTERNS.items():
        if pattern.search(payload):
            raise ValueError(f"plugin package contains {label}: {source.name}")
    return payload


def _archive_file(archive: zipfile.ZipFile, source: Path, archive_name: str) -> None:
    payload = _publishable_payload(source)
    info = zipfile.ZipInfo(archive_name, FIXED_TIMESTAMP)
    executable = bool(source.stat().st_mode & 0o111)
    info.external_attr = (0o100755 if executable else 0o100644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, payload)


def _admitted_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if EXCLUDED_PARTS.intersection(path.parts) or path.suffix in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return files


def build_package(plugin: Path, repository_root: Path, output: Path) -> dict[str, object]:
    plugin = plugin.resolve()
    repository_root = repository_root.resolve()
    output = output.resolve()
    manifest = plugin / ".codex-plugin" / "plugin.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    plugin_name = json.loads(manifest.read_text(encoding="utf-8"))["name"]
    if plugin.name != plugin_name:
        raise ValueError("plugin directory and manifest name must match")

    plugin_license = plugin / "LICENSE"
    repository_license = repository_root / "LICENSE"
    extras = (
        repository_license,
        repository_root / "README.md",
        repository_root / "README.zh-CN.md",
        repository_root / "docs" / "GUIDE.md",
        repository_root / "docs" / "GUIDE.zh-CN.md",
        repository_root / "docs" / "VIRTUAL_ENVIRONMENT.md",
        repository_root / "docs" / "VIRTUAL_ENVIRONMENT.zh-CN.md",
    )
    if not plugin_license.is_file() or any(not path.is_file() for path in extras):
        raise FileNotFoundError(
            "matching repository and plugin Apache-2.0 licenses plus English and Simplified "
            "Chinese documentation are required"
        )
    if plugin_license.read_bytes() != repository_license.read_bytes():
        raise ValueError("plugin LICENSE must exactly match the repository LICENSE")

    output.parent.mkdir(parents=True, exist_ok=True)
    extra_relatives = {path.relative_to(repository_root).as_posix() for path in extras}
    with zipfile.ZipFile(output, "w") as archive:
        for source in _admitted_files(plugin):
            relative = source.relative_to(plugin).as_posix()
            if relative in extra_relatives:
                continue
            _archive_file(archive, source, f"{plugin_name}/{relative}")
        for source in extras:
            relative = source.relative_to(repository_root).as_posix()
            _archive_file(archive, source, f"{plugin_name}/{relative}")

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum_path = output.with_suffix(f"{output.suffix}.sha256")
    checksum_path.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("plugin package contains duplicate entries")
        archive.testzip()
    return {
        "output": str(output),
        "checksum": str(checksum_path),
        "sha256": digest,
        "entries": len(names),
    }


def main() -> None:
    arguments = parse_arguments()
    result = build_package(arguments.plugin, arguments.repository_root, arguments.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
