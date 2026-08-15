from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

PACKAGE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "package_face3d_plugin.py"


def _package_fixture(root: Path, skill_text: str = "# Generic skill\n") -> Path:
    plugin = root / "plugins" / "fixture-plugin"
    manifest = plugin / ".codex-plugin" / "plugin.json"
    skill = plugin / "skills" / "generic" / "SKILL.md"
    manifest.parent.mkdir(parents=True)
    skill.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"name": "fixture-plugin", "version": "1.0.0", "skills": "./skills/"}),
        encoding="utf-8",
    )
    skill.write_text(skill_text, encoding="utf-8")
    (plugin / "requirements.txt").write_text("numpy>=1.26,<2\n", encoding="utf-8")
    (root / "README.md").write_text("# Fixture\n", encoding="utf-8")
    (root / "README.zh-CN.md").write_text("# 测试插件\n", encoding="utf-8")
    docs = root / "docs"
    docs.mkdir()
    (docs / "GUIDE.md").write_text("# Guide\n", encoding="utf-8")
    (docs / "GUIDE.zh-CN.md").write_text("# 指南\n", encoding="utf-8")
    (docs / "VIRTUAL_ENVIRONMENT.md").write_text("# Environment\n", encoding="utf-8")
    (docs / "VIRTUAL_ENVIRONMENT.zh-CN.md").write_text("# 虚拟环境\n", encoding="utf-8")
    return plugin


def _run_packager(plugin: Path, root: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PACKAGE_SCRIPT),
            "--plugin",
            str(plugin),
            "--repository-root",
            str(root),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_build_package_accepts_generic_text_only_plugin(tmp_path: Path) -> None:
    plugin = _package_fixture(tmp_path)
    output = tmp_path / "dist" / "fixture-plugin.zip"

    completed = _run_packager(plugin, tmp_path, output)

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["entries"] == 9
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert {
            "fixture-plugin/README.md",
            "fixture-plugin/README.zh-CN.md",
            "fixture-plugin/docs/GUIDE.md",
            "fixture-plugin/docs/GUIDE.zh-CN.md",
            "fixture-plugin/docs/VIRTUAL_ENVIRONMENT.md",
            "fixture-plugin/docs/VIRTUAL_ENVIRONMENT.zh-CN.md",
        } <= set(archive.namelist())
        assert all(
            not name.startswith("/") and ".." not in Path(name).parts
            for name in archive.namelist()
        )


def test_build_package_rejects_workstation_path(tmp_path: Path) -> None:
    plugin = _package_fixture(tmp_path, "source = /Users/example/private/model.glb\n")

    completed = _run_packager(plugin, tmp_path, tmp_path / "dist" / "fixture-plugin.zip")

    assert completed.returncode != 0
    assert "macOS user path" in completed.stderr


def test_build_package_rejects_binary_artifact(tmp_path: Path) -> None:
    plugin = _package_fixture(tmp_path)
    binary = plugin / "assets" / "model.glb"
    binary.parent.mkdir()
    binary.write_bytes(b"glTF")

    completed = _run_packager(plugin, tmp_path, tmp_path / "dist" / "fixture-plugin.zip")

    assert completed.returncode != 0
    assert "binary artifact" in completed.stderr
