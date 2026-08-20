from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAT_PLUGIN = ROOT / "plugins" / "viewforge-3d-toolkit"
LOCAL_PLUGIN = ROOT / "plugins" / "viewforge-3d-local"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_local_plugin_is_side_by_side_with_chat_plugin() -> None:
    chat_manifest = _json(CHAT_PLUGIN / ".codex-plugin" / "plugin.json")
    local_manifest = _json(LOCAL_PLUGIN / ".codex-plugin" / "plugin.json")
    marketplace = _json(ROOT / ".agents" / "plugins" / "marketplace.json")
    marketplace_names = {entry["name"] for entry in marketplace["plugins"]}

    assert chat_manifest["name"] == "viewforge-3d-toolkit"
    assert local_manifest["name"] == "viewforge-3d-local"
    assert {"viewforge-3d-toolkit", "viewforge-3d-local"} <= marketplace_names
    assert local_manifest["interface"]["displayName"] == "ViewForge 3D Local"


def test_local_plugin_uses_stdio_venv_launcher_without_tunnel() -> None:
    mcp = _json(LOCAL_PLUGIN / ".mcp.json")
    launcher = LOCAL_PLUGIN / "bin" / "viewforge-local-workbench"
    launcher_text = launcher.read_text(encoding="utf-8")

    assert set(mcp["mcpServers"]) == {"viewforge-local-workbench"}
    assert mcp["mcpServers"]["viewforge-local-workbench"]["command"] == (
        "./bin/viewforge-local-workbench"
    )
    assert "--transport stdio" in launcher_text
    assert "--edition local" in launcher_text
    assert ".venv/bin/python" in launcher_text
    assert "tunnel" not in launcher_text.lower()
    assert launcher.stat().st_mode & os.X_OK


def test_surface_smoothing_runtime_exists_only_in_local_plugin() -> None:
    relative = Path("runtime/blender/smooth_model_surface.py")

    assert (LOCAL_PLUGIN / relative).is_file()
    assert not (CHAT_PLUGIN / relative).exists()
    router = (LOCAL_PLUGIN / "skills" / "viewforge-3d-router" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "$viewforge-3d-local:topology-preserving-smooth" in router
    assert "$viewforge-3d-toolkit:" not in router
