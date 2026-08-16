# ViewForge Local

Private macOS host for ViewForge multiview reconstruction, geometry generation, Blender modeling,
biological skeleton, and animation MCP tools.

## Shape

- ChatGPT archetype: `tool-only`
- Local UI: SwiftUI process and workspace control
- MCP transport: streamable HTTP at `http://127.0.0.1:8765/mcp`
- Worker: bundled Python 3.11 runtime plus the complete lockfile-pinned ViewForge geometry stack
- Geometry engines: bundled NumPy/OpenCV/MediaPipe/Trimesh/Open3D pipeline and installed Blender
- Blender policy: background jobs disable auto-execution and accept declarative model JSON rather
  than arbitrary Python
- Storage: `~/Library/Application Support/ViewForge Local`

The server accepts only files inside the workspace selected in the app. Tool results use opaque
asset, job, and artifact IDs; absolute local paths stay in the local state store and logs.

## Install and connect

- [ViewForge Local 安装、Tunnel 与 API Key 指南](USER_SETUP.zh-CN.md)
- [OpenAI Secure MCP Tunnel documentation](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)
- [OpenAI Platform Tunnel settings](https://platform.openai.com/settings/organization/tunnels)
- [OpenAI Platform Runtime API keys](https://platform.openai.com/settings/organization/api-keys)
- [`tunnel-client` latest release](https://github.com/openai/tunnel-client/releases/latest)

## Build

```bash
./local-app/scripts/build_app.sh
```

The local app is written to `dist/viewforge-local/ViewForge Local.app`. The build uses a copied
uv-managed CPython distribution and installs the production dependency graph from `uv.lock`, so the
finished app does not depend on the repository `.venv`. Licensed model assets are not embedded;
production profiles continue to fail closed until the user provides and records them locally.

## Local checks

```bash
swift build --package-path local-app
.venv/bin/pytest -q \
  tests/test_local_mcp.py \
  tests/test_plugin_packaging.py \
  tests/test_biological_skeleton.py \
  tests/test_biological_animation.py
```

Use MCP Inspector with `http://127.0.0.1:8765/mcp`. For ChatGPT Developer Mode, create a Secure
MCP Tunnel profile that forwards to that URL, then start the profile from the app.
