# ViewForge Local

Private macOS host for ViewForge multiview reconstruction, geometry generation, Blender modeling
and rendering, biological skeleton, and animation MCP tools.
The UI follows the system language by default and also offers explicit Chinese and English modes.

## Shape

- ChatGPT archetype: `tool-only`
- Local UI: SwiftUI process and workspace control
- MCP transport: streamable HTTP at `http://127.0.0.1:8765/mcp`
- Worker: bundled Python 3.11 runtime plus the complete lockfile-pinned ViewForge geometry stack
- Geometry engines: bundled NumPy/OpenCV/MediaPipe/Trimesh/Open3D pipeline and installed Blender
- Blender policy: background jobs disable auto-execution and accept declarative model JSON or
  fixed render options rather than arbitrary Python
- Storage: `~/Library/Application Support/ViewForge Local`

The server accepts only files inside the workspace selected in the app. Tool results normally use
opaque asset, job, and artifact IDs; `read_image_artifact` can return one explicitly selected
rendered image to the conversation. Generated `artifact_` IDs can be passed directly into later
skeleton, animation, binding, and rendering jobs; they do not need to be copied into the local
asset registry. JSON control documents may also be supplied inline. Failed geometry jobs publish a
sanitized `error.json` artifact for `read_json_artifact`. Absolute local paths stay in the local
state store and logs.

## Install and connect

- [ViewForge Local 安装、Tunnel 与 API Key 指南](USER_SETUP.zh-CN.md)
- [OpenAI Secure MCP Tunnel documentation](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)
- [OpenAI Platform Tunnel settings](https://platform.openai.com/settings/organization/tunnels)
- [OpenAI Platform Runtime API keys](https://platform.openai.com/settings/organization/api-keys)
- [`tunnel-client` latest release](https://github.com/openai/tunnel-client/releases/latest)

## Build

```bash
./local-app/scripts/build_app.sh
./local-app/scripts/install_app.sh
```

The build artifact is written to `dist/viewforge-local/ViewForge Local.app`; the installer copies it
to `~/Applications/ViewForge Local.app`, where the cached Codex plugin launcher can discover it.
Existing installs are preserved as timestamped backups. The build uses a copied uv-managed CPython
distribution and installs the production dependency graph from `uv.lock`, so the finished app does
not depend on the repository `.venv`. Licensed model assets are not embedded; production profiles
continue to fail closed until the user provides and records them locally.

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
