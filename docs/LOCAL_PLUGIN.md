# ViewForge 3D Local plugin

[English](LOCAL_PLUGIN.md) | [简体中文](LOCAL_PLUGIN.zh-CN.md)

`viewforge-3d-local` is a separate Codex-local plugin. It can be installed beside
`viewforge-3d-toolkit`; neither plugin overwrites the other.

| Boundary | ChatGPT/Tunnel edition | Codex local edition |
| --- | --- | --- |
| Plugin ID | `viewforge-3d-toolkit` | `viewforge-3d-local` |
| Transport | App/Tunnel capable | Local stdio only |
| Python | Bundled app runtime with compatibility fallback | Repository `.venv/bin/python` only |
| State | `ViewForge Local` application state | `ViewForge 3D Local Plugin` state |
| Extra tools | None | `smooth_model_surface`, `get_local_artifact_path` |

The local edition does not configure or use a Tunnel, remote URL, control-plane key, or OpenAI API
key. Its setup script validates the existing Python 3.11 `.venv` and records the repository root in
a private locator file:

```text
~/Library/Application Support/ViewForge 3D Local Plugin/repository-root
```

The launcher reads that locator automatically. A person does not need to remember or paste the
repository path during normal use. Local jobs and asset indexes are isolated under:

```text
~/Library/Application Support/ViewForge 3D Local Plugin/state
```

## Installation and refresh

From the source checkout, the maintainer or setup agent runs:

```bash
plugins/viewforge-3d-local/scripts/setup-local-runtime.sh
codex plugin add viewforge-3d-local@viewforge-3d
```

Start a new Codex task after installation or a plugin cachebuster update.

## Surface smoothing

`smooth_model_surface` accepts a workspace `.blend`/`.glb` path or an existing asset/artifact ID.
It creates a new source-independent job and emits:

- `smoothed-model.blend`
- `smoothed-model.glb`
- `smoothing-qa.json`

The job preserves mesh topology, UVs, materials, transforms, protected vertices, and a configured
displacement budget. Volume and topological-boundary preservation default on. Because GLB import
may represent UV or normal seams as topological boundaries, prefer the source Blend. Disable
boundary preservation only after explicitly reviewing those boundaries. The output remains
`pendingUserSignoff` until fixed-view renders are reviewed.
