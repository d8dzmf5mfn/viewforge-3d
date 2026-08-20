# ViewForge 3D Local

This is the Codex-only local edition of ViewForge 3D. It is installed beside
`viewforge-3d-toolkit`; it does not replace the ChatGPT/Tunnel edition.

The launcher uses the repository's locked Python 3.11 `.venv`, stores jobs under a separate
`ViewForge 3D Local Plugin` state directory, and exposes two local-only MCP tools:

- `smooth_model_surface` creates a new topology-preserving smoothed Blend/GLB plus QA.
- `get_local_artifact_path` returns an exact output path for local desktop tools.

Configure the runtime from the repository checkout before installing the plugin:

```bash
plugins/viewforge-3d-local/scripts/setup-local-runtime.sh
codex plugin add viewforge-3d-local@viewforge-3d
```

Start a new Codex task after installation so the local tool schema is loaded.
