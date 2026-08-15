# ViewForge 3D

[English](README.md) | [简体中文](README.zh-CN.md)

ViewForge 3D is a Codex plugin and Python workspace for traceable, quality-gated construction of
3D models from multi-view 2D evidence. It supports people, stylized characters, products, and
general objects through continuous-template fitting, bounded geometry refinement, attached-feature
workflows, fixed-view QA, and appearance changes that preserve accepted geometry.

The toolkit fails closed: a missing production template, insufficient evidence, broken topology,
or rejected visual candidate does not silently fall back to a plausible primitive or voxel shell.

## Repository layout

- `plugins/viewforge-3d-toolkit/` — distributable Codex plugin source.
- `.agents/plugins/marketplace.json` — repo-local Codex marketplace manifest.
- `LICENSE` — Apache License 2.0 terms for the repository code and plugin distribution.
- `src/face3d/` — Python geometry and validation implementation.
- `scripts/` — deterministic builders and audit utilities.
- `tests/` — Python tests.
- `viewer/` — local model and annotation viewer.
- `docs/GUIDE.md` — detailed installation and skill invocation guide.
- `docs/VIRTUAL_ENVIRONMENT.md` — isolated Python environment setup.
- `README.zh-CN.md`, `docs/GUIDE.zh-CN.md`, and `docs/VIRTUAL_ENVIRONMENT.zh-CN.md` — independent
  Simplified Chinese documentation.

## Start here

1. Follow [the virtual environment guide](docs/VIRTUAL_ENVIRONMENT.md).
2. Install the plugin and invoke its skills with [the detailed guide](docs/GUIDE.md).
3. Use the router when the correct geometry workflow is unclear:

```text
$viewforge-3d-toolkit:viewforge-3d-router
```

## Boundaries

- No third-party finished 3D model is imported when the requested route is reconstruction from 2D.
- SDF, voxels, point clouds, and Marching Cubes remain QA-only unless a separate preview route is
  explicitly declared.
- Automated geometry gates and user visual acceptance are separate states.
- Private input images and restricted engineering drawings are not included in the plugin package.

The distributable plugin package is generated locally under `dist/`; generated models, evidence,
virtual environments, caches, and other run artifacts are intentionally excluded from Git.

The annotation bridge is source-agnostic. Configure its locked input explicitly before use:

```bash
export VIEWFORGE3D_ANNOTATION_SOURCE_MODEL=/absolute/path/to/source.glb
export VIEWFORGE3D_ANNOTATION_SOURCE_SHA256=<64-character-sha256>
export VIEWFORGE3D_ANNOTATION_SOURCE_VERSION=source-v1
export VIEWFORGE3D_ANNOTATION_SUBJECT_PROFILE=generic-object
npm --prefix viewer run annotate
```

Project-specific experiment scripts and generated annotation metadata are intentionally excluded
from the published repository.

## License

Repository code, documentation, and the distributable plugin are licensed under the
[Apache License 2.0](LICENSE). Third-party reference images and trademarks retain their respective
owners' rights and are not relicensed by this project.

The plugin directory carries an identical `LICENSE` copy so standalone Codex installations retain
the license terms.
