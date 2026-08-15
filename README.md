# Face3D Modeling Toolkit

Face3D Modeling Toolkit is a Codex plugin and Python workspace for traceable, quality-gated
multi-view 2D-to-3D reconstruction. It supports continuous-template fitting for faces and objects,
bounded geometry refinement, attached-feature workflows, fixed-view QA, and appearance changes
that preserve accepted geometry.

The toolkit fails closed: a missing production template, insufficient evidence, broken topology,
or rejected visual candidate does not silently fall back to a plausible primitive or voxel shell.

## Repository layout

- `plugins/face3d-modeling-toolkit/` — distributable Codex plugin source.
- `.agents/plugins/marketplace.json` — repo-local Codex marketplace manifest.
- `src/face3d/` — Python geometry and validation implementation.
- `scripts/` — deterministic builders and audit utilities.
- `tests/` — Python tests.
- `viewer/` — local model and annotation viewer.
- `docs/GUIDE.md` — detailed installation and skill invocation guide.
- `docs/VIRTUAL_ENVIRONMENT.md` — isolated Python environment setup.

## Start here

1. Follow [the virtual environment guide](docs/VIRTUAL_ENVIRONMENT.md).
2. Install the plugin and invoke its skills with [the detailed guide](docs/GUIDE.md).
3. Use the router when the correct geometry workflow is unclear:

```text
$face3d-modeling-toolkit:face3d-toolkit-router
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
export FACE3D_ANNOTATION_SOURCE_MODEL=/absolute/path/to/source.glb
export FACE3D_ANNOTATION_SOURCE_SHA256=<64-character-sha256>
export FACE3D_ANNOTATION_SOURCE_VERSION=source-v1
export FACE3D_ANNOTATION_SUBJECT_PROFILE=generic-object
npm --prefix viewer run annotate
```

Project-specific experiment scripts and generated annotation metadata are intentionally excluded
from the published repository.
