# ViewForge 3D

[English](README.md) | [简体中文](README.zh-CN.md)

> **Turn multi-view 2D evidence into traceable, quality-gated 3D models.**

ViewForge 3D is a Codex plugin and Python workspace for reconstructing, refining, rigging, and
validating 3D models from supplied visual evidence. It does not just return plausible geometry:
the workflow records its source, checks the result, and fails closed when the geometry cannot be
justified.

[Latest release](https://github.com/d8dzmf5mfn/viewforge-3d/releases/latest) ·
[Install and use](docs/GUIDE.md) · [Python environment](docs/VIRTUAL_ENVIRONMENT.md) ·
[ViewForge Local, Tunnel, and API key](local-app/USER_SETUP.zh-CN.md)

![iPhone 17 official 2D evidence compared with a ViewForge geometry preview](docs/assets/iphone17-evidence-vs-geometry-preview.png)

*Demo — official iPhone 17 2D reference evidence (left) compared with the project-authored
`TemplatePhoneV0` geometry preview (right). This illustrates evidence-to-preview traceability; it
is not a user-accepted reconstruction or an authoritative CAD model. Apple imagery and trademarks
remain the property of their respective owners.*

## Why ViewForge?

- **Evidence-driven** — reconstruct against supplied multi-view references instead of importing a
  finished third-party 3D model.
- **Quality-gated** — run automated geometry checks and fixed-view QA before asking for visual
  acceptance.
- **Fail-closed** — missing templates, insufficient evidence, broken topology, or rejected
  candidates do not silently become a plausible primitive or voxel shell.
- **Traceable** — keep source identity, workflow state, checks, and acceptance separate and
  auditable.

## What it can do

| Capability | Outcome |
| --- | --- |
| Multi-view reconstruction | Continuous-template fitting for supported people, stylized characters, and products; six-view visual-hull reconstruction for aligned object silhouettes. |
| Declarative Blender modeling | Reproducible, component-based Blender scenes and exports from structured model declarations. |
| Model rendering | Fixed-view PNG previews and a contact sheet from an existing Blend or GLB, returned directly to the conversation for review. |
| Bounded geometry refinement | Landmark fitting, annotation-guided lowering, topology-preserving smoothing, attached features, and guided manual polish. |
| Skeletons and animation | Auditable biological Armatures, rotation-only bone animation, and rigid Bone Parent motion for segmented models without skin weights. |
| Appearance-only changes | Update materials or skin while preserving accepted geometry and UVs. |
| QA and provenance | Geometry gates, fixed-view renders, reopen audits, checksums, and explicit preview/acceptance states. |

## Quick start

Clone the repository, then prepare the isolated Python environment if you will run the bundled
geometry scripts:

```bash
git clone https://github.com/d8dzmf5mfn/viewforge-3d.git
cd viewforge-3d
```

See [the virtual environment guide](docs/VIRTUAL_ENVIRONMENT.md) for the supported Python setup.
From the repository root, install ViewForge Local and the Codex plugin:

```bash
./local-app/scripts/build_app.sh release
./local-app/scripts/install_app.sh
codex plugin marketplace add "$(pwd)"
codex plugin add viewforge-3d-toolkit@viewforge-3d
```

Start a new Codex task so the plugin inventory refreshes, then let the router choose the safe
workflow:

```text
Use $viewforge-3d-toolkit:viewforge-3d-router to choose the safe route for this model.
```

For all specialist skills, packaging instructions, and output states, follow
[the detailed guide](docs/GUIDE.md). To expose the local geometry runtime to ChatGPT Developer
Mode or another device, follow the
[ViewForge Local installation, Tunnel, and API key guide](local-app/USER_SETUP.zh-CN.md).

## How it works

1. **Lock the evidence** — inventory the supplied views, hashes, subject profile, and authority of
   each source.
2. **Choose a declared route** — select reconstruction, declarative modeling, bounded refinement,
   appearance, skeleton, or animation without silently changing methods.
3. **Build within the evidence** — fit only supported geometry and keep hidden or unsupported depth
   uncertain.
4. **Check the artifact** — run topology, round-trip, texture, binding, and fixed-view checks that
   apply to the selected route.
5. **Separate acceptance states** — automated gates can pass while user visual acceptance remains
   pending.

## Repository layout

- `plugins/viewforge-3d-toolkit/` — distributable Codex plugin source.
- `.agents/plugins/marketplace.json` — repo-local Codex marketplace manifest.
- `LICENSE` — Apache License 2.0 terms for the repository code and plugin distribution.
- `src/face3d/` — Python geometry and validation implementation.
- `scripts/` — deterministic builders and audit utilities.
- `tests/` — Python tests.
- `viewer/` — local model and annotation viewer.
- `local-app/` — private macOS MCP host and its build/connect guide.
- `docs/GUIDE.md` — detailed installation and skill invocation guide.
- `docs/VIRTUAL_ENVIRONMENT.md` — isolated Python environment setup.
- `README.zh-CN.md`, `docs/GUIDE.zh-CN.md`, and `docs/VIRTUAL_ENVIRONMENT.zh-CN.md` — independent
  Simplified Chinese documentation.

## Boundaries

- No third-party finished 3D model is imported when the requested route is reconstruction from 2D.
- SDF, voxels, point clouds, and Marching Cubes remain QA-only unless a separate preview route is
  explicitly declared.
- Automated geometry gates and user visual acceptance are separate states.
- Static bone-only skeleton runs do not add weights, Armature modifiers, mesh parenting, or
  animation.
- Bone-only animation keys rotations only. No-skin model motion uses rigid Bone Parent only for
  segmented components; continuous meshes require skin weights to bend and therefore fail closed.
- Private input images and restricted engineering drawings are not included in the plugin package.

The distributable plugin package is generated locally under `dist/`. Generated models, working
evidence, virtual environments, caches, and other run artifacts are intentionally excluded from
Git. Curated documentation comparisons such as the demo above remain outside the generated plugin
ZIP and do not imply that third-party imagery is relicensed.

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
