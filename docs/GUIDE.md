# Detailed guide

[English](GUIDE.md) | [简体中文](GUIDE.zh-CN.md)

## Contents

1. Install the repository plugin
2. Invoke the router and individual skills
3. Choose a workflow
4. Run the unskinned object route
5. Validate and package the plugin
6. Interpret output states

## 1. Install the repository plugin

Create the Python environment first if you will run bundled geometry scripts. See
[`VIRTUAL_ENVIRONMENT.md`](VIRTUAL_ENVIRONMENT.md).

From the cloned repository root, register the repo-local marketplace and install the plugin:

```bash
codex plugin marketplace add "$(pwd)"
codex plugin add viewforge-3d-toolkit@viewforge-3d
```

Start a new Codex task after installation so the skill inventory is refreshed. To update a cloned
copy, pull the repository, reinstall the same plugin entry, and start another new task.

## 2. Invoke the router and individual skills

Use the router for an end-to-end request or when the correct branch is unclear:

```text
Use $viewforge-3d-toolkit:viewforge-3d-router to choose the safe route for this model.
```

Call a specialist directly when the stage is already known:

```text
Use $viewforge-3d-toolkit:reconstruct-3d-from-multiview to build an unskinned object preview from these images.

Use $viewforge-3d-toolkit:landmark-guided-refinement to fit these accepted facial landmarks without changing hidden depth.

Use $viewforge-3d-toolkit:annotation-region-lowering to press this annotated region inward without smoothing.

Use $viewforge-3d-toolkit:topology-preserving-smooth to fair only this approved region and preserve topology and UVs.

Use $viewforge-3d-toolkit:blender-manual-polish to guide a bounded manual Blender correction.

Use $viewforge-3d-toolkit:same-geometry-skin to change appearance without moving accepted geometry.
```

In the Codex app, the plugin can also be referenced with its plugin mention and then described in
plain language. Fully qualified skill names remain the least ambiguous form.

## 3. Choose a workflow

### New reconstruction or provenance audit

Start with `reconstruct-3d-from-multiview`. Inventory evidence, declare a subject profile, select a
continuous template or preview route, and lock the immutable source before fitting.

### Accepted geometry requiring a local edit

Use landmark refinement, annotation lowering, bounded smoothing, or manual polish according to the
requested operation. Do not reclassify the job as a new reconstruction.

### Appearance-only change

Use `same-geometry-skin` only after geometry acceptance. Require identical geometry and UV hashes
before and after the appearance stage.

### Unskinned object

Use reconstruction with the object-template reference. Stop after geometry QA, retain neutral
color factors only, and verify the exported GLB contains no image textures.

## 4. Run the unskinned object route

The iPhone 17 example requires two admitted Apple reference images whose shortest side is at least
1024 pixels. Supply the source images explicitly; do not download or import an existing 3D model.

```bash
source .venv/bin/activate

python scripts/build_iphone17_unskinned.py \
  --hero /absolute/path/to/official-hero.jpg \
  --side /absolute/path/to/official-side.jpg \
  --output dist/iphone17-unskinned-v1
```

The destination is immutable. Use a new versioned output directory when evidence, code, template,
dimensions, thresholds, or materials change.

Required output evidence includes:

- `models/iphone17-unskinned.glb`;
- `manifest.json` with `external3dImported=false` and `skinApplied=false`;
- `evidence/ledger.json` with source URLs, hashes, resolution gates, and authority classes;
- `qa/geometry-quality.json` with topology, self-intersection, GLB roundtrip, and texture checks;
- fixed front, oblique, side, back, top, and bottom renders;
- `checksums.json` covering the immutable run.

Treat rounded shell parameters and feature outlines that are not published as bounded 2D fits, not
authoritative CAD measurements. Keep `userSignoff=false` until the fixed views are reviewed.

## 5. Validate and package the plugin

Validate every skill and then the plugin manifest:

```bash
python /path/to/skill-creator/scripts/quick_validate.py \
  plugins/viewforge-3d-toolkit/skills/reconstruct-3d-from-multiview

python /path/to/plugin-creator/scripts/validate_plugin.py \
  plugins/viewforge-3d-toolkit
```

Build the deterministic ZIP:

```bash
source .venv/bin/activate
python scripts/package_viewforge_plugin.py \
  --plugin plugins/viewforge-3d-toolkit \
  --repository-root . \
  --output dist/viewforge-3d-toolkit-0.4.0.zip
```

The ZIP contains the Apache-2.0 license, plugin manifest, all skills, and separate English and
Simplified Chinese versions of the README, detailed guide, and virtual-environment guide. It
excludes caches, generated models, source evidence, comparison images, and virtual environments.

## 6. Interpret output states

- `preview` — inferred or exploratory geometry; not suitable for production identity claims.
- `geometry-preview` — geometry and automated checks exist, but visual acceptance is pending.
- `automated-gates-passed` — declared automated gates passed; user signoff is still separate.
- `user-accepted` — the user explicitly accepted the exact immutable candidate.
- `rejected` — never use as the next baseline.

Do not turn one state into another based on a screenshot, successful command exit, or package
existence alone.
