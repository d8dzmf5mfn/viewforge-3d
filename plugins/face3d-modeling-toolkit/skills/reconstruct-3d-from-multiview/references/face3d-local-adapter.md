# Face3D workspace adapter

Use this adapter only when the project contains `configs/face-v3.yaml`,
`src/face3d/stages/template_v3.py`, and the TemplateHeadV0 anatomy assets.

## Existing capability map

| Generic stage | Local implementation |
| --- | --- |
| Profile | `src/face3d/profiles/face_v3.py` |
| Intake | `src/face3d/stages/intake.py` |
| Continuous asset | `src/face3d/unified_head.py` |
| Non-rigid fit | `src/face3d/stages/template_fit.py` |
| Same-geometry skin | `src/face3d/stages/template_skin.py`, `src/face3d/skin_v2.py` |
| QA-only distance | `src/face3d/stages/template_qa.py` |
| Package | `src/face3d/package.py`, manifest v3 |
| Viewer | `viewer/src/three/DualViewport.tsx` |

For production, do not route a new experiment through v1/v2 pixel, cube, voxel, Marching Cubes,
procedural cranium, stitched-ear, or profile-loft code. Those paths are historical compatibility
or preview-only. The profile-loft exception below applies only when the user explicitly requests
an existing loft refinement or accepts a provenance-labelled stylized preview.

## Profile-loft preview adapter

Use a host project's profile-loft builder only under route `profile-loft-preview`. Record the
builder path and code hash in the run ledger. Treat legacy `left45`/`right45` enum names as aliases;
record the actual camera yaw from each `CameraRecord`. Generated profile images remain
`model-inferred` and cannot satisfy production multiview admission.

For a ring-like bulge in an existing regular loft, prefer the reusable skill script:

```bash
python3 <skill-root>/scripts/polish_profile_loft.py \
  --input <source.glb> \
  --output <new-run>/loft-polished.glb \
  --lower-y <lower> \
  --upper-y <upper> \
  --support-rows 8 \
  --strength 1.0 \
  --metrics <new-run>/qa/polish-metrics.json
```

A host project may wrap this operation with its own fixed-view renderer. Preserve the source GLB,
create a new output directory, record the wrapper's code hash, and keep the result `preview` after
automated gates pass. Read `references/profile-loft-preview.md` for the construction and validation
contract.

## Attached-feature preview adapter

For independently reviewed stylized nose, ear, antenna, or other attached parts, read
`references/attached-feature-integration-preview.md`. Reuse accepted component hashes and pose
records; do not rebuild a user-approved feature merely to create its counterpart. Mirror the posed
component and its short bridge when symmetry is intended, correct reflection winding, then union in
`base + bridge + feature` order with the skill's generic scripts.

Keep these experiments outside the production TemplateHeadV0 claim. They are explicit
`profile-loft-preview` geometry even when the exact Boolean and topology gates pass. The project
integration scripts may provide rendering and collision helpers, but the delivered artifact must
still pass the skill's independent NPZ and persisted-GLB surface validator.

## Preflight

Run from the repository root without downloading:

```bash
UV_CACHE_DIR=.uv-cache uv sync --all-groups --offline
.venv/bin/face3d assets status --config configs/face-v3.yaml
.venv/bin/ruff check src tests scripts
```

Require `ready=true`. If a licensed asset is absent or its hash differs, stop. Do not silently use
the old FLAME, cube, or procedural fallback. Estimate downloads first and request approval before
any single or cumulative download above 1 GB.

## Reconstruction

Name the source images `front`, `left45`, and `right45` using supported image extensions, then run:

```bash
.venv/bin/face3d validate-input \
  --input <input-dir> \
  --config configs/face-v3.yaml

.venv/bin/face3d reconstruct \
  --input <input-dir> \
  --config configs/face-v3.yaml \
  --output <run-dir>
```

The first reconstruction is expected to stop at `mask-review-required` when masks are not yet
confirmed. Review the generated masks, then run:

```bash
.venv/bin/face3d confirm-masks --run <run-dir>
.venv/bin/face3d reconstruct \
  --input <input-dir> \
  --config configs/face-v3.yaml \
  --output <run-dir>

.venv/bin/face3d package \
  --run <run-dir> \
  --config configs/face-v3.yaml \
  --output <run-dir>.face3d
```

Do not confirm masks without visual inspection. Do not retry a failed fit by loosening all gates.
Fix the view, mask, profile mapping, or fitting cause.

## Verification

Run:

```bash
.venv/bin/pytest -q tests/test_template_head_anatomy.py
.venv/bin/pytest -q tests/test_template_fit.py
.venv/bin/pytest -q tests/test_package.py
npm --prefix viewer run build
npm --prefix viewer run test:e2e
```

Check that `models/head.glb` contains `HeadSkin`, `Eyeball.L`, and `Eyeball.R`; fitted, skin, QA,
manifest, and GLB geometry hashes match; canonical UV does not move; and `.face3d` contains no
voxel/pixel final surface. Inspect neutral and skin renders at front, left/right 45 degrees, and
side before user signoff.

## Known runtime boundary

Blender background Boolean behavior depends on the exact build and mesh. Probe the current version,
preserve every attempt, and re-run all topology and self-intersection gates after any Boolean.
Never use a crash as permission to create another surface.

The licensed Lee Perry-Smith preview texture is a material-path diagnostic, not a reconstructed new
person. Do not use it as identity acceptance evidence.
