---
name: reconstruct-3d-from-multiview
description: "Build, audit, and refine quality-gated multi-view 2D-to-3D reconstructions using licensed continuous-template deformation for production or a provenance-labelled profile-loft preview for stylized/anime work. Use when Codex needs to reconstruct a face, character, body part, or object; classify observed versus inferred views; trace an existing GLB; polish loft bulges; build and mirror standalone stylized features such as anime noses or ears; connect an appendage through a short organic bridge; perform exact triangle-mesh Boolean integration; project appearance; or distinguish genuine reconstruction from a template, demo, voxel, or preview surface."
---

# Reconstruct 3D from multiview images

Choose and record the route before writing code. Never let a failed production route silently fall
back to a plausible shell, primitive, voxel surface, or single-view relief.

## Select the route

Inventory every input as `observed`, `user-annotated`, `model-inferred`, `template-prior`, or
`derived`. Record content hashes, intended view roles, camera assumptions, and authority limits.

| Route | Use for | Surface lineage | Maximum claim |
| --- | --- | --- | --- |
| `continuous-template-deformation` | Production or identity-sensitive work with a licensed semantic template and enough real views | template -> fitted -> skinned -> QA -> delivered | `user-accepted` after all gates and visual signoff |
| `profile-loft-preview` | Stylized/anime shape exploration, refinement of an existing regular loft, or work without a suitable template where the user accepts inference | silhouettes -> scanline profiles -> ring mesh -> optional topology-preserving polish | `preview`; never production identity or hidden-surface acceptance |

Use the profile-loft route only when the user explicitly wants a preview/refinement or when its
limitations are acceptable. Set `previewOnly=true` and `identityAcceptanceAllowed=false`. Mark
AI-generated side views as inferred evidence. If the user requests production but no suitable
template or sufficient observed views exist, stop with `template-required` or `request-input`.

Before modifying an existing GLB, establish provenance from the artifact hash, manifests, source
code, run logs, and mesh structure. Do not infer the construction method from colour or a screenshot.
For a non-face object, read
[`references/object-template-route.md`](references/object-template-route.md) before choosing the
template, evidence contract, geometry stages, or packaging claim.
If the accepted source already exists and the task is strict contour/landmark refinement, invoke
`$viewforge-3d-toolkit:landmark-guided-refinement` instead of treating the edit as a new
reconstruction.
Read [`references/profile-loft-preview.md`](references/profile-loft-preview.md) when building,
auditing, or polishing a loft. For the ViewForge 3D repository, also read
[`references/viewforge-local-adapter.md`](references/viewforge-local-adapter.md).
Read
[`references/attached-feature-integration-preview.md`](references/attached-feature-integration-preview.md)
before building, mirroring, positioning, or joining a nose, ear, horn, handle, limb, or other
attached stylized feature.

## Apply common preflight

1. Preserve all supplied inputs and existing outputs. Create a new versioned run directory.
2. Probe the environment without installing:

```bash
python3 <skill-root>/scripts/probe_environment.py \
  --project <project-root> \
  --output <experiment-parent>/environment.json
```

3. Reuse locked project dependencies, local models, renderers, and viewers when available.
4. Estimate downloads before starting. Obtain explicit approval before any single or cumulative
   download above 1 GB.
5. Keep private source images and derived assets local unless the user authorizes transfer.
6. Fix review cameras before comparing geometry. Include front, intermediate-oblique, and side
   views; add rear views when posterior shape matters.

## Run continuous-template production

Read [`references/subject-profile-contract.md`](references/subject-profile-contract.md) before
creating a new subject class and [`references/pipeline-contract.md`](references/pipeline-contract.md)
before changing stage ownership.

### Initialize immutable evidence

```bash
python3 <skill-root>/scripts/init_experiment.py \
  --name subject-001 \
  --output <runs>/subject-001 \
  --project <project-root> \
  --profile-id <profile-id> \
  --target-class <class> \
  --template <template-file> \
  --license-record <license-file> \
  --config <profile-config> \
  --view front=<front-image> \
  --view left45=<left-image> \
  --view right45=<right-image>

python3 <skill-root>/scripts/validate_experiment.py \
  --experiment <runs>/subject-001/experiment.json \
  --stage admission
```

Do not overwrite an experiment. Create a new experiment ID when any input, template, profile,
code, model, or threshold changes.

### Admit and lock inputs

Require distinct profile-defined view hashes, adequate resolution and coverage, the same subject,
compatible capture conditions, and reviewed masks/features. Require the template to provide:

- one continuous primary surface with stable POSITION/INDEX topology;
- fixed UVs and a render-to-compute mapping for seam-duplicated vertices;
- semantic regions and stable feature-to-surface bindings;
- explicit attached-component/contact rules;
- source, license, version, SHA-256, and fixed review cameras.

Fail closed. Do not replace a missing template with procedural cranium, cubes, stitched geometry,
or a generated loft and continue calling the result production.

### Fit in bounded stages

Run camera initialization, global similarity, low-frequency form, reliable local features, and
attached-part refinement as separate resumable stages. Use ARAP, cotangent Laplacian, symmetry when
appropriate, bounded displacement, and an inversion barrier. Check all fitted views plus held-out
or orbit views after every stage. Let a failed critical region override any aggregate score.

Manual XY feature correction may refine observed feature positions and connection boundaries.
Do not use it as authority for unseen Z or posterior geometry. For stylized subjects, preserve the
source nose, smile line, lips, and other authored proportions rather than normalizing them into a
generic realistic template.

### Project appearance after geometry acceptance

Match cameras, de-light sources, perform visibility-aware z-buffer projection, normalize exposure,
and bake into the fixed template UVs. Retain source view/pixel, depth, confidence, and inference
provenance. Require identical fitted/skinned/QA/delivered geometry hashes, unchanged UV hash, and
zero maximum UV delta. Keep unseen texture neutral and labelled template inference.

### Record and validate stages

Record `intake`, `fit`, `skin`, `qa`, and `package` in order:

```bash
python3 <skill-root>/scripts/record_stage.py \
  --experiment <runs>/subject-001/experiment.json \
  --stage fit \
  --status pass \
  --metrics-json <native-fit-metrics.json> \
  --artifact fitted-model=<fitted-model.glb>

python3 <skill-root>/scripts/validate_experiment.py \
  --experiment <runs>/subject-001/experiment.json \
  --stage final \
  --output <runs>/subject-001/qa/skill-contract.json
```

Use [`references/quality-gates.md`](references/quality-gates.md) for the metric contract.

## Run a profile-loft preview

Require a reviewed front silhouette and left/right profile silhouettes over a common vertical span.
Prefer independently observed profiles. If profiles are generated from the front or from other
inferred views, record that lineage and keep the result preview-only. Exclude hair, background,
and detached accessories from the geometry mask unless they are intentionally part of the surface.

Record camera yaw/pitch/roll, focal length, principal point, translation, mask hashes, smoothing
parameters, vertical/radial sample counts, and the code hash. Construct the mesh from per-height
front width and profile front/back depth. Cap it only after confirming ring order and winding. The
green or other solid material is review presentation, not reconstruction evidence.

Inspect yaw 0, 45, and 90 degrees before accepting the base preview. A horizontal annular bulge
usually means one strong silhouette span was swept around a full ring. Fix the responsible height
profile or apply a bounded local ring-grid polish; do not hide it with lighting, material, or camera.

Treat exact X/Y/Z bounds as admission evidence only. A candidate can match all six extrema while
its jaw, chin, cheek, or neck contour remains far outside the reference. For strict frame matching,
use target and candidate silhouette points with residual lines and require surface continuity in
the same fixed views.

For an existing regular loft, run the bundled cubic-Hermite polish into a new file:

```bash
python3 <skill-root>/scripts/polish_profile_loft.py \
  --input <source.glb> \
  --output <new-run>/loft-polished.glb \
  --lower-y <band-lower-y> \
  --upper-y <band-upper-y> \
  --support-rows 8 \
  --strength 1.0 \
  --metrics <new-run>/qa/polish-metrics.json
```

Choose the smallest band that covers the defect and place both boundaries on visually acceptable
rings. The operation must change X/Z only, preserve Y, faces, vertex order, topology hash, material
lineage, watertightness, one-component structure, and consistent winding. Render before/after at
yaw 0/45/90. If the defect remains at an intermediate view, adjust the band/support locally rather
than globally smoothing the entire head.

Do not apply skin until the geometry-only preview passes user review. Texture or skin does not
upgrade inferred geometry into observed reconstruction.

## Integrate an attached feature preview

Build and review the feature by itself before touching the accepted base. Lock the base hash,
feature hash, coordinate convention, front axis, image-space side, anatomical side, attachment
angle, connection boundary, and hidden-surface authority. For ears, inspect front, oblique, side,
and rear views; do not infer the posterior surface from the frontal annotation alone.

When continuity is required, use the chain `base -> short organic bridge -> feature plate`.
Require the bridge to overlap both solids while the feature plate does not directly intersect the
base. Mirror the accepted posed feature across the declared sagittal plane, reverse winding after
the reflection, and verify exact canonical mirror residuals before applying any bounded placement
clearance.

Use the bundled scripts for deterministic mesh reflection, ordered exact Boolean union, and final
surface validation. Run the Boolean as `base + bridge`, then `assembly + feature`; never replace a
failed exact union with SDF, voxels, or Marching Cubes. Preserve failed attempts, diagnose their
triangle pairs, and bound retries. Validate both the in-memory mesh and persisted GLB, then keep
`userSignoff=false` until the fixed-view sheet is reviewed.

Read the attached-feature reference for commands, collision predicates, coplanar-seam handling,
run layout, and required metrics. Numeric clearances from one subject are evidence records, not
defaults for another subject.

## Preserve hard invariants

- Keep one declared surface lineage per delivered artifact.
- Keep SDF/occupancy/voxels/points `role=qa-only`, `surfaceGenerated=false`; never use Marching
  Cubes as the production delivery surface.
- Reject non-finite vertices, unintended disconnected components, boundary/non-manifold edges,
  degenerate/inverted faces, self-intersections, and undeclared collisions.
- Reject independent ear, limb, handle, branch, or carrier geometry when continuity is required.
- Reject appearance/model geometry mismatch even when screenshots look plausible.
- Reject a global visual pass when a critical region fails.
- Preserve observed/inferred labels through fitting, projection, QA, and packaging.
- Keep automated gates separate from user visual signoff.

Use `img2threejs` only for reference decomposition, material vocabulary, comparison sheets, and
multi-angle review. Use Blender for explicit inspection or asset preparation only; never let a
Blender failure trigger an undeclared fallback surface.

## Return evidence

Report:

- route, output state, experiment path, and subject profile;
- artifact, input, template/config/code hashes, and exact surface lineage;
- observed, annotated, generated, template-prior, and inferred evidence separately;
- geometry/topology/UV identity checks appropriate to the route;
- per-view and critical-region failures, fixed-view before/after renders, and remaining uncertainty;
- generated GLB/package/viewer artifacts, tests run, network activity, and download approvals.

Use `preview`, `automated-gates-passed`, or `user-accepted` precisely. A profile-loft artifact remains
`preview` even after its automated topology gates and local visual polish pass.
