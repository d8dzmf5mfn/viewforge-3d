# Attached-feature integration preview

Use this workflow for a stylized nose, ear, horn, handle, limb, or similar feature that must be
reviewed independently and then joined to an accepted base mesh. Keep this route `preview` unless
the feature is fitted through the production template contract.

## Contract

Record and preserve:

```json
{
  "route": "profile-loft-preview",
  "state": "preview",
  "previewOnly": true,
  "identityAcceptanceAllowed": false,
  "hiddenSurfacePolicy": "explicit-inference",
  "userSignoff": false
}
```

Classify the base, feature geometry, connection geometry, pose, annotations, inferred rear shape,
and generated review renders separately. Preserve every source hash and create a new run for each
geometry, pose, Boolean order, tolerance, or gate change.

Start the run record from `assets/attached-feature-preview.template.json` and replace every `null`
or placeholder before declaring automated gates passed.

## Lock coordinates and naming

Declare all of the following before positioning geometry:

- world front axis and up axis;
- sagittal plane and its origin;
- front-image `left` or `right` side;
- anatomical side for a front-facing subject;
- local-to-world transform and determinant sign;
- review-camera yaw convention.

Do not use `leftEar` or `rightEar` without the naming convention. For a front-facing head whose
front axis is world `-Z`, front-image left commonly maps to world `+X`, while front-image right maps
to world `-X`; verify this from the accepted head instead of assuming it.

## Approve the standalone feature first

Build the feature as a closed review solid with stable vertex order and outward winding. Review its
silhouette, authored relief, connection boundary, material, and fixed views before integration.
Keep the base unchanged until the feature passes this review.

For a stylized ear:

- preserve the approved outer outline and root arc;
- place detailed auricle relief on the declared anterior side;
- keep the posterior surface smooth unless rear evidence supports more detail;
- record the intended head-to-ear angle, such as 45 degrees;
- treat rear/head support depth as inferred unless an observed rear or profile view supports it.

A front annotation may authorize feature XY and the connection boundary. It does not authorize
unobserved depth or posterior anatomy.

## Build a connection chain

When the art shows the feature standing away from the base, use:

```text
accepted base -> short organic bridge -> approved feature plate
```

The bridge must grow from the base and taper into the feature. It is not a visible pedestal or a
permanent independent node. Before Boolean integration require these collision predicates:

```text
feature face centroids inside base       == 0
bridge face centroids inside base        > 0
bridge face centroids inside feature     > 0
feature face centroids inside bridge     > 0
```

Use signed distance plus multi-ray occupancy or an equivalent robust test. Record sample counts,
inside counts, fractions, minimum, p05, and median signed distance. Do not substitute bounding-box
overlap for volume overlap.

## Mirror an accepted feature

Mirror the accepted posed feature and bridge, not a screenshot and not an already-Booleaned final
head. Reconstruct the accepted posed components from their locked source hashes and pose record
when necessary.

For sagittal plane `X = x0`, apply:

```text
right.x = 2*x0 - left.x
right.y = left.y
right.z = left.z
```

The reflection has negative determinant. Reverse each triangle from `(a,b,c)` to `(a,c,b)`, then
require positive volume and outward winding. Use `scripts/mirror_mesh_npz.py`:

```bash
python3 <skill-root>/scripts/mirror_mesh_npz.py \
  --input <accepted-posed-feature.npz> \
  --output <new-run>/working/mirrored-feature.npz \
  --axis x \
  --origin 0 \
  --metrics <new-run>/references/mirrored-feature.json
```

Verify canonical mirror residuals before placement clearance:

```text
max(abs(right.x + left.x - 2*x0)) == 0
max(abs(right.y - left.y))         == 0
max(abs(right.z - left.z))         == 0
```

If the persisted base is slightly asymmetric and the exact mirror penetrates it, apply the smallest
bounded translation along the declared outward axis. Keep the feature shape, Y/Z placement, and
attachment angle unchanged. Record the pre-clearance collision, translation, scale-normalized
magnitude, and unchanged invariants. Do not treat a subject-specific clearance as a default.

## Run ordered exact Boolean integration

Use exact triangle-mesh Boolean union in this order:

```text
1. accepted base + bridge
2. assembly + feature
```

The bridge is therefore the sole connection. Run Blender explicitly:

```bash
<blender> --background \
  --python <skill-root>/scripts/blender_exact_union_npz.py -- \
  --base <accepted-base.npz> \
  --operand bridge=<posed-bridge.npz> \
  --operand feature=<posed-feature.npz> \
  --output <new-run>/working/boolean-result.npz \
  --stats <new-run>/references/blender-boolean-stats.json
```

Require the stats to say `solver=EXACT`, `voxelOrSdfUsed=false`, and
`marchingCubesUsed=false`. A background-mode crash or Boolean failure is not permission to create a
different surface.

## Handle seam failures narrowly

Run exact self-intersection detection after every Boolean attempt. If a result fails:

1. preserve the failed mesh, stats, intersecting face pairs, and world-space triangle coordinates;
2. classify whether the defect is duplicate seam vertices, a coplanar tangent, or real geometry
   penetration;
3. retry exact/near-exact seam merging only for duplicate vertices;
4. if the output hash and intersecting pairs are unchanged, stop treating it as duplication;
5. for a confirmed coplanar tangent, apply a bounded translation smaller than the declared local
   tolerance along a non-shape-changing direction, then rerun all gates;
6. stop after the bounded retry budget instead of increasing global cleanup tolerances.

Never delete intersecting triangles blindly, globally smooth the base, or widen merge tolerances
until the screenshot looks clean. Record every retry and its geometry change.

## Validate memory and persistence

Run the bundled validator on the Boolean NPZ and the exported GLB:

```bash
python3 <skill-root>/scripts/validate_surface_mesh.py \
  --input <new-run>/working/boolean-result.npz \
  --metrics <new-run>/qa/boolean-surface-gates.json \
  --exact-self-intersections

python3 <skill-root>/scripts/validate_surface_mesh.py \
  --input <new-run>/models/result.glb \
  --metrics <new-run>/qa/persisted-glb-gates.json \
  --exact-self-intersections
```

Require both passes to report:

- finite positions and positive volume;
- one connected component;
- zero boundary, non-manifold, degenerate, and duplicate faces;
- watertight and consistently wound geometry;
- zero exact self-intersecting triangle pairs;
- for GLB, one geometry, one geometry node, and identity scene transform;
- the accepted material after persistence.

Render fixed front, both obliques, both sides, and rear. Inspect the critical attachment from the
front and rear; a front-only sheet cannot validate the bridge or posterior surface. Automated gates
do not set `visualReviewPassed` or `userSignoff`.

## Recommended run layout

```text
<run>/
  environment.json
  metrics.json
  models/result.glb
  working/base.npz
  working/bridge-posed.npz
  working/feature-posed.npz
  working/boolean-result.npz
  qa/pose-before-boolean-overview.png
  qa/overview-six-view.png
  qa/boolean-surface-gates.json
  qa/persisted-glb-gates.json
  references/pose.json
  references/blender-boolean-stats.json
  references/failed-attempt-*.json
  references/source-*-metrics.json
```

Include source/code hashes, coordinate mapping, collision evidence, canonical symmetry residuals,
bounded clearances, Boolean stage counts, topology hashes, material values, fixed camera records,
network activity, and the exact state in `metrics.json`.

## Generic integration record

Record project-local artifacts without embedding workstation paths or private experiment names:

- standalone feature: `<run-root>/<feature-id>`;
- support bridge: `<run-root>/<bridge-id>`;
- accepted unilateral or unsymmetrical base: `<run-root>/<base-id>`;
- integration code: `<project-root>/<integration-script>`;
- accepted result: `<run-root>/<result-id>`.

Record the coordinate map, mirror residuals, bounded clearances, failed Boolean attempts, exact
intersection pairs, and persisted topology metrics for the current mesh. Never reuse numeric
clearances, vertex counts, triangle counts, or tolerances from another subject as defaults. The
state remains `preview` with `userSignoff=false` until the exact candidate is reviewed.
