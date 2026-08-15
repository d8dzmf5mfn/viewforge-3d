# Pipeline contract

## State machine

Use this order and never skip a failed upstream gate:

```text
draft -> admission-ready -> intake -> fit -> skin -> qa -> package -> user-signoff
             |               |       |      |       |
             +------------ blocked / failed ----------+
```

Keep each stage resumable by hashing its inputs, config, code, models, and upstream signature.
Changing any dependency invalidates that stage and every downstream stage.

## Layer ownership

| Stage | Owns | Must emit | Must not do |
| --- | --- | --- | --- |
| Admission | files, roles, hashes, dimensions | immutable evidence ledger | infer hidden geometry |
| Intake | normalization, masks, features | reviewed masks, feature records | accept inconsistent views |
| Camera | intrinsics/extrinsics | one camera per view, shared values when justified | hide a geometry error with camera drift |
| Fit | template vertex positions | fitted geometry hash, per-view metrics | change topology or UV |
| Skin | colour projection | atlas, source/confidence maps, provenance | move vertices or create a mesh |
| QA | topology, collision, distance | machine-readable hard-gate report | repair or replace the surface silently |
| Package | delivery archive and viewer contract | hashes, artifacts, notices | include a second competing final surface |

## Template preparation

Prepare a template once per subject class, not once per identity. Lock:

- compute vertices/faces and delivery vertices/faces;
- render-to-compute mapping for UV seams;
- canonical UV and its hash;
- semantic regions and critical feature bindings;
- attached-component topology/contact rules;
- fixed review cameras;
- source, license, version, and SHA-256.

Reject templates with a visibly stitched primary surface. Separate eyeballs, teeth, or mechanical
inserts are valid only when the profile declares them and defines contact/intersection gates.

## Fitting schedule

Use bounded, reversible stages:

1. initialize camera and global similarity;
2. fit low-frequency whole-form basis;
3. fit reliable silhouette and feature constraints;
4. fit profile-specific critical regions;
5. fit attached-component parameters;
6. attenuate unsafe local offsets and re-run geometry gates.

Keep per-stage snapshots. If a stage creates non-finite loss, inversion, self-intersection, or a
critical-view regression, restore its last accepted snapshot and mark the stage failed. Do not
continue with the unsafe geometry.

## Appearance projection

Project only after fit acceptance. For every source sample, retain view role, pixel coordinate,
camera depth, target triangle/barycentric coordinate or vertex, confidence, and inference flag.
Blend by visibility, incidence angle, sharpness, mask confidence, and region priority. Normalize
exposure before blending and de-light inputs before treating them as albedo.

Keep unseen texture neutral and mark it inferred. Never copy face, logo, or other identity features
into unobserved regions.

## Diagnostics and delivery

Use signed-distance, occupancy, point clouds, and voxels to measure distance, inside/outside,
holes, or collision. Do not extract the delivery surface from them. Deliver one primary GLB plus
declared secondary components, provenance, fixed-view renders, QA JSON, licenses, and a local-only
viewer/report when available.
