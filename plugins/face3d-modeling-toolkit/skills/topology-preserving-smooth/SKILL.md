---
name: topology-preserving-smooth
description: "Apply and audit bounded geometry smoothing, fairing, or loft polishing without changing mesh topology, UVs, materials, or protected features. Use for local ripples, ring bulges, scanline bands, stair-stepping, pinching, and high-frequency surface noise when the user explicitly requests smoothing or polishing rather than a directional shape edit."
---

# Topology-preserving smooth

Require explicit user authorization for geometry smoothing. A request to match a reference, improve
quality, or fix a contour does not authorize smoothing. If the user says not to smooth, preserve
that constraint until they revoke it.

Separate shading, local smoothing, and shape correction before editing:

- Faceted lighting with acceptable geometry: change split/display normals only.
- High-frequency positional noise: use a bounded weighted smooth or fairing pass.
- A regular-loft band or annular bulge: correct its source profile or use cubic-Hermite ring polish.
- A requested inward/outward shape change: use a directional edit skill, not smoothing.

## Establish the contract

1. Hash-lock the source model and compute positions, faces, topology, UV, material, and transform
   evidence.
2. Create a new run and a region mask. Keep protected features and all unselected vertices at zero
   displacement.
3. Record the requested method, pass count, strength, support width, coordinate locks, and symmetry.
4. Do not introduce an arbitrary normal-angle acceptance threshold. Treat normal-change statistics
   as diagnostics; use topology, inversion, intersection, displacement, feature-lock, and fixed-view
   gates for acceptance.

## Choose the smallest operation

For regular profile lofts, use the bundled reconstruction skill's deterministic tool:

```bash
python3 ../reconstruct-3d-from-multiview/scripts/polish_profile_loft.py \
  --input <source.glb> \
  --output <new-run>/loft-polished.glb \
  --lower-y <band-lower-y> \
  --upper-y <band-upper-y> \
  --support-rows 8 \
  --strength 1.0 \
  --metrics <new-run>/qa/polish-metrics.json
```

For an irregular mesh, use an existing project operator that supports region weights, feature
locks, boundary tapering, bounded displacement, and reproducible parameters. Prefer HC,
cotangent-Laplacian, bilateral, or ARAP fairing according to whether volume, curvature edges, or
local rigidity must be preserved. Never substitute a global smooth for a local request.

On a high-density mesh, diagnose whether the defect came from the source positions or from an
applied displacement field. When rings were introduced by contour fitting, smooth or rebuild the
candidate displacement field relative to the last clean baseline; do not repeatedly smooth the
accepted source surface. Avoid dense per-height corrections that move whole horizontal rings.
Prefer finite 3D support with continuous XYZ falloff, and verify that adjacent cheek vertices do not
become the new silhouette after a central correction.

When a contract locks one coordinate, project every update into the permitted plane before
persisting it. Reapply exact feature, size-extrema, frame-anchor, and outside-region source
positions after every iteration. If a modifier removes a temporary vertex group, treat it as an
execution detail and rebuild/clean it safely; never continue from a half-applied in-memory state.

## Validate before release

Require:

- geometry changed only inside authorized support;
- identical face order, topology hash, UVs, material partitions, and scene transforms;
- exact protected-feature and outside-region locks;
- no non-finite vertices, boundary/non-manifold regressions, flips, degeneracy, or new intersections;
- recorded maximum, mean, p99 displacement and normal-change diagnostics;
- fixed front, oblique, side, and rear before/after renders;
- a new artifact hash and `pendingUserSignoff`.

Do not add texture to hide a geometric failure. Do not call a numerically passing preview accepted
until the user reviews the fixed views. Reject visible rings, grooves, mouth discontinuities, or
thickness drift even when topology and displacement gates pass.
