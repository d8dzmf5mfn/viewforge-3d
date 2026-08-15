---
name: annotation-region-lowering
description: "Convert user-drawn screen-space or surface annotations into bounded inward mesh displacement while preserving topology, UVs, materials, and protected features. Use when the user asks to lower, press in, recess, flatten inward, or deepen selected GLB/glTF mesh regions, especially when submitted polygons, brush strengths, mirrored regions, or no-smoothing constraints must be followed exactly."
---

# Annotation region lowering

Treat the annotation as edit authority only for the surface it actually hits. A 2D polygon is not
authority for hidden depth, posterior geometry, or the opposite side unless the package explicitly
requests mirroring.

## Validate the annotation package

Require the locked source path/version/hash, fixed camera and viewport, operation type, polygon
coordinates, surface-hit samples, per-region strength, mirror flag, global note, protected regions,
and picking-proxy lineage. Reject a proxy/model mismatch or a polygon that cannot be reprojected to
the same source surface.

If only half of a visible selection becomes active, inspect visibility, facing tests, seam mapping,
surface-component filters, and mirror authorization. Do not compensate by expanding the region
beyond its submitted boundary. Report active vertex counts for both sides and ensure diagnostic
images do not hide one side with a display-only filter.

## Build authorized support

1. Reproject every submitted surface sample to the immutable full-resolution source.
2. Form support only from vertices covered by the submitted polygon and its declared brush radius.
3. Multiply support by the submitted strength and boundary taper. Normalize region-local support
   only when needed to make the user's stated strength observable, and record that normalization.
4. Apply declared mirroring through the locked bilateral axis, then independently validate the
   mirrored support against the source surface.
5. Subtract protection annotations and semantic locks before displacement.

## Apply inward displacement

Use the immutable source normal, never an iteratively smoothed candidate normal. With a locked Y
coordinate, apply only the normalized XZ component:

```text
direction = normalize([normal.x, 0, normal.z])
candidate = source - direction * support * strength * maximum_displacement
candidate.y = source.y
```

Use the project's declared coordinate system and inward sign. If Y is not locked, use the full
source normal. Do not add smoothing, relaxation, fairing, polishing, inflation, or contour changes
unless the user separately requests them.

Find a stronger safe result with bounded line search. Stop before the first flip, degeneracy,
self-intersection, feature-lock violation, or displacement-bound failure. Do not use a universal
normal-angle cutoff as a hidden strength cap; normal changes are advisory unless the project
contract explicitly says otherwise.

Keep submission and execution separable when the user wants manual control. Persist the annotation
package and mark it pending; do not automatically start another agent or geometry job. Execute only
after the user says the submitted regions are ready.

## Preserve and prove

Require unchanged face order/topology hash, UVs, material partitions, textures, transforms, and all
protected-feature positions. Require zero outward displacement, exact locked-coordinate equality,
zero flipped/degenerate faces, and exact local intersection testing for every moved-face candidate.
Render fixed before/after views including the rear, record active vertices per side, per-region
strengths, maximum/mean/p99 displacement, source/output hashes, and `pendingUserSignoff`.

Create a new versioned model. Never overwrite the locked source or use the picking proxy as the
rendered, validated, or delivered surface.
