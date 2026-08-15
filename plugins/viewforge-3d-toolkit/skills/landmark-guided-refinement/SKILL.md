---
name: landmark-guided-refinement
description: "Fit an existing head or face mesh to fixed-view reference contours, then refine eyes, nose, mouth, cheeks, chin/jaw, and neck with classified landmarks and thickness checks while preserving topology, UVs, materials, and locked dimensions. Use for strict frame-point fitting, contour overlays, six-view comparison, anime facial refinement, chin-versus-neck separation, cheek-thickness correction, or any coarse-to-fine GLB/Blender refinement that must reject visually damaged candidates even when numeric gates pass."
---

# Landmark-guided refinement

Separate coarse silhouette conformance from facial-feature refinement. Never repair details on a
candidate whose major contour, thickness, or surface continuity is still rejected.

## Lock the baseline

1. Record source paths and hashes, object transforms, vertex/face counts, topology, UV, material
   partitions, dimensions, and fixed review cameras.
2. Render front, left/right 45-degree, left/right 90-degree, and rear views under neutral lighting.
3. Classify each reference as observed, user-marked, inferred, or derived. Use `img2threejs` only
   to decompose references and create review boards; keep Blender or the declared geometry engine
   authoritative for mesh changes.
4. Create a new candidate for every materially different control field or operator. Never overwrite
   the source or promote a rejected pass.

## Fit the coarse frame

1. Normalize reference and candidate cameras/crops before measuring.
2. Draw the target contour, reference points on/inside/outside the contour, candidate silhouette
   points, and point-to-point residual lines. Record pixel and model-unit errors.
3. Treat global X/Y/Z bounds and object scale as a precondition only. Exact bounding-box extrema do
   not prove the intervening contour fits.
4. Correct low-frequency head radius, height, depth, jaw width, chin position, and neck envelope
   before eyes, nose, mouth, or material work.
5. Require both acceptable frame residuals and continuous fixed-view surfaces. A low point error
   with rings, grooves, steps, mouth breaks, or thickness loss is rejected.

Prefer Blender-native low-frequency Lattice, Grab, Scale, Sculpt Filter, or other explicit operators.
Use project tools to produce coordinates, masks, weights, and QA. Use a custom geometry operator
only when the user authorizes it and Blender's native operation cannot meet the contract.

Avoid projecting dense per-row or per-pixel silhouette errors directly onto a high-density mesh;
this creates horizontal bands even when sampled points fit. Prefer sparse low-frequency 3D support,
finite-support handles, ARAP/cage deformation, or a continuous displacement field whose influence
decays in X, Y, and Z. Keep mouth/nose feature blocks rigid when the edit targets only their
placement, and restore hard anchors exactly after each pass.

## Freeze the frame and refine features

After the user stops hard contour fitting or the coarse frame passes, freeze overall dimensions and
build separate landmark groups:

- `E`: eyes and eye sockets;
- `N`: nose bridge, tip, and alae;
- `M`: mouth corners, smile line, and lip contour;
- `F`: cheeks and side-thickness samples;
- `C`: chin and jaw;
- `K`: neck and jaw-neck boundary.

Give each group distinct colors, IDs, and connecting lines. Never connect chin and neck into one
unclassified curve. Preserve the authored anime nose, smile line, and lip proportions instead of
normalizing them into realistic anatomy.

Check cheek and facial depth from both 45-degree and 90-degree views. Allow bounded flattening when
requested. Treat outward inflation as high risk: require explicit need, a small bound, and immediate
thickness review. When the neck is out of scope, hard-lock it by semantic mask or coordinate range
and prove zero displacement.

## Validate and classify every pass

Require:

- unchanged face order, topology, UVs, material partitions, and declared transforms;
- exact locks for dimensions, protected features, outside-region vertices, and out-of-scope groups;
- no non-finite vertices, flips, degenerate faces, boundary/non-manifold regressions, or new local
  intersections;
- per-group landmark movement, thickness deltas, maximum/mean/p95 displacement, and source/output
  hashes;
- fixed six-view renders plus the classified landmark review board.

Classify each attempt as `rejected`, `coarse-frame-pass`, `refinement-preview`, or `user-accepted`.
Technical gates never override visible damage. Keep the best clean baseline and the rejection reason
for every failed pass; do not continue refining a damaged candidate.
