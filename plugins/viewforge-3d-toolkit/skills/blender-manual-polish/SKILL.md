---
name: blender-manual-polish
description: "Guide topology-safe manual polishing of local defects in Blender meshes such as GLB, glTF, FBX, or OBJ assets. Use when the user wants to hand-smooth dents, ripples, ridges, pinching, or noisy facial regions in Blender while preserving source geometry lineage, vertex and face topology, UVs, materials, authored features, and a versioned original."
---

# Blender manual polish

Treat Blender as an explicit inspection and asset-preparation tool. Never let a failed import or
manual edit justify an undeclared remesh, voxel surface, or replacement model.

## Lock the source

1. Record the source path, SHA-256, vertex/face counts, topology hash, UV hash, material slots, and
   object names.
2. Preserve the source GLB. Save the Blender scene and any export under a new versioned name.
3. Capture fixed front, left/right 45-degree, side, and rear views before editing.
4. Identify the visible outer surface. Do not sculpt inner shells, eyeballs, ears, or attached
   components unless the user explicitly includes them.

## Polish manually

1. Import the source through Blender's glTF importer and select only the intended mesh object.
2. Enter Sculpt Mode. Disable Dyntopo, Voxel Remesh, remesh modifiers, subdivision changes,
   decimation, and automatic modifier application.
3. Enable `Front Faces Only` plus occlusion/topology automasking when available. Enable X symmetry
   only when the edit is explicitly bilateral.
4. Mask authored features and transition boundaries that must not move. For a face, default locks
   include the nose body, eyelids and eye sockets, mouth line, lip contour, ears, and jaw silhouette.
5. Use the Smooth brush, or hold Shift while using another sculpt brush. Start with strength
   `0.03..0.10` and a radius only slightly larger than the defect. Apply short, tangent-following
   strokes; do not repeatedly scrub one point.
6. Recheck front and both 45-degree views every two or three strokes. Undo immediately if a contour,
   authored crease, or feature proportion changes.
7. Reduce radius and strength before adding more passes. Stop when the defect is no longer visible
   at review scale; do not chase subpixel noise by flattening the region.

Use Blender-native tools as geometry authority when available. Use project scripts or other skills
to compute coordinates, masks, and QA. Switch to a custom geometry operator only with user
authorization and a recorded reason the native operation could not satisfy the contract. If Blender
MCP is unavailable and the user authorizes GUI control, use Computer Use on the visible Blender
session without changing the declared operation.

For facial work, review eye, nose, mouth, cheek, chin/jaw, and neck groups separately. Check cheek
thickness in both 45-degree and 90-degree views. Treat outward inflation as higher risk than bounded
flattening, and do not polish a rejected coarse silhouette into apparent acceptance.

`Shade Smooth` changes displayed normals only. Do not report it as geometry polishing.

## Export safely

- Save a new `.blend` file before export.
- Export a new GLB with the original object transforms and material assignments.
- Do not apply geometry-changing modifiers during export.
- Never overwrite the source artifact.

## Validate the derivative

Require finite vertices, identical face array and topology hash, unchanged UVs/material partitions,
zero boundary/non-manifold regressions, zero flipped or degenerate faces, and no new local
self-intersections. Record the edited vertex set, maximum/mean displacement, feature-lock maximum
displacement, classified facial landmarks, side-thickness deltas, before/after fixed views,
source/output hashes, and `pendingUserSignoff`.

Return the exact source, `.blend`, exported model, and QA paths. State clearly whether only display
normals changed or vertex positions changed.
