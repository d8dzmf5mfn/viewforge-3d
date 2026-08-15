---
name: same-geometry-skin
description: "Project, paint, bake, replace, or repair a 3D model's skin and materials while keeping accepted geometry and UV topology exact. Use for face or character texture transfer, visibility-aware multiview projection, Blender texture paint, ear-material preservation, seam repair, and GLB/glTF appearance updates that must not move vertices or change faces."
---

# Same-geometry skin

Apply appearance only after the geometry state is accepted for the current stage. Texture quality
does not upgrade inferred or preview geometry into observed reconstruction.

## Lock geometry and evidence

1. Record source model, mesh, topology, face-order, UV, material-partition, and transform hashes.
2. Classify every appearance input as observed, user-authored, model-inferred, template-prior, or
   derived. Preserve parent hashes and camera records.
3. Require the fitted and skinned artifacts to share identical vertex positions, faces, UVs, and
   scene transforms.
4. Preserve declared independent material regions such as left/right ears, eyes, mouth, or scalp.

## Produce the skin

- Match source cameras to the accepted geometry.
- De-light and exposure-normalize sources before blending.
- Use z-buffer visibility and view-angle confidence; never project through the head or across
  occluded folds.
- Blend seams and fill gaps in texture space only. Keep each texel's source view/pixel, confidence,
  and inference label when the pipeline supports it.
- Leave unobserved areas neutral or label their fill as inferred/template-prior.
- In Blender, use UV Editor, Shader Editor, Texture Paint, and image baking without geometry
  modifiers. Save new texture files and a new `.blend`/GLB rather than overwriting the source.
- Keep material tuning separate from surface smoothing. For ceramic, jade, or another hard-surface
  look, adjust Principled BSDF roughness, IOR/specular level, and coat in a material-only derivative;
  do not move vertices to improve highlights.

Do not move vertices to make a texture line up. Correct cameras, UV correspondence, projection, or
source selection instead.

## Validate the result

Require exact equality for persisted positions, faces, topology hash, UV values, material
partitions, and transforms. Confirm all referenced textures exist, decode correctly, have recorded
hashes, and are embedded or packaged as intended. Render fixed front, oblique, side, and rear views
under both consistent neutral lighting and the intended hard-material lighting. Inspect critical
seams, ears, eyes, mouth, occlusion boundaries, and whether strong reflections exaggerate inherited
bands or dents.

Return the source/output model and texture hashes, geometry/UV identity proof, per-view provenance,
fixed renders, remaining inferred areas, and `pendingUserSignoff`.
