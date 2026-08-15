---
name: face3d-toolkit-router
description: "Route Face3D Modeling Toolkit tasks to the correct reconstruction, landmark refinement, annotation lowering, smoothing, Blender polishing, or same-geometry skin skill. Use when the user asks how to call the plugin, which Face3D skill to use, or requests an end-to-end workflow spanning multiple geometry or appearance stages."
---

# Face3D toolkit router

Use the smallest skill set that covers the request. Keep geometry stages, appearance stages, and
user acceptance separate.

## Invoke a skill

Use the fully qualified name when calling a skill from this plugin:

- `$face3d-modeling-toolkit:reconstruct-3d-from-multiview` — reconstruct or audit source lineage.
- `$face3d-modeling-toolkit:landmark-guided-refinement` — fit reference contours, then refine
  classified facial features with fixed-view landmark QA.
- `$face3d-modeling-toolkit:annotation-region-lowering` — apply a submitted inward/press-down
  annotation without adding smoothing.
- `$face3d-modeling-toolkit:blender-manual-polish` — guide a manual local Blender polish.
- `$face3d-modeling-toolkit:topology-preserving-smooth` — run an explicitly authorized bounded
  smoothing or fairing pass.
- `$face3d-modeling-toolkit:same-geometry-skin` — change textures or materials while locking
  accepted geometry and UVs.

Invoke this router as `$face3d-modeling-toolkit:face3d-toolkit-router` when the correct branch is
unclear or the request spans multiple branches.

## Route the request

1. Establish whether the artifact is a reconstruction, template derivative, preview, or accepted
   geometry derivative.
2. For a new surface or provenance audit, start with reconstruction.
   For an object, require a dedicated subject profile and read the reconstruction skill's
   `references/object-template-route.md`; do not inherit face landmarks or anatomy assumptions.
3. For an accepted source needing contour/feature work, use landmark-guided refinement.
4. For a user-drawn inward edit, add annotation-region lowering only.
5. Add smoothing or manual polishing only when explicitly authorized. Do not infer permission from
   a request for better quality.
6. Apply skin or material only after the geometry stage is accepted for that iteration.

If the user explicitly requests an unskinned model, stop after geometry QA and fixed-view review.
Neutral color factors may aid inspection, but do not bake image textures or invoke the skin skill.

## Sequence combined work

Use this order unless the user narrows it:

```text
immutable source
  -> provenance and route
  -> coarse contour and thickness fit
  -> classified feature refinement
  -> optional authorized polish/smooth
  -> geometry QA and user review
  -> same-geometry skin/material
  -> final QA and user signoff
```

Never merge a rejected candidate into the next baseline. Keep `rejected`, `preview`,
`automated-gates-passed`, and `user-accepted` distinct. Stop immediately when the user pauses; do
not continue validation, packaging, or the next stage.
