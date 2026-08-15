---
name: viewforge-3d-router
description: "Route ViewForge 3D tasks to the correct reconstruction, biological skeleton construction or animation, landmark refinement, annotation lowering, smoothing, Blender polishing, or same-geometry appearance skill. Use when the user asks how to call the plugin, which ViewForge 3D skill to use, or requests an end-to-end workflow spanning geometry, bone-only Armature, no-skin rigid binding, animation, or appearance stages for a person, animal, character, product, or object."
---

# ViewForge 3D router

Use the smallest skill set that covers the request. Keep geometry stages, appearance stages, and
user acceptance separate.

## Invoke a skill

Use the fully qualified name when calling a skill from this plugin:

- `$viewforge-3d-toolkit:reconstruct-3d-from-multiview` — reconstruct or audit source lineage.
- `$viewforge-3d-toolkit:build-biological-skeleton` — derive and validate a bone-only Armature for
  a person, humanoid, quadruped, or other biological model.
- `$viewforge-3d-toolkit:animate-biological-skeleton` — derive a rotation-only Action from
  Imagegen skeleton-overlaid poses, separate full-body or prop trajectories, and optionally Bone
  Parent segmented body parts without skin.
- `$viewforge-3d-toolkit:landmark-guided-refinement` — fit reference contours, then refine
  classified facial features with fixed-view landmark QA.
- `$viewforge-3d-toolkit:annotation-region-lowering` — apply a submitted inward/press-down
  annotation without adding smoothing.
- `$viewforge-3d-toolkit:blender-manual-polish` — guide a manual local Blender polish.
- `$viewforge-3d-toolkit:topology-preserving-smooth` — run an explicitly authorized bounded
  smoothing or fairing pass.
- `$viewforge-3d-toolkit:same-geometry-skin` — change textures or materials while locking
  accepted geometry and UVs.

Invoke this router as `$viewforge-3d-toolkit:viewforge-3d-router` when the correct branch is
unclear or the request spans multiple branches.

## Route the request

1. Establish whether the artifact is a reconstruction, template derivative, preview, or accepted
   geometry derivative.
2. For a new surface or provenance audit, start with reconstruction.
   For an object, require a dedicated subject profile and read the reconstruction skill's
   `references/object-template-route.md`; do not inherit face landmarks or anatomy assumptions.
3. For an accepted source needing contour/feature work, use landmark-guided refinement.
   For a biological source needing joints or an Armature without weights, use biological skeleton.
   For an accepted biological Armature needing motion, keyframe retargeting, or a fix for bones
   moving while segmented body parts remain still, use biological skeleton animation.
4. For a user-drawn inward edit, add annotation-region lowering only.
5. Add smoothing or manual polishing only when explicitly authorized. Do not infer permission from
   a request for better quality.
6. Apply skin or material only after the geometry stage is accepted for that iteration.

If the user explicitly requests an unskinned model, stop after geometry QA and fixed-view review.
Neutral color factors may aid inspection, but do not bake image textures or invoke the skin skill.
If the user requests a static skeleton only, do not add vertex groups, Armature modifiers,
automatic weights, mesh parenting, IK controls, or animation. If the user explicitly requests
bone-only animation, allow rotation-only bone Action data but keep source meshes unbound. If the
user also expects a no-skin model to move, allow Bone Parent only for a reviewed segmented-rigid
model; fail closed for continuous meshes. Keep image-generated bone overlays as visual hypotheses.

## Sequence combined work

Use this order unless the user narrows it:

```text
immutable source
  -> provenance and route
  -> coarse contour and thickness fit
  -> classified feature refinement
  -> optional authorized polish/smooth
  -> geometry QA and user review
  -> biological Armature when requested
  -> pose evidence and bone-only Action when requested
  -> optional rigid segmented binding without skin
  -> animation QA and user review
  -> same-geometry skin/material
  -> final QA and user signoff
```

Never merge a rejected candidate into the next baseline. Keep `rejected`, `preview`,
`automated-gates-passed`, and `user-accepted` distinct. Stop immediately when the user pauses; do
not continue validation, packaging, or the next stage.
