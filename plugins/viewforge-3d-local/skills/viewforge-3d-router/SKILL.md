---
name: viewforge-3d-router
description: "Route ViewForge 3D Local tasks to the correct venv-backed reconstruction, rendering, biological skeleton construction or animation, landmark refinement, annotation lowering, topology-preserving smoothing, Blender polishing, or same-geometry appearance skill. Use when the user asks how to use the separate local plugin or requests an end-to-end local workflow for a person, animal, character, product, or object."
---

# ViewForge 3D Local router

Use the smallest skill set that covers the request. Keep geometry stages, appearance stages, and
user acceptance separate.

## Invoke a skill

Use the fully qualified name when calling a skill from this plugin:

- `$viewforge-3d-local:reconstruct-3d-from-multiview` — reconstruct or audit source lineage.
- `$viewforge-3d-local:render-model-preview` — render immutable fixed-view PNG previews from an
  existing Blend or GLB and return a selected image to the conversation.
- `$viewforge-3d-local:build-biological-skeleton` — derive and validate a bone-only Armature for
  a person, humanoid, quadruped, or other biological model.
- `$viewforge-3d-local:animate-biological-skeleton` — derive a rotation-only Action from
  Imagegen skeleton-overlaid poses, separate full-body or prop trajectories, and optionally Bone
  Parent segmented body parts without skin.
- `$viewforge-3d-local:landmark-guided-refinement` — fit reference contours, then refine
  classified facial features with fixed-view landmark QA.
- `$viewforge-3d-local:annotation-region-lowering` — apply a submitted inward/press-down
  annotation without adding smoothing.
- `$viewforge-3d-local:blender-manual-polish` — guide a manual local Blender polish.
- `$viewforge-3d-local:topology-preserving-smooth` — run an explicitly authorized bounded
  smoothing or fairing pass.
- `$viewforge-3d-local:same-geometry-skin` — change textures or materials while locking
  accepted geometry and UVs.

Invoke this router as `$viewforge-3d-local:viewforge-3d-router` when the correct branch is
unclear or the request spans multiple branches.

This local plugin has two tools that the ChatGPT/Tunnel edition intentionally does not expose:

- `smooth_model_surface` runs the bounded immutable smoothing job from a workspace path or ID.
- `get_local_artifact_path` returns an exact output path for subsequent local CLI or desktop work.

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
7. Use render-model-preview whenever an existing artifact needs visual review. A render does not
   promote `preview` or `automated-gates-passed` to `user-accepted`.

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
  -> fixed-view render preview when requested
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
