---
name: animate-biological-skeleton
description: "Create and validate bone-only Blender animation for humans, humanoids, quadrupeds, and other biological armatures from Imagegen skeleton-overlaid keyframes, including full-body locomotion and prop interaction, then optionally rigid-bind segmented mesh components with Bone Parent while keeping skin weights, Armature modifiers, and mesh deformation absent. Use when Codex must turn reviewed biological pose images into relative joint directions, retarget them onto an existing non-deforming Armature, infer smooth intermediate frames, separate root or prop trajectories from bone rotation, fix a model whose bones animate but segmented body parts remain still, or audit a no-skin biological animation."
---

# Animate biological skeleton

Preserve the accepted source and bone-only baseline. Write every animation, binding, render, and QA
artifact to a new versioned path.

## Gate the input

1. Require an accepted `bone-only-armature.blend` and its `skeleton.json` from
   `$viewforge-3d-toolkit:build-biological-skeleton`.
2. Confirm one Armature, non-deforming bones, unchanged rest lengths, no source vertex groups, no
   Armature modifiers, and no mesh-to-Armature parenting.
3. Classify the model as either:
   - `segmented-rigid`: each anatomical part that must articulate is a separate mesh component; or
   - `continuous`: limb joints share one continuous mesh.
4. For `continuous`, allow the bone-only Action but stop before model binding while skin remains
   forbidden. Bone Parent cannot bend a continuous limb.

## Derive pose evidence

1. Render the actual model and full skeleton from a fixed front camera.
2. Use built-in `$imagegen` edit mode to create the requested key poses and ending pose. Keep the
   complete skeleton visible in every image. Read
   [references/imagegen-pose-contract.md](references/imagegen-pose-contract.md) before prompting.
3. Treat Imagegen output as a 2D pose hypothesis. Do not infer depth from a single front view.
4. Review the semantic chain and reject images with missing, merged, or extra joint markers.
5. Extract normalized directions with `scripts/extract_pose_coordinates.py`. Read
   [references/coordinate-contract.md](references/coordinate-contract.md) when changing cameras,
   species profiles, moving chains, or marker conventions.

Use the bundled humanoid wave profile as a concrete recipe:

```bash
python3 scripts/extract_pose_coordinates.py \
  --skeleton skeleton.json \
  --profile assets/humanoid-wave-v1.animation.json \
  --pose raised=wave-keyframe-raised.png \
  --pose square=wave-keyframe-square.png \
  --pose outward=wave-keyframe-outward.png \
  --pose end=wave-endframe-rest.png \
  --output relative-coordinates.json
```

Read [references/humanoid-wave-recipe.md](references/humanoid-wave-recipe.md) for the four matching
Imagegen edit prompts and semantic pose constraints.

For another species or action, copy the profile to the run directory, change the named three-bone
chain, pose schedule, and reviewed marker-detection side, then validate it before Blender work.

For walking, turning, sitting, reaching, or manipulating a prop, do not force the action through the
three-bone extractor. Read
[references/multi-chain-interaction-contract.md](references/multi-chain-interaction-contract.md)
and create a reviewed multi-chain profile in the run directory. Keep bone rotations, character root
motion, and prop motion in separate Actions.

## Build the bone-only Action

Run Blender deterministically:

```bash
/Applications/Blender.app/Contents/MacOS/Blender bone-only-armature.blend \
  --background --python scripts/build_bone_animation.py -- \
  --input-blend bone-only-armature.blend \
  --skeleton skeleton.json \
  --coordinates relative-coordinates.json \
  --output-blend bone-only-animation-v1.blend \
  --qa bone-animation-qa-v1.json
```

Retarget directions while preserving Blender rest lengths. Key only local bone quaternions. Use
Bezier auto-clamped interpolation to infer intermediate frames; never key translations or scale.
Keep the Imagegen reference frames, derived coordinates, schedule, and action name auditable.

For a full-body interaction, “rotation-only” applies to the Armature Action. Put global character
translation and turning on a parent Empty, and put each rigid prop trajectory on its own Empty.
Audit those object Actions independently; never disguise root motion as pose-bone translation.

Reopen the saved file in a separate Blender process and run `scripts/audit_bone_animation.py`.
Require the gates in [references/qa-contract.md](references/qa-contract.md).

```bash
/Applications/Blender.app/Contents/MacOS/Blender bone-only-animation-v1.blend \
  --background --python scripts/audit_bone_animation.py -- \
  --skeleton skeleton.json \
  --coordinates relative-coordinates.json \
  --output bone-animation-reopen-audit-v1.json
```

## Bind a segmented model without skin

Run this stage only for `segmented-rigid`. Create and review a complete component-to-bone mapping;
start from `assets/segmented-humanoid-v1.bind.json` when names match.

```bash
/Applications/Blender.app/Contents/MacOS/Blender bone-only-animation-v1.blend \
  --background --python scripts/bind_rigid_components.py -- \
  --input-blend bone-only-animation-v1.blend \
  --skeleton skeleton.json \
  --mapping segmented-bind.json \
  --output-blend rigid-bound-animation-v1.blend \
  --qa rigid-bind-qa-v1.json
```

Use Bone Parent per mesh component and preserve its bind-frame world transform. Do not create
vertex groups, Armature modifiers, automatic weights, envelopes, or mesh deformation. Reopen and
run `scripts/audit_rigid_binding.py`; verify animated components move and components mapped to
non-animated bones remain still.

```bash
/Applications/Blender.app/Contents/MacOS/Blender rigid-bound-animation-v1.blend \
  --background --python scripts/audit_rigid_binding.py -- \
  --skeleton skeleton.json \
  --mapping segmented-bind.json \
  --output rigid-bind-reopen-audit-v1.json
```

## Render and return

Run `scripts/render_animation_qa.py` on the saved Blend. For a bound model, render the source and
bone overlay as separate layers, then use `scripts/compose_animation_qa.py`; the final frames must
show both model and complete skeleton. Rendering changes materials only in memory and must not save
the Blend.

```bash
/Applications/Blender.app/Contents/MacOS/Blender rigid-bound-animation-v1.blend \
  --background --python scripts/render_animation_qa.py -- \
  --skeleton skeleton.json \
  --coordinates relative-coordinates.json \
  --output-dir qa-renders \
  --mode auto \
  --frames all

python3 scripts/compose_animation_qa.py \
  --render-dir qa-renders \
  --output-dir qa-preview
```

Use `--frames schedule` for a fast key-pose contact sheet and `--frames all` for the interpolated
animation GIF.

Return the Imagegen key/end frames, relative-coordinate JSON, animation profile, bone-only Blend,
reopen audit, and fixed-view preview. When rigid binding is valid, also return the mapping, bound
Blend, binding QA, reopen audit, and model-plus-skeleton preview. Keep the state
`pendingUserSignoff` until the user accepts the motion.
