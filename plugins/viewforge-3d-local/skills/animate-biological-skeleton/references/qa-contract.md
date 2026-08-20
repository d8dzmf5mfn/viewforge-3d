# Animation and rigid-binding QA contract

## Bone-only Action

Require:

- input Blend hash unchanged;
- source mesh geometry, transforms, materials, parents, vertex groups, and modifiers unchanged;
- one Armature with unchanged non-deforming bones and rest lengths;
- only the profile's named bones have F-curves;
- all animation F-curves target `rotation_quaternion`;
- no keyed translation or scale;
- keyframe directions match reviewed 2D references within the profile tolerance;
- planar motion when the profile declares `Y=0`;
- finite matrices, stable lengths, rest start/end, and bounded per-frame angular steps;
- preview geometry follows bones only through constraints and is not source geometry.

## Rigid segmented binding

Require:

- explicit `segmented=true` and a complete reviewed component-to-bone mapping;
- bind-frame world transforms and mesh digests preserved;
- every mapped source component uses `parent_type=BONE` and the named parent bone;
- zero vertex groups and zero Armature modifiers;
- stable component-to-bone offsets across every animation frame;
- all declared animated components move above the configured threshold;
- all other mapped components remain within transform tolerance;
- the expected Action remains active after save and independent reopen.

Rigid binding is not skinning. It supports block characters and other naturally segmented models.
It must fail closed for a continuous biological mesh when the expected motion needs a bending joint.

## Full-body motion and prop interaction

When the animation includes locomotion, turning, sitting, or a movable prop, also require:

- the Armature Action contains pose-bone quaternion channels only;
- global character translation and rotation live on a separate parent Empty Action;
- every prop has a separate object Action and remains outside the character Armature;
- object Actions contain only the reviewed location and rotation channels, with no keyed scale;
- local bone angular continuity and root-turn continuity are measured separately;
- declared contact frames pass geometry-based distance checks rather than visual inspection alone;
- the final support relation is measured, such as hips centered over a seat and feet near the floor;
- every rigid-bound component affected by global root motion is declared animated;
- the saved file passes the same checks after an independent Blender reopen.

## States and outputs

Use `failed` for any failed gate. Use `pendingUserSignoff` after automated gates pass. Do not mark
motion accepted until the user reviews the fixed-front model-plus-skeleton contact sheet or preview.
