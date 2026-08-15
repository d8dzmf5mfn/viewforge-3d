---
name: build-biological-skeleton
description: "Build and validate bone-only Blender armatures for biological 3D models, including humans, humanoids, quadrupeds, and other animals. Use when Codex must mark species-appropriate joints from real model views, derive a skeleton from named rigid components or explicit 3D landmarks, create an Armature without skin weights or mesh deformation, and export auditable skeleton JSON, Blend, and GLB preview artifacts."
---

# Build biological skeleton

Keep the source geometry immutable. Treat image annotations as a visual hypothesis, never as 3D
coordinate authority.

## Establish evidence

1. Record the source path, SHA-256, object names, mesh counts, existing armatures, vertex groups,
   modifiers, and parents.
2. Classify the species and articulation profile: `humanoid-v1`, `quadruped-v1`, or a custom
   profile.
3. Use built-in `imagegen` edit mode to overlay joints and bone chains on fixed front and side
   renders. Read [references/annotation-contract.md](references/annotation-contract.md) before
   prompting.
4. Use annotations only to review species and limb-chain intent. Resolve exact 3D joints from:
   - named segmented components and their actual geometry; or
   - an explicit landmark JSON in Blender world coordinates.
5. If a continuous mesh has neither reliable component semantics nor explicit 3D landmarks, stop
   after annotation. Do not infer unsupported depth.

## Build the Armature

Prefer Blender background CLI or Blender MCP. Use Computer Use only for an explicitly authorized
visible Blender correction that cannot be expressed deterministically.

Run:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
  --python scripts/build_biological_skeleton.py -- \
  --input model.glb \
  --output-dir output-run \
  --profile humanoid-v1 \
  --front-annotation skeleton-front.png \
  --side-annotation skeleton-side.png
```

For a continuous human or animal mesh, also pass `--landmarks landmarks.json`. Use the profile
files in `assets/` as the required landmark and bone-graph contracts. Use `--component-map` only
to map reliable source object names to humanoid semantic roles.

Never add vertex groups, Armature modifiers, automatic weights, mesh parenting, IK controls, or
animation in this skill. Every created bone must have `use_deform=false`.

## Validate

Require all of the following before reporting success:

- source GLB hash unchanged;
- source mesh fingerprints, transforms, materials, parents, vertex groups, and modifiers unchanged;
- one new Armature with the expected bone names and hierarchy;
- zero mesh-to-armature parents, zero Armature modifiers, and zero added weights;
- preview GLB contains no `skins` and no animation;
- `qa.json` reports `passed=true`.

Return `skeleton.json`, `bone-only-armature.blend`, `bone-preview.glb`, `qa.json`, the fixed-view
annotations, and their exact paths. Keep the result `pendingUserSignoff` until the user accepts the
bone placement.

For deterministic visual QA, open the generated Blend file in Blender background mode and run
`scripts/render_skeleton_qa.py -- --output-dir qa-renders`, composite its source and bone layers
with `scripts/composite_skeleton_qa.py --input qa-renders --output qa-renders`, then assemble the
fixed views with the project's six-view sheet utility when available. The renderer changes
materials only in memory and does not save the Blend file.
