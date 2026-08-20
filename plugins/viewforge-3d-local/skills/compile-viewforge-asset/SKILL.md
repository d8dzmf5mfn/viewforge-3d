---
name: compile-viewforge-asset
description: Validate and compile semantic ViewForge Asset IR with the separate local plugin. Use for intent-first procedural assets that should become immutable Blend/GLB outputs through an allowlisted deterministic compiler; do not use it to disguise visual hull, parametric fitting, learned reconstruction, or raw vertex arrays as validated geometry.
---

# Compile a ViewForge asset locally

Use this route when the asset can be described with semantic intent, bounded parameters, and
allowlisted primitives. The IR is an authoring and verification contract; GLB remains an output
format.

## Workflow

1. Call `list_viewforge_capabilities`. Treat maturity and `preview_only` as hard routing inputs.
2. Draft schema version 1 ViewForge Asset IR. Use
   `../../runtime/viewforge-asset-ir-v1.example.json` as the minimal procedural example.
3. Record observed, inferred, and generated evidence separately. Generated hidden views are
   hypotheses and must not receive the same authority as user observations.
4. Do not put raw `vertices`, `faces`, index buffers, byte offsets, Python, or Blender scripts in
   the IR. Use semantic roles, construction strategy, bounded parameters, constraints, and
   acceptance gates.
5. Call `validate_viewforge_ir`. Continue only when `valid=true` and
   `acceptance_state=ready_to_compile`.
6. Call `compile_viewforge_ir`, poll `get_viewforge_job`, then read both `modeling-qa.json` and
   `viewforge-ir-report.json`.
7. Render canonical views. Compilation success means the deterministic job ran; it does not satisfy
   `canonical_render` or `user_signoff` by itself.

## Routing boundaries

- `declarative_primitives_v1` is the only compiler route in this milestone.
- `six_view_visual_hull_v1` stays experimental and preview-only; use the multiview reconstruction
  skill and its specialized tool.
- `parametric_template_fit_v1` and `learned_multiview_completion_v1` are planned capabilities. Do
  not fabricate a fallback or lower them to primitive geometry.
- Surface smoothing is a later, explicitly authorized refinement step and cannot repair missing
  shape evidence.

Keep the original IR, normalized digest, capability maturity, compiled spec digest, QA, renders,
and final human decision traceable as separate records.
