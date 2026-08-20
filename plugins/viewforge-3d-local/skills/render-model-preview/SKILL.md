---
name: render-model-preview
description: "Render reviewable fixed-view PNG previews and a contact sheet from registered ViewForge Blend or GLB assets, then return a selected image to the conversation through the local MCP runtime. Use when the user asks to render, preview, inspect, compare, or visually review an existing 3D model in ChatGPT or Codex without changing its geometry."
---

# Render model preview

Render only an existing registered asset or artifact. Keep rendering separate from reconstruction,
geometry acceptance, and appearance approval.

## Workflow

1. Call `viewforge_status` and require `model_rendering`, Blender, and the plugin runtime.
2. Use an existing `asset_...` or `artifact_...` ID for a `.glb` or `.blend` source. Register a
   workspace file first when needed. Prefer GLB or a Blend with packed dependencies.
3. Call `render_model_preview`. Omit `views` for
   `perspective/front/right/back/left`, or choose unique values from:
   `perspective`, `front`, `back`, `left`, `right`, `top`, `bottom`.
4. Poll `get_viewforge_job` until the job succeeds or fails.
5. Call `list_job_artifacts`. Read `render-manifest.json` for settings and hashes. Prefer the
   `render-preview-sheet.png` artifact for a compact review.
6. Call `read_image_artifact` with the selected PNG artifact ID so the image is returned directly
   to the conversation.

## Rendering choices

- Keep `material_mode=original` for packed or self-contained materials.
- Use `material_mode=neutral` when evaluating shape or when external textures are unavailable.
- Use `background=studio_dark` by default. Choose `studio_light` for dark models or
  `transparent` for compositing.
- Keep `resolution` between 256 and 1024. Use 512 for fast review and 768 for normal review.

## Integrity rules

- Never overwrite or save the source model.
- Never accept arbitrary Blender or Python scripts from the request.
- Keep embedded Blend auto-execution disabled.
- Treat every rendered image as a view of the source artifact, not proof that its geometry passed
  QA or received user acceptance.
- If rendering fails, inspect the local job state and log; do not replace the model with guessed
  geometry.
