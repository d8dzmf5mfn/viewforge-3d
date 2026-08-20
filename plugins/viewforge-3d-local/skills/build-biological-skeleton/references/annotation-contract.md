# Skeleton annotation contract

Use built-in `imagegen` edit mode on real fixed-view renders. Generate separate front and side
annotations when both views exist.

## Prompt requirements

- State the species or articulation class.
- Preserve the exact canvas, camera, silhouette, proportions, limb angles, component gaps, and
  background.
- Add only thin bone lines and small joint markers.
- Align joints to visible segment boundaries and bones to component centerlines.
- Include only joints supported by model geometry. Do not invent fingers, toes, wings, horns, tail
  segments, or facial bones that the model does not contain.
- Request no labels, title, legend, arrows, extra text, surface anatomy, texture changes, or
  watermark.

## Evidence boundary

The generated image records intended chains and catches species mistakes. It does not provide
metric XY or Z coordinates. Perspective, redrawing, and small silhouette drift are expected image
generation errors. Derive production joints from mesh components, calibrated multi-view evidence,
or explicit reviewed 3D landmarks.

## Species minima

- Humanoid: root/pelvis, spine, chest, neck, head; clavicle, upper arm, forearm, hand; thigh, shin,
  foot on both sides.
- Quadruped: pelvis, spine, chest, neck, head; paired upper/lower forelimbs and fore paws; paired
  thighs/shins/hocks and hind paws; tail only when supported by geometry.
