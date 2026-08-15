# Imagegen pose contract

Use edit mode with the actual fixed front render as the referenced image. Do not regenerate the
character design, proportions, camera, crop, or rest skeleton topology.

Require every keyframe and ending frame to show:

- the complete body inside the frame;
- the complete skeleton overlay, including non-moving bones;
- one bright cyan circular marker at every joint;
- continuous cyan bone lines between the intended joints;
- no motion trails, duplicate limbs, perspective change, labels, or decorative marks;
- the moving limb unobstructed and separated enough for marker detection.

Suggested prompt skeleton:

```text
Edit the supplied fixed-front model render. Preserve the exact block character, proportions,
orthographic front camera, framing, background, and full cyan skeleton overlay. Pose only
<semantic chain> into <pose description>. Keep all joints as separate bright cyan circular markers
and all bones as continuous cyan lines. Show the complete skeleton and full body. Do not add skin
deformation, extra limbs, motion blur, text, labels, or perspective change.
```

Generate one image per reviewed extreme plus an ending rest image. Treat the images as 2D pose
hypotheses. Marker pixels authorize projected X/Z directions only; they never authorize hidden
depth.
