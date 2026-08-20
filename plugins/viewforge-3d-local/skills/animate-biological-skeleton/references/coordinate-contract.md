# Coordinate and profile contract

The bundled extractor supports a reviewed three-bone chain shown in a fixed front orthographic
Imagegen overlay.

## Authority

- Image markers define only projected directions.
- `skeleton.json` defines bone hierarchy and rest lengths.
- The animation profile defines semantic bone names, schedule, frame rate, interpolation, and QA
  tolerances.
- Unsupported depth remains zero in Armature space.

For the bundled front camera:

```text
image right  -> -Blender X
image up     -> +Blender Z
depth        -> Blender Y = 0
origin       -> moving shoulder
scale        -> rest upper-arm length = 1
```

## Marker detection

The extractor finds connected cyan components, selects the shoulder nearest the body on the
configured image side, then selects the nearest outward marker as the middle joint and the farthest
outward marker as the chain end. It interpolates the wrist/end split from the actual rest lengths of
the second and third Blender bones.

Reject automatic extraction when the overlay contains merged markers, an occluded chain, fewer
than the configured minimum markers, or ambiguous outward markers. Correct the image or provide a
new reviewed profile; do not silently hand-pick plausible pixels.

## Profile changes

Keep `kind=three-bone-direction-animation`. Supply exactly three connected bones, unique positive
schedule frames, `rest` at both endpoints, and an Imagegen image for every non-rest pose. Include an
ending image in `requiredPoseImages` and name it in `restReferencePose` so the extractor can compare
it to the projected Blender rest direction.

For a side camera, non-planar motion, multiple moving chains, or a different marker layout, create a
new extractor/profile version. Do not reinterpret the bundled front-camera axes.
