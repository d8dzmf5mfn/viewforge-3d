# Humanoid wave Imagegen recipe

Use built-in Imagegen edit mode. Keep the immutable front render as the geometry and camera source.
Character right is viewer-left. Use the complete-skeleton overlay contract for every image.

## Raised keyframe

```text
Edit the supplied fixed-front block character into the first key pose of a friendly right-hand
wave. Keep the torso, head, pelvis, both legs, and character left arm unchanged. Raise only the
character right upper arm diagonally upward about 65 degrees from its hanging rest direction; bend
the right elbow about 95 degrees so the forearm points upward and the hand is beside the head.
Preserve exact cuboid proportions, colors, component lengths, front orthographic camera, scale,
crop, and background. Overlay the complete body skeleton with continuous yellow-orange centered
bone lines and separate bright cyan circular joint markers. Add no skin, face, clothing, fingers,
extra limbs, perspective, text, labels, or watermark.
```

## Square keyframe

```text
Use the immutable front render for geometry and the raised pose for skeleton-overlay style. Keep
head, torso, pelvis, legs, and character left arm unchanged. Keep the right shoulder raised, with
the upper arm approximately horizontal. Bend the right elbow near 90 degrees and point the right
forearm and hand vertically upward to form the square wave extreme. Preserve the exact camera,
crop, scale, background, proportions, and limb lengths. Show the complete body skeleton and every
cyan joint marker. Add no skin, extra anatomy, perspective, text, labels, or watermark.
```

## Outward keyframe

```text
Use the immutable front render for geometry and the raised pose for overlay style. Keep the right
shoulder and elbow at the reviewed raised positions. Rotate only the right forearm plus hand as one
rigid chain about 30 degrees outward toward viewer-left around the fixed elbow. The wrist must move
left of the elbow and the forearm must be diagonal up-left. Keep every other body component
unchanged. Preserve exact front orthographic camera, proportions, colors, lengths, scale, crop, and
background. Show the complete body skeleton and every cyan joint marker. Add no skin, extra limbs,
perspective, text, labels, or watermark.
```

## Ending rest frame

```text
Return the waving character right arm completely to the immutable relaxed rest pose. The ending
pose must visually match the source so the Action finishes cleanly. Preserve all dimensions,
colors, limb lengths, joint centers, front orthographic camera, scale, crop, and background. Show
the complete body skeleton with continuous bone lines and separate cyan joint markers, including
head, neck, clavicles, both arms, spine, pelvis branches, both legs, ankles, and toe tips. Add no
skin, extra anatomy, perspective, text, labels, or watermark.
```

Review all four images before extraction. Reject any pose with a shifted camera, changed body part,
missing full-body bone, merged cyan markers, or an ending arm that differs from the projected
Blender rest chain beyond the animation profile tolerance.
