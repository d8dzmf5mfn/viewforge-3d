# Multi-chain interaction contract

Use this contract for full-body locomotion, turning, sitting, reaching, or manipulating a rigid
prop. The bundled three-bone extractor remains the deterministic recipe for simple planar limb
motion; do not reinterpret it as a full-body solver.

## Authority split

- Imagegen skeleton overlays define semantic pose hypotheses and visible chain intent.
- `skeleton.json` defines hierarchy, rest directions, lateral offsets, and bone lengths.
- A reviewed run-local profile defines pose names, schedule, moving chains, contact frames, root
  trajectory, prop geometry, and QA tolerances.
- Blender world geometry defines production contact distances and support alignment.
- Unsupported depth remains explicit and must not be invented from one image.

Require every accepted reference image to show the complete skeleton. Reject merged, missing, or
extra joints. Record rejected references separately and keep every accepted image hash in the
coordinate artifact.

## Action split

Create independent Actions:

1. Bone Action — key only `pose.bones[...].rotation_quaternion`.
2. Character trajectory Action — attach the Armature to a parent Empty and key only reviewed
   `location` and `rotation_euler` channels.
3. Prop Action — attach each rigid prop to its own Empty and key only reviewed object channels.

Never key pose-bone translation or scale. Never parent a prop to the character Armature. Preserve
the Armature world transform when adding its trajectory parent.

## Pose construction

Build the torso chain before arms and legs so shoulder and hip chain heads inherit the current torso
pose. Start each limb chain from its current PoseBone head, then rotate the accepted rest direction
and preserve each Blender rest length. This prevents child bone translation drift when the torso
leans.

Use named phases rather than only start/end poses. For a precise restrained walk, include contact,
weight acceptance, passing, and swing-rise poses for each side. For sitting, include preparation,
controlled lowering, seat contact, and seated settle. Use auto-clamped Bezier interpolation for
organic bone motion. Use linear object rotation when an auto-clamped root turn creates an excessive
per-frame angular peak.

## Geometry-based interaction QA

Measure interaction in the reopened saved file:

- hand or foot endpoint to prop AABB distance at declared contact frames;
- prop displacement between grasp and pull-complete frames;
- total character root turn;
- final hip center to seat center in the horizontal plane;
- final hip height to seat top;
- maximum per-frame local bone angle;
- maximum per-frame character root turn;
- stable rigid component-to-bone offsets across every frame.

Use tolerances scaled to the accepted model and prop. A passing render is evidence only after these
numeric gates pass. Render the model, prop, and complete skeleton from one fixed camera for user
review, and keep the result `pendingUserSignoff` until the motion is accepted.
