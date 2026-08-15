# Profile-loft preview route

Use this route for stylized/anime geometry exploration or topology-preserving refinement of an
existing regular loft. It is not a substitute for continuous-template production.

## Route contract

Set and preserve:

```json
{
  "route": "profile-loft-preview",
  "previewOnly": true,
  "identityAcceptanceAllowed": false,
  "hiddenSurfacePolicy": "explicit-inference",
  "state": "preview"
}
```

Start from `assets/profile-loft-preview.template.json`. Record each image as one of:

- `observed`: independently captured source evidence;
- `user-annotated`: reviewed semantic or silhouette correction;
- `model-inferred`: generated view or geometry suggestion;
- `template-prior`: pre-existing geometry prior;
- `derived`: mask, render, projection, or measurement derived from another record.

An image generated from the front or from other inferred views is not a new observation. Preserve
its parent hashes and generation record. Generated left/right profiles keep the entire mesh
preview-only even when topology and silhouette checks pass.

## Silhouette preparation

Require a reviewed front mask and two profile masks sharing a non-empty vertical span. Record the
actual camera yaw instead of trusting legacy role names such as `left45` when a camera is at 90
degrees. Lock focal length, principal point, rotation, translation, image dimensions, crop, and
mask hashes.

Exclude hair, background, floating accessories, and facial colour details from the geometry mask
unless the user explicitly wants them in the surface. Keep the geometry pass solid-colour and
untextured until shape review.

## Scanline loft construction

For each sampled height `y`:

1. Intersect the front mask scanline and recover `x_left(y)` and `x_right(y)`.
2. Compute:

```text
center_x(y) = (x_left + x_right) / 2
radius_x(y) = (x_right - x_left) / 2
```

3. Intersect both profile masks and average their back/front depth bounds:

```text
profile_back(y)  = (left_back + right_back) / 2
profile_front(y) = (left_front + right_front) / 2
```

4. Smooth along height, not around the circumference. The validated anime prototype used Gaussian
   sigma values `3.0` for `center_x`, `1.1` for `radius_x`, `2.8` for `profile_back`, `0.7` for the
   detailed front profile, and `7.0` for the front-profile baseline. Treat these as a recorded
   starting configuration, not universal acceptance thresholds.
5. Separate authored front relief from the low-frequency body:

```text
front_detail = smooth(profile_front, 0.7) - smooth(profile_front, 7.0)
middle_z = (front_base + profile_back) / 2
front_radius = front_base - middle_z
back_radius = middle_z - profile_back
```

6. Sample each ring at angle `theta`:

```text
x = center_x + radius_x * sin(theta)
z = middle_z + front_radius * cos(theta)   when cos(theta) >= 0
z = middle_z + back_radius  * cos(theta)   otherwise
z += front_detail * max(cos(theta), 0)^3
```

7. Connect adjacent rings with two consistently wound triangles per radial segment. Add one cap
   vertex at each end after checking ring order.

With `V` vertical samples and `R` radial samples, the regular capped loft contains `V*R + 2`
vertices and `2*V*R` triangles. The validated `256 x 256` prototype therefore contains 65,538
vertices and 131,072 triangles.

## Diagnose annular bulges

A scanline width/depth measurement controls a full ring. A large local span therefore becomes a
horizontal annular ridge even if only the front contour seemed problematic. Inspect:

- front, 45-degree, and side renders with identical camera and lighting;
- ring-center, X-radius, front-depth, and back-depth curves versus height;
- first and second differences of those curves;
- the source mask scanlines at the defective band.

Prefer correcting an erroneous silhouette/profile curve and rebuilding the loft. If the source
curves are acceptable and only a bounded transition is defective, use local cubic-Hermite polish.

## Apply topology-preserving cubic-Hermite polish

Require a regular loft whose ring vertices appear first in bottom-to-top order, with constant Y
inside each ring and two cap vertices at the end. Preserve the original artifact and write a new
output.

Choose lower/upper boundary rings outside the defect and `k` support rows on each side. For every
radial column, take boundary XZ profiles `p0`, `p1` and estimate slopes from support rings:

```text
m0 = (p0 - p(lower-k)) / (y0 - y(lower-k))
m1 = (p(upper+k) - p1) / (y(upper+k) - y1)
```

For each row in the band, with `t=(y-y0)/(y1-y0)`, compute:

```text
target = h00*p0 + h10*(y1-y0)*m0 + h01*p1 + h11*(y1-y0)*m1
new_xz = current_xz + strength*(target-current_xz)
```

where `h00=2t^3-3t^2+1`, `h10=t^3-2t^2+t`, `h01=-2t^3+3t^2`, and
`h11=t^3-t^2`. Never modify Y, faces, vertex order, UVs, or materials. Use the smallest effective
band. A large displacement is a reason for visual review, not proof of failure by itself; compare
it with normalized subject scale and set an explicit maximum when appropriate.

Run `scripts/polish_profile_loft.py` for the deterministic geometry edit. It fails on non-regular
lofts, existing output paths, non-identity scene transforms, unsafe input topology, or persisted
topology drift.

## Quality gates

Require before/after evidence for:

- finite positions;
- geometry hash changed;
- face array and topology hash unchanged;
- one connected component;
- zero boundary and non-manifold edges;
- watertightness and consistent winding;
- unchanged Y outside and inside the edit band;
- no vertex displacement outside the resolved band;
- front/45/side fixed-view comparison;
- explicit user shape review before skin.

Record requested/resolved Y bands, boundary/support rows, strength, edited vertex count, maximum and
mean edited displacement, input/output file hashes, and all gates. Keep state `preview` after an
automated pass. The bundled script does not replace the host project's self-intersection test or
fixed-view renderer; run and record both separately.

## Generic implementation map

- Loft builder: `<project-root>/<loft-builder>` and its recorded function or command.
- Source preview: `<run-root>/<source-id>/models/source.glb`.
- Project wrapper with fixed-view renders: `<project-root>/<render-wrapper>`.
- Reusable geometry-only polish: this skill's `scripts/polish_profile_loft.py`.

Record actual camera yaw, focal length, evidence authority, material values, edit bands, support
rows, strength, displacement, topology counts, and hashes in the current run. Generated side views
remain `model-inferred`; review colour is not provenance. Never copy another run's numeric values
as defaults.
