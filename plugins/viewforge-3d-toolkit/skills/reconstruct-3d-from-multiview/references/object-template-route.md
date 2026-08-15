# Continuous-template route for objects

Use this route for phones, products, props, and other non-face subjects. Keep object geometry,
evidence, and acceptance independent from the face pipeline.

## Establish the object profile

Record:

- `profileId`, object class, variant, and coordinate convention;
- authoritative metric dimensions and their source URLs;
- observed, annotated, inferred, and template-prior evidence separately;
- the minimum admitted image resolution and distinct-view requirements;
- visible attached parts, required rear/underside evidence, and unobserved interior scope;
- whether appearance or image textures are requested.

Do not reuse face landmarks, FLAME parameters, anatomy regions, or real-person identity gates.

## Select the geometry source

Start from a project-authored or licensed continuous object template with one declared primary
surface. Record its source, license, version, topology hash, UV hash, semantic regions, and
attached-component policy.

Do not download or import a finished third-party model when the requested route is 2D-to-3D.
Do not replace a missing template with primitives or an SDF shell while continuing to call the
result production reconstruction. A project-authored procedural template is allowed only when its
lineage and inferred parameters are explicit.

## Admit evidence

Require each admitted image to pass the object profile's resolution and coverage gates. Preserve
the original bytes and SHA-256. Treat marketing perspective renders as shape/appearance evidence,
not calibrated orthographic views. Use engineering drawings only within their stated terms and do
not redistribute restricted documents in the output package.

Classify every parameter as one of:

- `authoritative-measurement`;
- `bounded-2d-fit`;
- `appearance-only`;
- `template-prior`;
- `unobserved-not-modeled`.

## Fit and assemble

Fit global metric dimensions first without changing topology. Refine rounded profiles and visible
feature locations in bounded stages. Add cameras, buttons, ports, handles, or similar features as
declared attached components; validate each component independently and state whether overlaps are
intentional preview integration or an exact continuous union.

Do not infer interior, hidden rear, or underside geometry beyond the admitted evidence and template
prior. Create a new immutable run when inputs, dimensions, code, thresholds, or template change.

## Stop before appearance when requested

For an unskinned request:

- set `skinApplied=false` and `imageTexturesApplied=false`;
- permit only neutral material color factors for geometry inspection;
- verify the GLB contains no image or texture entries;
- skip the same-geometry skin skill;
- keep user visual signoff pending after geometry QA.

## Validate and report

Require for the primary compute surface:

- exact expected dimensions within the declared tolerance;
- one connected component, zero boundaries, zero non-manifold edges;
- zero degenerate, duplicate, inverted, and self-intersecting faces;
- unchanged topology and UV hashes after bounded fitting.

Validate attached components individually, reload the persisted GLB, verify geometry names and
dimensions, and render fixed front, oblique, side, rear, top, and bottom views. Use a z-buffered
comparison renderer when near-coplanar glass, screen, or body layers overlap.

Return `geometry-preview` or `automated-gates-passed` until the user reviews the fixed views. Never
write `user-accepted` automatically.

## ViewForge 3D repository example

The repository's iPhone 17 route uses a project-authored `TemplatePhoneV0`, official 2D and metric
evidence, an immutable unskinned GLB run, fixed-view QA, per-component topology checks, GLB
roundtrip validation, and checksums. Run the repository builder only after creating its Python 3.11
environment and admitting the required high-resolution reference images.
