# SubjectProfile contract

Create one profile per anatomy or object family. Do not reuse face assumptions for an ear, hand,
torso, shoe, vehicle, or arbitrary prop.

## Required fields

Start from `assets/subject-profile.template.json` and define:

- `profileId` and `targetClass`;
- required input roles, expected pose ranges, and held-out review views;
- feature provider, stable feature mapping, and critical feature groups;
- template topology, UV, semantic regions, attached components, and license source;
- staged deformation parameters and regularizers;
- appearance projection priorities and unseen-region policy;
- SDF/occupancy diagnostic role;
- output nodes, review views, and subject-specific acceptance thresholds.

## Adapter interface

Expose equivalent capabilities even when the host project uses another language or API:

```python
class SubjectProfile:
    id: str

    def required_views(self): ...
    def admit_inputs(self, views): ...
    def load_template(self): ...
    def extract_features(self, view): ...
    def initialize_cameras(self, evidence): ...
    def fit_stages(self): ...
    def semantic_regions(self): ...
    def projection_policy(self): ...
    def quality_contract(self): ...
    def delivery_contract(self): ...
```

Keep host-specific objects behind the adapter. Pass plain arrays and versioned JSON/NPZ records
between stages so a viewer, QA tool, or future profile does not depend on optimizer internals.

## View contract

Define roles by evidence, not filenames. For every role record:

- intended yaw/pitch/roll range;
- whether intrinsics are shared;
- required coverage and allowed occlusion;
- features and silhouette regions it is authoritative for;
- whether it is fitted, held out for review, or both.

At least two non-degenerate angles must constrain any non-planar production surface. Use three or
more for asymmetric or identity-sensitive targets.

## Template contract

Require a primary surface lineage and explicitly list allowed secondary nodes. For each attached
part define:

- whether topology is shared or separate;
- attachment/contact region;
- position, scale, orientation, and allowed deformation parameters;
- gap, penetration, and continuity thresholds;
- which views provide evidence.

Do not use a rectangular carrier, hidden support, or Boolean patch to make an appendage appear
attached unless that structure belongs to the real subject and is declared.

## Feature and deformation contract

Map each feature to stable surface triangles/barycentric coordinates or stable vertex IDs. Split
features into global, silhouette, local-critical, and appearance-only groups. Never let an
appearance landmark move geometry after the fit gate.

Set regularizer weights and displacement limits in normalized subject units. Record scale
normalization so thresholds remain comparable across identities.

## Acceptance contract

Specialize `assets/quality-contract.template.json`. Increase thresholds where identity matters and
add hard critical-region gates. A profile is incomplete until it can name the expected failure,
the metric that detects it, and the recovery action for every critical part.
