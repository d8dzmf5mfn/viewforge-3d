# ViewForge 3D Local: next-generation architecture

[English](NEXT_LOCAL_ARCHITECTURE.md) | [简体中文](NEXT_LOCAL_ARCHITECTURE.zh-CN.md)

This plan applies only to `viewforge-3d-local`. It does not change the ChatGPT/Tunnel edition or
its release line.

## Design boundary

The model produces semantic intent, evidence roles, bounded parameters, constraints, and an
explicit construction strategy. Local deterministic code owns geometry construction, topology,
serialization, validation, and artifact provenance. ViewForge Asset IR is the handoff contract;
GLB is a validated deployment output, not a reasoning format.

Evidence authority is ordered as follows:

1. `observed` — user-provided views, measurements, landmarks, or masks.
2. `inferred` — geometry implied by observed evidence and declared priors.
3. `generated` — hypothetical hidden views or completion cues; never equal to observations.

## Capability and acceptance states

Capabilities report both implementation status and maturity: `planned`, `experimental`,
`validated`, or `trusted`. A route may be installed yet remain preview-only.

| Capability | Current state | Output boundary |
| --- | --- | --- |
| Declarative primitive compiler | validated, implemented | Compiles after IR validation |
| Six-view visual hull | experimental, implemented | Preview only |
| Parametric template fitting | planned | Fails closed until implemented |
| Learned multi-view completion | planned | Fails closed until implemented |
| Topology-preserving smoothing | validated, implemented | Refinement, never missing-evidence repair |

IR validation means the contract is structurally admissible. Compilation means the deterministic
backend completed. Neither means the shape is visually accepted. A compiled result remains
`needs_model_verification` until canonical rendering, reference comparison when applicable, and
user signoff are complete.

## Six-view reconstruction sequence

The next six-view backend should fit stable canonical topology instead of promoting a visual hull
to final geometry:

1. Validate the six roles, image integrity, subject identity, scale assumptions, masks, and camera
   metadata. Reject mixed subjects and unresolved contradictory evidence.
2. Select an explicit category template with fixed topology, landmarks, symmetry regions, and
   protected features. A person uses one whole-body template spanning the head and face, torso,
   and limbs so identity, proportions, and connections are checked in one scan rather than a
   separate face pipeline. Generic objects use their own category templates.
3. Fit camera and shape parameters against observed silhouettes and landmarks. Weight generated
   or inferred evidence below observed views and retain the weights in provenance.
4. Apply bounded topology-preserving refinement. Record displacement, protected vertices,
   boundary changes, UV/material changes, and volume deltas.
5. Validate finite/nonempty geometry, topology, GLB parsing, six source-aligned renders, canonical
   renders, and reference differences. Keep every failed or pending gate visible.
6. Export immutable Blend/GLB/QA/provenance artifacts. Promotion beyond preview requires explicit
   user signoff.

## Local milestones

- Milestone 1 — complete: typed Asset IR, capability registry, fail-closed routing, deterministic
  primitive compiler, provenance report, and mandatory postconditions.
- Milestone 2 — next: category-specific parametric template registry and a first whole-body
  consistency scanner driven by six observed views, full-body landmarks, and cross-view identity
  and proportion constraints. The head and face, torso, and limbs are fitted and accepted in one
  workflow.
- Milestone 3 — later: optional learned completion adapter with local model/version hashes and
  generated-evidence labeling.
- Milestone 4 — later: repair loop that converts failed metrics into bounded IR parameter updates
  while preserving the original evidence and every compiled revision.

Every milestone remains local stdio, uses the repository virtual environment, stores only private
local paths, and requires no Tunnel or OpenAI API key.
