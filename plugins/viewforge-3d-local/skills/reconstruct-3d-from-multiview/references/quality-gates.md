# Quality gates and stage metrics

Use these JSON shapes with `record_stage.py`. Native projects may include additional fields, but
must preserve the fields below for final skill validation.

## Intake metrics

```json
{
  "viewConsistencyConfirmed": true,
  "masksConfirmed": true,
  "sameSubjectConfirmed": true
}
```

## Fit metrics

```json
{
  "geometryHash": "64 lowercase hex characters",
  "uvHash": "64 lowercase hex characters",
  "perView": {
    "front": {"silhouetteIou": 0.96, "featureNme": 0.012},
    "left45": {"silhouetteIou": 0.94, "featureNme": 0.017},
    "right45": {"silhouetteIou": 0.94, "featureNme": 0.017}
  },
  "invertedFaces": 0,
  "selfIntersectionPairs": 0
}
```

Omit `featureNme` only when the profile disables that metric. Never average away a failed view.

## Skin metrics

```json
{
  "geometryHash": "same fitted hash",
  "uvHash": "same canonical UV hash",
  "geometryChanged": false,
  "uvMaximumDifference": 0.0,
  "observedFraction": 0.6,
  "seamDeltaE00Median": 2.0,
  "seamDeltaE00P95": 6.0
}
```

## QA metrics

```json
{
  "geometryHash": "same fitted hash",
  "topology": {
    "connectedComponents": 1,
    "boundaryEdges": 0,
    "nonManifoldEdges": 0,
    "degenerateFaces": 0,
    "invertedFaces": 0,
    "selfIntersectionPairs": 0
  },
  "sdf": {
    "role": "qa-only",
    "surfaceGenerated": false
  },
  "reviewViews": ["front", "left45", "right45", "side"],
  "criticalRegionsPassed": true,
  "userSignoff": true
}
```

Set `userSignoff` only after the user views the fixed renders. Automated success alone is false.

## Package metrics

```json
{
  "geometryHash": "same fitted hash",
  "finalSurfaceSource": "deformed-template",
  "localOnly": true,
  "externalRequests": 0
}
```

## Hard failure rules

- Fail when any required view misses its profile threshold.
- Fail when a critical region fails even if all aggregate metrics pass.
- Fail when fitted, skin, QA, and package geometry hashes differ.
- Fail when fitted and skin UV hashes differ or UV delta is non-zero.
- Fail on unintended disconnected components, boundary/non-manifold edges, degenerate/inverted
  faces, self-intersection, non-finite values, or undeclared collision.
- Fail when SDF or occupancy generates the delivery surface.
- Fail when unseen areas are presented as observed.
- Fail when the package/viewer makes an external request for private inputs or models.

Use exactly one state after review: `continue`, `refine-profile`, `refine-fit`, `request-input`, or
`stop`. Bound correction attempts and stop on repeated defect, oscillation, or plateau.
