from __future__ import annotations

import hashlib
import json
import math
import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
)

from .models import (
    DeclarativeBevel,
    DeclarativeMaterial,
    DeclarativeModelObject,
    DeclarativeModelSpec,
    DeclarativeWorld,
    PublicModel,
)

MAX_IR_BYTES = 128 * 1024
IDENTIFIER_PATTERN = r"^[a-z][a-z0-9_-]{0,63}$"
PARAMETER_PATTERN = r"^[A-Za-z][A-Za-z0-9]{0,63}$"
REFERENCE_PATTERN = r"^(asset|artifact)_[a-f0-9]{8,64}$"
REQUIRED_ACCEPTANCE_GATES = {
    "finite_geometry",
    "glb_parse",
    "canonical_render",
    "user_signoff",
}


class CapabilityMaturity(StrEnum):
    PLANNED = "planned"
    EXPERIMENTAL = "experimental"
    VALIDATED = "validated"
    TRUSTED = "trusted"


class CapabilityDescriptor(PublicModel):
    id: str
    title: str
    strategy: Literal[
        "procedural",
        "parametric",
        "multiview",
        "learned",
        "refinement",
        "verification",
        "export",
    ]
    maturity: CapabilityMaturity
    implemented: bool
    available: bool
    preview_only: bool = False
    recommended_tool: str | None = None
    summary: str


class CapabilityRegistry(PublicModel):
    schema_version: Literal[1] = 1
    capabilities: list[CapabilityDescriptor]


class ViewForgeIRCoordinateSystem(PublicModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    units: Literal["meters"] = "meters"
    up_axis: Literal["Z"] = Field(default="Z", alias="upAxis")
    handedness: Literal["right"] = "right"


class ViewForgeIREvidence(PublicModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    role: str = Field(pattern=IDENTIFIER_PATTERN)
    source_id: str = Field(alias="sourceId", pattern=REFERENCE_PATTERN)
    authority: Literal["observed", "inferred", "generated"]
    confidence: float = Field(ge=0.0, le=1.0)


class ViewForgeIRConstraint(PublicModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    kind: Literal["geometry", "topology", "visual", "semantic", "format", "provenance"]
    metric: str = Field(pattern=IDENTIFIER_PATTERN)
    operator: Literal["eq", "lte", "gte", "between", "required"]
    value: Any
    severity: Literal["error", "warning"] = "error"


class ViewForgeIRProceduralObject(PublicModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(min_length=1, max_length=80)
    semantic_role: str | None = Field(
        default=None,
        alias="semanticRole",
        pattern=IDENTIFIER_PATTERN,
    )
    primitive: Literal[
        "cube",
        "uv_sphere",
        "ico_sphere",
        "cylinder",
        "cone",
        "torus",
    ]
    parameters: dict[str, StrictFloat | StrictInt] = Field(default_factory=dict)
    location: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_degrees: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        alias="rotationDegrees",
    )
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    material: DeclarativeMaterial | None = None
    bevel: DeclarativeBevel | None = None
    shade_smooth: bool = Field(default=False, alias="shadeSmooth")


class ViewForgeIRConstruction(PublicModel):
    strategy: Literal["procedural", "parametric", "multiview", "learned"]
    capability: str = Field(pattern=IDENTIFIER_PATTERN)
    objects: list[ViewForgeIRProceduralObject] | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )
    parameters: dict[str, StrictStr | StrictInt | StrictFloat | StrictBool] = Field(
        default_factory=dict
    )


class ViewForgeIRValidationPolicy(PublicModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    minimum_maturity: CapabilityMaturity = Field(
        default=CapabilityMaturity.VALIDATED,
        alias="minimumMaturity",
    )
    required_gates: list[
        Literal[
            "finite_geometry",
            "nonempty_geometry",
            "topology",
            "glb_parse",
            "canonical_render",
            "reference_comparison",
            "user_signoff",
        ]
    ] = Field(
        default_factory=lambda: [
            "finite_geometry",
            "nonempty_geometry",
            "glb_parse",
            "canonical_render",
            "user_signoff",
        ],
        alias="requiredGates",
        min_length=1,
    )


class ViewForgeAssetIR(PublicModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal[1] = Field(default=1, alias="schemaVersion")
    name: str = Field(min_length=1, max_length=80)
    asset_type: str = Field(alias="assetType", pattern=IDENTIFIER_PATTERN)
    intent: str = Field(min_length=1, max_length=2_000)
    coordinate_system: ViewForgeIRCoordinateSystem = Field(
        default_factory=ViewForgeIRCoordinateSystem,
        alias="coordinateSystem",
    )
    construction: ViewForgeIRConstruction
    evidence: list[ViewForgeIREvidence] = Field(default_factory=list, max_length=64)
    constraints: list[ViewForgeIRConstraint] = Field(default_factory=list, max_length=128)
    validation_policy: ViewForgeIRValidationPolicy = Field(
        default_factory=ViewForgeIRValidationPolicy,
        alias="validationPolicy",
    )


class ViewForgeIRValidation(PublicModel):
    schema_version: Literal[1] = 1
    valid: bool
    ir_sha256: str | None = None
    normalized_ir: dict[str, Any] | None = None
    capability: CapabilityDescriptor | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    execution_plan: list[str] = Field(default_factory=list)
    required_postconditions: list[str] = Field(default_factory=list)
    acceptance_state: Literal[
        "rejected",
        "ready_to_compile",
        "valid_for_specialized_tool",
        "preview_only",
        "planned_not_executable",
    ]
    next_tool: str | None = None


_MATURITY_ORDER = {
    CapabilityMaturity.PLANNED: 0,
    CapabilityMaturity.EXPERIMENTAL: 1,
    CapabilityMaturity.VALIDATED: 2,
    CapabilityMaturity.TRUSTED: 3,
}


def build_capability_registry(
    *,
    blender_available: bool,
    modeling_runtime_available: bool,
) -> CapabilityRegistry:
    return CapabilityRegistry(
        capabilities=[
            CapabilityDescriptor(
                id="declarative_primitives_v1",
                title="Deterministic declarative primitive compiler",
                strategy="procedural",
                maturity=CapabilityMaturity.VALIDATED,
                implemented=True,
                available=blender_available,
                recommended_tool="compile_viewforge_ir",
                summary=(
                    "Compiles allowlisted primitives and bounded transforms into immutable "
                    "Blend/GLB outputs; raw mesh arrays and arbitrary code are excluded."
                ),
            ),
            CapabilityDescriptor(
                id="six_view_visual_hull_v1",
                title="Six-view silhouette visual hull",
                strategy="multiview",
                maturity=CapabilityMaturity.EXPERIMENTAL,
                implemented=True,
                available=modeling_runtime_available,
                preview_only=True,
                recommended_tool="reconstruct_six_view_visual_hull",
                summary=(
                    "Intersects six reviewed silhouettes. It cannot recover hidden concavities "
                    "and must remain preview-only."
                ),
            ),
            CapabilityDescriptor(
                id="parametric_template_fit_v1",
                title="Parametric template fitting",
                strategy="parametric",
                maturity=CapabilityMaturity.PLANNED,
                implemented=False,
                available=False,
                summary=(
                    "Planned next route for fitting stable canonical topology to silhouettes and "
                    "landmarks instead of treating a visual hull as final geometry."
                ),
            ),
            CapabilityDescriptor(
                id="learned_multiview_completion_v1",
                title="Learned multi-view completion",
                strategy="learned",
                maturity=CapabilityMaturity.PLANNED,
                implemented=False,
                available=False,
                summary=(
                    "Reserved for a future local model adapter. Generated hidden views must be "
                    "lower-authority evidence than user observations."
                ),
            ),
            CapabilityDescriptor(
                id="topology_preserving_smooth_v1",
                title="Topology-preserving surface smoothing",
                strategy="refinement",
                maturity=CapabilityMaturity.VALIDATED,
                implemented=True,
                available=blender_available,
                recommended_tool="smooth_model_surface",
                summary="Applies bounded immutable smoothing and emits displacement/topology QA.",
            ),
            CapabilityDescriptor(
                id="local_verification_v1",
                title="Local contract and artifact verification",
                strategy="verification",
                maturity=CapabilityMaturity.VALIDATED,
                implemented=True,
                available=True,
                recommended_tool="validate_viewforge_ir",
                summary=(
                    "Validates IR structure, evidence authority, capability maturity, acceptance "
                    "gates, and deterministic compilation eligibility."
                ),
            ),
        ]
    )


def _canonical_bytes(document: dict[str, Any]) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _validation_error_messages(error: ValidationError) -> list[str]:
    messages: list[str] = []
    for issue in error.errors(include_url=False)[:32]:
        location = ".".join(str(part) for part in issue["loc"])
        messages.append(f"{location or 'document'}: {issue['msg']}")
    return messages


def _finite_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return isinstance(value, str)


def _constraint_value_valid(constraint: ViewForgeIRConstraint) -> bool:
    value = constraint.value
    if constraint.operator == "between":
        return (
            isinstance(value, list)
            and len(value) == 2
            and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
            and all(math.isfinite(float(item)) for item in value)
            and float(value[0]) <= float(value[1])
        )
    if constraint.operator == "required":
        return isinstance(value, bool) and value is True
    return _finite_scalar(value)


def validate_viewforge_ir_document(
    document: dict[str, Any],
    registry: CapabilityRegistry,
) -> tuple[ViewForgeAssetIR | None, ViewForgeIRValidation]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        raw_bytes = _canonical_bytes(document)
    except (TypeError, ValueError) as error:
        return None, ViewForgeIRValidation(
            valid=False,
            errors=[f"document is not finite JSON: {error}"],
            acceptance_state="rejected",
        )
    if len(raw_bytes) > MAX_IR_BYTES:
        return None, ViewForgeIRValidation(
            valid=False,
            errors=[f"document exceeds the {MAX_IR_BYTES}-byte IR limit"],
            acceptance_state="rejected",
        )
    digest = hashlib.sha256(raw_bytes).hexdigest()
    try:
        parsed = ViewForgeAssetIR.model_validate(document)
    except ValidationError as error:
        return None, ViewForgeIRValidation(
            valid=False,
            ir_sha256=digest,
            errors=_validation_error_messages(error),
            acceptance_state="rejected",
        )

    normalized = parsed.model_dump(by_alias=True, exclude_none=True, mode="json")
    normalized_digest = hashlib.sha256(_canonical_bytes(normalized)).hexdigest()
    capabilities = {item.id: item for item in registry.capabilities}
    capability = capabilities.get(parsed.construction.capability)
    if capability is None:
        errors.append(f"unknown local capability: {parsed.construction.capability}")
    elif capability.strategy != parsed.construction.strategy:
        errors.append(
            "construction.strategy does not match the registered capability strategy "
            f"({capability.strategy})"
        )

    if parsed.construction.strategy == "procedural":
        if not parsed.construction.objects:
            errors.append("procedural construction requires at least one allowlisted object")
    elif parsed.construction.objects is not None:
        errors.append("construction.objects is permitted only for the procedural strategy")

    for object_ in parsed.construction.objects or []:
        for key, value in object_.parameters.items():
            if not re.fullmatch(PARAMETER_PATTERN, key):
                errors.append(f"{object_.name}.parameters contains an invalid key: {key}")
            if isinstance(value, bool) or not math.isfinite(float(value)):
                errors.append(f"{object_.name}.parameters.{key} must be a finite number")
        if any(not math.isfinite(float(value)) for value in (*object_.location, *object_.scale)):
            errors.append(f"{object_.name} contains a non-finite transform")
        if any(value <= 0 for value in object_.scale):
            errors.append(f"{object_.name}.scale values must be positive")

    evidence_roles = [item.role for item in parsed.evidence]
    if len(evidence_roles) != len(set(evidence_roles)):
        errors.append("evidence roles must be unique")
    for evidence in parsed.evidence:
        if evidence.authority == "generated" and evidence.confidence > 0.75:
            warnings.append(
                f"generated evidence {evidence.role} has confidence above 0.75; "
                "generated hidden views are hypotheses, not observations"
            )
    if parsed.construction.capability == "six_view_visual_hull_v1":
        required_roles = {"front", "back", "left", "right", "top", "bottom"}
        missing = sorted(required_roles - set(evidence_roles))
        if missing:
            errors.append(f"six-view visual hull is missing evidence roles: {missing}")

    constraint_ids = [item.id for item in parsed.constraints]
    if len(constraint_ids) != len(set(constraint_ids)):
        errors.append("constraint IDs must be unique")
    for constraint in parsed.constraints:
        if not _constraint_value_valid(constraint):
            errors.append(
                f"constraint {constraint.id} has an invalid value for {constraint.operator}"
            )

    required_gates = parsed.validation_policy.required_gates
    if len(required_gates) != len(set(required_gates)):
        errors.append("validationPolicy.requiredGates must not contain duplicates")
    missing_gates = sorted(REQUIRED_ACCEPTANCE_GATES - set(required_gates))
    if missing_gates:
        errors.append(f"validationPolicy is missing mandatory gates: {missing_gates}")

    if (
        capability is not None
        and _MATURITY_ORDER[capability.maturity]
        < _MATURITY_ORDER[parsed.validation_policy.minimum_maturity]
    ):
        errors.append(
            f"capability maturity {capability.maturity} is below required minimum "
            f"{parsed.validation_policy.minimum_maturity}"
        )

    execution_plan = [
        "validate semantic IR and evidence authority",
        f"resolve capability {parsed.construction.capability}",
    ]
    next_tool = capability.recommended_tool if capability else None
    if capability and capability.id == "declarative_primitives_v1":
        execution_plan.extend(
            [
                "lower allowlisted semantic objects to declarative Blender spec v1",
                "run immutable local Blender compilation",
                "inspect modeling QA and GLB artifact",
                "render canonical views and require user signoff",
            ]
        )
    elif capability:
        specialized_tool = capability.recommended_tool or "when available"
        execution_plan.append(f"use specialized tool {specialized_tool}")

    if errors:
        state = "rejected"
    elif capability and not capability.implemented:
        state = "planned_not_executable"
    elif capability and capability.preview_only:
        state = "preview_only"
    elif capability and capability.id == "declarative_primitives_v1":
        state = "ready_to_compile"
    else:
        state = "valid_for_specialized_tool"

    report = ViewForgeIRValidation(
        valid=not errors,
        ir_sha256=normalized_digest,
        normalized_ir=normalized,
        capability=capability,
        errors=errors,
        warnings=warnings,
        execution_plan=execution_plan,
        required_postconditions=required_gates,
        acceptance_state=state,
        next_tool=next_tool,
    )
    return parsed, report


def lower_procedural_ir(ir: ViewForgeAssetIR) -> DeclarativeModelSpec:
    if ir.construction.strategy != "procedural":
        raise ValueError("Only procedural IR can be lowered by the deterministic compiler.")
    if ir.construction.capability != "declarative_primitives_v1":
        raise ValueError("The deterministic compiler requires declarative_primitives_v1.")
    if not ir.construction.objects:
        raise ValueError("Procedural IR requires objects.")
    objects = [
        DeclarativeModelObject(
            name=object_.name,
            primitive=object_.primitive,
            parameters=object_.parameters,
            location=object_.location,
            rotationDegrees=object_.rotation_degrees,
            scale=object_.scale,
            material=object_.material,
            bevel=object_.bevel,
            shadeSmooth=object_.shade_smooth,
        )
        for object_ in ir.construction.objects
    ]
    return DeclarativeModelSpec(
        name=ir.name,
        units="meters",
        world=DeclarativeWorld(),
        objects=objects,
    )
