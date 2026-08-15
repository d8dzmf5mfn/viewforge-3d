from __future__ import annotations

from pathlib import Path
from typing import Any

from face3d.config import Face3DConfig
from face3d.errors import fail
from face3d.stages.hybrid_v2 import release_stage_memory
from face3d.stages.template_fit import run_template_fit
from face3d.stages.template_qa import run_template_qa
from face3d.stages.template_skin import run_template_skin


def run_template_v3(run_dir: Path, config: Face3DConfig) -> dict[str, Any]:
    if not config.is_v3:
        fail(
            "config-invalid",
            "TemplateHeadV0 完整管线只接受 face-v3 配置",
            stage="face-v3",
        )
    fit = run_template_fit(run_dir, config)
    release_stage_memory()
    skin = run_template_skin(run_dir, config)
    release_stage_memory()
    qa = run_template_qa(run_dir, config)
    release_stage_memory()
    return {
        "mode": "template-head-v0",
        "geometryHash": fit["fittedGeometrySha256"],
        "triangles": fit["triangleCount"],
        "skinObservedVertexFraction": skin["observedVertexFraction"],
        "selfIntersectionPairCount": qa["geometry"]["selfIntersectionPairCount"],
        "eyeIntersectionCount": qa["eyes"]["intersectionCount"],
        "sdfRole": qa["sdf"]["role"],
        "surfaceGeneratedBySdf": qa["sdf"]["surfaceGenerated"],
    }
