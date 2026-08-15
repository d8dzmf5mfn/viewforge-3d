from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from face3d import __version__
from face3d.assets import asset_status, prepare_face_v2, record_assets
from face3d.config import Face3DConfig, load_config
from face3d.errors import Face3DError, error_json

app = typer.Typer(
    name="viewforge3d",
    no_args_is_help=True,
    invoke_without_command=True,
    pretty_exceptions_enable=False,
    help="ViewForge 3D 多视图模型构建与验证工具",
)
assets_app = typer.Typer(no_args_is_help=True, help="本地模型资产")
app.add_typer(assets_app, name="assets")
FACE_V3_CONFIG = Path("configs/face-v3.yaml")
FACE_V2_CONFIG = Path("configs/face-v2.yaml")
DEFAULT_CONFIG = FACE_V3_CONFIG


def _emit(value: object) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _config(path: Path) -> Face3DConfig:
    return load_config(path)


@app.callback()
def root(
    version: Annotated[
        bool,
        typer.Option("--version", help="显示版本", is_eager=True),
    ] = False,
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command("validate-input")
def validate_input(
    input: Annotated[Path, typer.Option("--input", exists=True, file_okay=False)],
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    from face3d.stages.intake import validate_only

    _emit(validate_only(input, _config(config), output))


@app.command("confirm-masks")
def confirm_masks(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
) -> None:
    from face3d.stages.intake import confirm_masks as confirm_masks_stage

    _emit(confirm_masks_stage(run))


@app.command("fit-template-head-v0")
def fit_template_head_v0(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    config: Annotated[Path, typer.Option("--config")] = FACE_V3_CONFIG,
) -> None:
    """Fit the continuous TemplateHeadV0 asset to an admitted three-view run."""
    from face3d.assets import require_assets
    from face3d.stages.template_fit import run_template_fit

    loaded = _config(config)
    require_assets(loaded)
    _emit(run_template_fit(run, loaded))


@app.command("project-template-head-v0-skin")
def project_template_head_v0_skin(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    config: Annotated[Path, typer.Option("--config")] = FACE_V3_CONFIG,
) -> None:
    """Project the reviewed three views onto the accepted fitted topology."""
    from face3d.assets import require_assets
    from face3d.stages.template_skin import run_template_skin

    loaded = _config(config)
    require_assets(loaded)
    _emit(run_template_skin(run, loaded))


@app.command("qa-template-head-v0")
def qa_template_head_v0(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    config: Annotated[Path, typer.Option("--config")] = FACE_V3_CONFIG,
) -> None:
    """Run topology, eye-contact, and QA-only signed-distance gates."""
    from face3d.assets import require_assets
    from face3d.stages.template_qa import run_template_qa

    loaded = _config(config)
    require_assets(loaded)
    _emit(run_template_qa(run, loaded))


@app.command()
def reconstruct(
    input: Annotated[Path, typer.Option("--input", exists=True, file_okay=False)],
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
    output: Annotated[Path, typer.Option("--output")] = Path("runs/face-001"),
) -> None:
    from face3d.pipeline import reconstruct as reconstruct_pipeline

    _emit(reconstruct_pipeline(input, output, _config(config)))


@app.command("package")
def package_command(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
) -> None:
    from face3d.package import package_run

    _emit(package_run(run, output, _config(config)))


@app.command("generate-synthetic")
def generate_synthetic(
    output: Annotated[Path, typer.Option("--output")],
    count: Annotated[int, typer.Option("--count", min=3, max=20)] = 3,
    config: Annotated[Path, typer.Option("--config")] = FACE_V2_CONFIG,
) -> None:
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        from face3d.errors import fail

        fail(
            "optional-dependency-missing",
            "FLAME 合成基准需要先执行: uv sync --extra legacy-sdf",
            stage="assets",
        )
    from face3d.synthetic import generate_synthetic_dataset

    _emit(generate_synthetic_dataset(output, _config(config), count=count))


@app.command("generate-pixel-cube")
def generate_pixel_cube(
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    side_cm: Annotated[float, typer.Option("--side-cm", min=0.1)] = 20.0,
    cells_per_edge: Annotated[
        int,
        typer.Option("--cells-per-edge", min=2, max=512),
    ] = 128,
) -> None:
    from face3d.pixel_cube import PixelCubeSpec, create_pixel_cube

    _emit(
        create_pixel_cube(
            output,
            PixelCubeSpec(
                side_length_m=side_cm / 100,
                cells_per_edge=cells_per_edge,
            ),
        )
    )


@app.command("generate-cube-front-relief")
def generate_cube_front_relief(
    input_image: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    face_landmarker_model: Annotated[
        Path,
        typer.Option("--face-landmarker", exists=True, dir_okay=False),
    ] = Path(".local/models/mediapipe/face_landmarker.task"),
    scan_config: Annotated[
        Path,
        typer.Option("--scan-config", exists=True, dir_okay=False),
    ] = Path("configs/cube-front-hires.yaml"),
    side_cm: Annotated[float, typer.Option("--side-cm", min=0.1)] = 24.0,
    cells_per_edge: Annotated[
        int,
        typer.Option("--cells-per-edge", min=4, max=512),
    ] = 128,
    max_inset_cm: Annotated[float, typer.Option("--max-inset-cm", min=0.01)] = 3.0,
) -> None:
    from face3d.pixel_cube import PixelCubeSpec
    from face3d.pixel_cube_relief import FrontFaceReliefSpec, create_front_face_relief

    scan_settings = _config(scan_config)
    _emit(
        create_front_face_relief(
            input_image,
            face_landmarker_model,
            output,
            FrontFaceReliefSpec(
                cube=PixelCubeSpec(
                    side_length_m=side_cm / 100,
                    cells_per_edge=cells_per_edge,
                ),
                front_cells_per_edge=scan_settings.pixel.grid_size,
                border_rim_cells=(scan_settings.pixel.grid_size / cells_per_edge),
                max_inset_m=max_inset_cm / 100,
                coarse_depth_grid=scan_settings.pixel.coarse_depth_grid,
                complex_region_radius_pixels=(scan_settings.pixel.complex_region_radius_pixels),
                depth_scale_face_width=scan_settings.pixel.depth_scale_face_width,
                maximum_cells=scan_settings.pixel.maximum_cells,
                scan_config_path=scan_config.resolve(),
            ),
            canonical_face_model=scan_settings.resolve_asset(
                scan_settings.assets.canonical_face_model
            ),
        )
    )


@assets_app.command("status")
def assets_status(
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
) -> None:
    _emit(asset_status(_config(config), require_recorded=True))


@assets_app.command("record")
def assets_record(
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
) -> None:
    _emit(record_assets(_config(config)))


@assets_app.command("prepare-face-v2")
def assets_prepare_face_v2(
    config: Annotated[Path, typer.Option("--config")] = FACE_V2_CONFIG,
) -> None:
    """Validate licensed FLAME inputs and build the locked v2 topology/UV asset."""
    _emit(prepare_face_v2(_config(config)))


@assets_app.command("prepare-template-head-v0")
def assets_prepare_template_head_v0(
    quality_baseline: Annotated[
        Path,
        typer.Option("--quality-baseline", exists=True, dir_okay=False),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", file_okay=False),
    ] = Path("assets/template-head-v0"),
    package: Annotated[
        Path | None,
        typer.Option("--package", exists=True, dir_okay=False),
    ] = None,
    source_glb: Annotated[
        Path | None,
        typer.Option("--source-glb", exists=True, dir_okay=False),
    ] = None,
    source_license: Annotated[
        Path | None,
        typer.Option("--source-license", exists=True, dir_okay=False),
    ] = None,
) -> None:
    """Extract the continuous head/neck template and lock its visual baseline."""
    from face3d.template_head_v0 import prepare_template_head_v0

    _emit(
        prepare_template_head_v0(
            package,
            quality_baseline,
            output,
            source_glb=source_glb,
            source_license=source_license,
        )
    )


@assets_app.command("prepare-template-head-v0-anatomy")
def assets_prepare_template_head_v0_anatomy(
    template_root: Annotated[
        Path,
        typer.Option("--template-root", exists=True, file_okay=False),
    ] = Path("assets/template-head-v0"),
    face_landmarker_model: Annotated[
        Path,
        typer.Option("--face-landmarker", exists=True, dir_okay=False),
    ] = Path(".local/models/mediapipe/face_landmarker.task"),
) -> None:
    """Add open eyelid rings, complete eyeballs, UV, and semantics to TemplateHeadV0."""
    from face3d.template_head_anatomy import prepare_template_head_v0_anatomy

    _emit(
        prepare_template_head_v0_anatomy(
            template_root,
            face_landmarker_model,
        )
    )


@assets_app.command("rebind-template-head-v0-eyelids")
def assets_rebind_template_head_v0_eyelids(
    template_root: Annotated[
        Path,
        typer.Option("--template-root", exists=True, file_okay=False),
    ] = Path("assets/template-head-v0"),
) -> None:
    """Rebind both full eyelid contours without modifying geometry or UV."""
    from face3d.template_head_anatomy import rebind_template_head_v0_eyelids

    _emit(rebind_template_head_v0_eyelids(template_root))


def main() -> None:
    try:
        app()
    except Face3DError as exc:
        typer.echo(error_json(exc), err=True)
        raise SystemExit(exc.exit_code) from None
    except KeyboardInterrupt:
        typer.echo(
            json.dumps(
                {"ok": False, "error": {"code": "interrupted", "stage": "runtime"}},
                ensure_ascii=False,
            ),
            err=True,
        )
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
