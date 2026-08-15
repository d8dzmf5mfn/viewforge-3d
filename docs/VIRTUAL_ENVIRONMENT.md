# Python virtual environment

[English](VIRTUAL_ENVIRONMENT.md) | [简体中文](VIRTUAL_ENVIRONMENT.zh-CN.md)

The geometry tools require Python 3.11. Keep the environment inside the repository as `.venv/`;
it is ignored by Git and must not be included in plugin packages.

The complete repository environment contains the ViewForge CLI, NumPy, SciPy, OpenCV, Pillow,
Trimesh, and Open3D. The current repository supplies the ViewForge CLI, the Python package name for
OpenCV is `opencv-python-headless`, and Pillow is imported as `PIL`. The setup commands below
install these components together; do not install them one by one.

## Cloned repository: uv

Install `uv` using its official instructions, then run from the repository root. These commands
are for first-time environment creation: `uv sync --frozen` installs the current ViewForge 3D
project, creates the `viewforge3d` command, and installs every dependency exactly as locked in
`uv.lock`:

```bash
uv python install 3.11
UV_CACHE_DIR=.uv-cache uv venv --python 3.11 .venv
UV_CACHE_DIR=.uv-cache uv sync --frozen
source .venv/bin/activate
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

Verify the interpreter and core geometry dependencies:

```bash
python --version
viewforge3d --version
python -c "import cv2, numpy, open3d, PIL, scipy, trimesh; print('ViewForge environment ready')"
```

The complete environment is ready only when all three commands succeed. After that verification,
modeling tasks must reuse the existing `.venv` and plugin scripts without running `uv sync`,
`uv pip install`, `pip install`, or any other package-install command.

Use the project interpreter explicitly in automation:

```bash
.venv/bin/python scripts/build_iphone17_unskinned.py --help
.venv/bin/pytest tests/test_phone_v1.py
```

## Release ZIP: uv

The Release ZIP contains the plugin and its minimum helper-script dependencies. It does not contain
the ViewForge CLI, the complete Python workspace, or the full geometry dependency set listed
above. Use the cloned-repository setup when the complete environment is required; do not treat this
section as a full environment installation.

If only the generic helper scripts bundled in the ZIP are needed, extract it, enter its top-level
directory, and run:

```bash
uv python install 3.11
UV_CACHE_DIR=.uv-cache uv venv --python 3.11 .venv
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

On Windows PowerShell, replace `.venv/bin/python` with `.venv\Scripts\python.exe` and activate
with `.venv\Scripts\Activate.ps1`.

## Standard library venv

If `uv` is unavailable, create the environment with a local Python 3.11 installation:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install "pytest>=8.3,<9" "pytest-cov>=6,<7" "ruff>=0.9,<1"
```

The `python -m pip install -e .` command installs the ViewForge CLI and all runtime dependencies.
After it completes, run the `viewforge3d --version` and six-library import checks shown above; do
not reinstall those libraries individually.

For an extracted Release ZIP, install its minimum dependencies instead of the repository package:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell equivalent:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install "pytest>=8.3,<9" "pytest-cov>=6,<7" "ruff>=0.9,<1"
```

## Recreate a broken environment

Do not repair a partially installed environment in place. Move the old `.venv` outside the
repository or remove it only after confirming the exact path, then create a fresh `.venv` and run
the locked install again.

If `uv sync --frozen` fails, preserve the error. Do not delete `uv.lock` or silently perform an
unlocked dependency upgrade. Network, architecture, Python-version, and cache-permission failures
must be diagnosed separately.

## Scope

The Codex skill instructions themselves do not require this Python environment to appear in the
skill list. The environment is required when a skill invokes the bundled geometry, validation,
comparison, or packaging scripts in this repository.

`bpy` and `bmesh` are provided by Blender's Python runtime and are intentionally not pip
dependencies. Run the exact-union helper through Blender as documented by the reconstruction
skill.
