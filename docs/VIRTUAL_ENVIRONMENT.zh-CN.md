# Python 虚拟环境

[English](VIRTUAL_ENVIRONMENT.md) | [简体中文](VIRTUAL_ENVIRONMENT.zh-CN.md)

几何工具要求 Python 3.11。环境应放在仓库内的 `.venv/`；该目录被 Git 忽略，也不得进入插件包。

完整仓库环境包含 ViewForge CLI、NumPy、SciPy、OpenCV、Pillow、Trimesh 和 Open3D。
ViewForge CLI 由当前仓库安装，OpenCV 的 Python 包名是 `opencv-python-headless`，Pillow 的
导入名是 `PIL`。下列安装命令会一次性安装这些组件，不需要逐个安装。

## 克隆仓库：使用 uv

按照 `uv` 官方说明安装后，在仓库根目录运行。以下是首次创建环境的命令；
`uv sync --frozen` 会安装当前 ViewForge 3D 项目、生成 `viewforge3d` 命令，并严格按照
`uv.lock` 安装全部锁定依赖：

```bash
uv python install 3.11
UV_CACHE_DIR=.uv-cache uv venv --python 3.11 .venv
UV_CACHE_DIR=.uv-cache uv sync --frozen
source .venv/bin/activate
```

Windows PowerShell 激活命令：

```powershell
.venv\Scripts\Activate.ps1
```

检查解释器和核心几何依赖：

```bash
python --version
viewforge3d --version
python -c "import cv2, numpy, open3d, PIL, scipy, trimesh; print('ViewForge environment ready')"
```

只有三个命令全部成功，才表示完整环境可用。环境确认后，后续建模任务应直接使用现有 `.venv`
和插件脚本，不再运行 `uv sync`、`uv pip install`、`pip install` 或其他包安装命令。

自动化脚本应显式使用项目解释器：

```bash
.venv/bin/python scripts/build_iphone17_unskinned.py --help
.venv/bin/pytest tests/test_phone_v1.py
```

## Release ZIP：使用 uv

Release ZIP 包含插件及辅助脚本的最小依赖，但不包含 ViewForge CLI、完整 Python 工作区或上述
完整几何依赖集合。需要完整环境时，必须使用克隆仓库的安装方法；不要把本节当作完整环境安装。

如果只需要运行 ZIP 内附带的通用插件辅助脚本，解压后进入顶层目录并运行：

```bash
uv python install 3.11
UV_CACHE_DIR=.uv-cache uv venv --python 3.11 .venv
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

Windows PowerShell 中，把 `.venv/bin/python` 替换为 `.venv\Scripts\python.exe`，再使用前面的
PowerShell 激活命令。

## 使用标准库 venv

如果无法使用 `uv`，可使用本机 Python 3.11 创建环境：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install "pytest>=8.3,<9" "pytest-cov>=6,<7" "ruff>=0.9,<1"
```

其中 `python -m pip install -e .` 会安装 ViewForge CLI 和全部运行时依赖。安装完成后，运行前文
列出的 `viewforge3d --version` 和六个 Python 库导入检查；不要再逐个重复安装这些库。

对于解压后的 Release ZIP，安装最小依赖而不是仓库包：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell 等价命令：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install "pytest>=8.3,<9" "pytest-cov>=6,<7" "ruff>=0.9,<1"
```

## 重建损坏的环境

不要原地修补安装不完整的环境。先确认 `.venv` 的准确路径，再把旧环境移出仓库，或在确认后
删除，然后重新创建 `.venv` 并执行锁定安装。

如果 `uv sync --frozen` 失败，应保留错误信息。不要删除 `uv.lock`，也不要静默执行未锁定的依赖
升级。网络、架构、Python 版本和缓存权限错误必须分别诊断。

## 适用范围

Codex 技能指令本身不要求这个 Python 环境出现在技能列表中。只有技能调用本仓库的几何、验证、
对比或打包脚本时才需要该环境。

`bpy` 和 `bmesh` 由 Blender 的 Python 运行时提供，因此不会作为 pip 依赖。精确布尔并集辅助脚本
应按照重建技能文档通过 Blender 运行。
