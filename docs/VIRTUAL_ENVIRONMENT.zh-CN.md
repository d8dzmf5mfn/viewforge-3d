# Python 虚拟环境

[English](VIRTUAL_ENVIRONMENT.md) | [简体中文](VIRTUAL_ENVIRONMENT.zh-CN.md)

几何工具要求 Python 3.11。环境应放在仓库内的 `.venv/`；该目录被 Git 忽略，也不得进入插件包。

## 克隆仓库：使用 uv

按照 `uv` 官方说明安装后，在仓库根目录运行：

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
python -c "import cv2, numpy, open3d, scipy, trimesh; print('environment ready')"
```

自动化脚本应显式使用项目解释器：

```bash
.venv/bin/python scripts/build_iphone17_unskinned.py --help
.venv/bin/pytest tests/test_phone_v1.py
```

## Release ZIP：使用 uv

Release ZIP 包含插件及辅助脚本的最小依赖，但不包含完整 `face3d` Python 工作区。解压后进入顶层
目录并运行：

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
