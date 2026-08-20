# ViewForge 3D Local

这是仅供 Codex 本地使用的 ViewForge 3D 版本。它与 `viewforge-3d-toolkit` 并行安装，不会覆盖
ChatGPT/Tunnel 版本。

启动器固定使用仓库内锁定的 Python 3.11 `.venv`，任务写入独立的
`ViewForge 3D Local Plugin` 状态目录，并额外提供两个本地工具：

- `smooth_model_surface`：生成新的拓扑保持平滑 Blend/GLB 与 QA 报告。
- `get_local_artifact_path`：把产物的真实本地路径交给 Codex 或桌面工具。

先在仓库中绑定本地运行环境，再安装插件：

```bash
plugins/viewforge-3d-local/scripts/setup-local-runtime.sh
codex plugin add viewforge-3d-local@viewforge-3d
```

安装后请新建 Codex 任务，以加载本地版工具定义。
