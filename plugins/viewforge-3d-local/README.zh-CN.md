# ViewForge 3D Local

这是仅供 Codex 本地使用的 ViewForge 3D 版本。它与 `viewforge-3d-toolkit` 并行安装，不会覆盖
ChatGPT/Tunnel 版本。

启动器固定使用仓库内锁定的 Python 3.11 `.venv`，任务写入独立的
`ViewForge 3D Local Plugin` 状态目录，并提供下一代本地架构专属工具：

- `list_viewforge_capabilities`：区分 trusted、validated、experimental 和 planned 路线。
- `validate_viewforge_ir`：检查语义意图、证据权限、约束和验收门禁。
- `compile_viewforge_ir`：只编译允许列表内的 procedural IR，并保留来源记录。
- `smooth_model_surface`：生成新的拓扑保持平滑 Blend/GLB 与 QA 报告。
- `get_local_artifact_path`：把产物的真实本地路径交给 Codex 或桌面工具。

编译器明确拒绝 raw mesh 数组和任意代码。六视图 visual hull 仍是 experimental、
`previewOnly` 路线；参数化模板拟合会显示在能力注册表中，但在本地实现完成前不会伪造结果。

先在仓库中绑定本地运行环境，再安装插件：

```bash
plugins/viewforge-3d-local/scripts/setup-local-runtime.sh
codex plugin add viewforge-3d-local@viewforge-3d
```

安装后请新建 Codex 任务，以加载本地版工具定义。
