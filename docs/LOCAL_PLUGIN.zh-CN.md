# ViewForge 3D 本地插件

[English](LOCAL_PLUGIN.md) | [简体中文](LOCAL_PLUGIN.zh-CN.md)

`viewforge-3d-local` 是独立的 Codex 本地插件，可与 `viewforge-3d-toolkit` 并行安装，互不覆盖。

| 边界 | ChatGPT/Tunnel 版本 | Codex 本地版本 |
| --- | --- | --- |
| 插件 ID | `viewforge-3d-toolkit` | `viewforge-3d-local` |
| 传输 | 可使用 App/Tunnel | 仅本地 stdio |
| Python | App 内置运行时及兼容回退 | 只使用仓库 `.venv/bin/python` |
| 状态目录 | `ViewForge Local` App 状态 | `ViewForge 3D Local Plugin` 独立状态 |
| 额外工具 | 无 | 语义 IR、能力成熟度、确定性编译、平滑、本地路径 |

本地版不会配置或使用 Tunnel、远程 URL、控制面密钥或 OpenAI API Key。设置脚本只验证现有
Python 3.11 `.venv`，并由插件自己把仓库根目录写进私有定位文件：

```text
~/Library/Application Support/ViewForge 3D Local Plugin/repository-root
```

启动器会自动读取该文件。日常使用时，用户不需要记住、填写或粘贴仓库路径。任务和资产索引
独立保存在：

```text
~/Library/Application Support/ViewForge 3D Local Plugin/state
```

## 安装与刷新

由维护者或设置代理在源码仓库中执行：

```bash
plugins/viewforge-3d-local/scripts/setup-local-runtime.sh
codex plugin add viewforge-3d-local@viewforge-3d
```

安装或更新 cachebuster 后，应新建 Codex 任务加载新工具；用户不需要配置 Tunnel。

## 下一代本地架构：先 IR，后 GLB

本地版现在把 GLB 当作经过验证的输出容器，而不是 AI 的推理语言。第一个下一代里程碑加入：

- `list_viewforge_capabilities`：报告能力成熟度、当前可用性和 preview-only 边界。
- `validate_viewforge_ir`：检查 schema version 1 的语义意图、证据权限、约束、强制验收门禁和
  选定能力。
- `compile_viewforge_ir`：只把 `declarative_primitives_v1` IR 降低为现有的确定性 Blender
  编译任务，并在 Blend、GLB、建模 QA 旁保存 IR 来源记录。

raw vertices/faces、buffer、byte offset、Python 和 Blender 脚本不属于这条 IR 路线。六视图
visual hull 继续保持 experimental、preview-only。`parametric_template_fit_v1` 只登记为 planned，
运行时会关闭失败，而不会把 visual hull 假装成标准人体或人脸模板。

编译成功后的状态是 `needs_model_verification`；固定视角渲染和用户确认仍是强制门禁。

六视图模板拟合的分阶段设计和验收状态详见
[NEXT_LOCAL_ARCHITECTURE.zh-CN.md](NEXT_LOCAL_ARCHITECTURE.zh-CN.md)。

## 表面平滑/打磨

`smooth_model_surface` 接受工作区内的 `.blend`/`.glb` 路径或已有 asset/artifact ID，创建不可变
新任务并输出：

- `smoothed-model.blend`
- `smoothed-model.glb`
- `smoothing-qa.json`

作业会检查拓扑、UV、材质、变换、受保护顶点和位移预算。体积保持与拓扑边界锁默认开启。
GLB 导入可能把 UV/法线接缝表现为拓扑边界，因此优先使用源 Blend；只有在明确审查边界后才关闭
边界锁。固定视角渲染通过人工检查前，结果保持 `pendingUserSignoff`。
