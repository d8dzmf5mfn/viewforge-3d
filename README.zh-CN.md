# ViewForge 3D

[English](README.md) | [简体中文](README.zh-CN.md)

ViewForge 3D 是一个 Codex 插件和 Python 工作区，用于根据多视图二维证据，以可追踪、带质量
门槛的方式构建三维模型。它适用于人物、风格化角色、产品和普通物体，支持连续模板拟合、局部
几何精修、附着特征工作流、固定视角质量检查，以及不移动已验收几何的外观修改。
现在也支持为人物、人形角色和动物建立、驱动可审计的纯骨架 Armature，并为分段模型提供不使用
权重的刚性 Bone Parent 绑定。

工具包采用失败关闭策略：如果生产模板缺失、证据不足、拓扑损坏或视觉候选被拒绝，不会静默
退回到看似合理的基础体或体素外壳。

## 仓库结构

- `plugins/viewforge-3d-toolkit/` — 可分发的 Codex 插件源码。
- `.agents/plugins/marketplace.json` — 仓库本地 Codex marketplace 清单。
- `LICENSE` — 适用于仓库代码和插件分发内容的 Apache License 2.0 条款。
- `src/face3d/` — Python 几何和验证实现。
- `scripts/` — 确定性构建和审计工具。
- `tests/` — Python 测试。
- `viewer/` — 本地模型与标注查看器。
- `local-app/` — 私有 macOS MCP 宿主及其构建、连接指南。
- `docs/GUIDE.zh-CN.md` — 详细安装和技能调用指南。
- `docs/VIRTUAL_ENVIRONMENT.zh-CN.md` — 隔离 Python 环境配置指南。
- 英文文档分别位于 `README.md`、`docs/GUIDE.md` 和 `docs/VIRTUAL_ENVIRONMENT.md`。

## 从这里开始

1. 按照[虚拟环境指南](docs/VIRTUAL_ENVIRONMENT.zh-CN.md)创建隔离环境。
2. 按照[详细指南](docs/GUIDE.zh-CN.md)安装插件并调用技能。
3. 需要把本地几何运行时连接到 ChatGPT Developer Mode 时，按照
   [ViewForge Local 安装、Tunnel 与 API Key 指南](local-app/USER_SETUP.zh-CN.md)操作。
4. 不确定应使用哪条几何路线时，先调用路由技能：

```text
$viewforge-3d-toolkit:viewforge-3d-router
```

当人物、人形角色或动物模型只需要骨架、不需要权重时：

```text
$viewforge-3d-toolkit:build-biological-skeleton
```

当已验收 Armature 需要根据带完整骨架的关键姿势图生成动画，并让分段肢体在无权重条件下随骨骼
移动时：

```text
$viewforge-3d-toolkit:animate-biological-skeleton
```

## 边界

- 当任务要求从二维证据重建时，不导入第三方成品三维模型。
- 除非明确声明为独立预览路线，否则 SDF、体素、点云和 Marching Cubes 仅用于质量检查。
- 自动几何门槛通过与用户视觉验收是两个独立状态。
- 静态纯骨架流程不会添加权重、Armature modifier、网格父级关系或动画。
- 纯骨架动画只记录旋转。无 skin 的模型运动只对独立分段组件使用刚性 Bone Parent；连续网格
  无法在无权重条件下弯曲，因此会失败关闭。
- 私有输入图片和受限工程图不会进入插件包。

可分发插件包在本地生成到 `dist/`。生成模型、证据、虚拟环境、缓存和其他运行产物均不会进入
Git。GitHub Release 可以单独附带对比图片，但它不是插件包的一部分。

标注桥接器与来源无关。使用前必须显式配置锁定输入：

```bash
export VIEWFORGE3D_ANNOTATION_SOURCE_MODEL=/absolute/path/to/source.glb
export VIEWFORGE3D_ANNOTATION_SOURCE_SHA256=<64-character-sha256>
export VIEWFORGE3D_ANNOTATION_SOURCE_VERSION=source-v1
export VIEWFORGE3D_ANNOTATION_SUBJECT_PROFILE=generic-object
npm --prefix viewer run annotate
```

项目专用实验脚本和生成的标注元数据不会进入公开仓库内容。

## 许可证

仓库代码、文档和可分发插件采用 [Apache License 2.0](LICENSE)。第三方参考图片和商标仍归
各自权利人所有，本项目许可证不会对它们重新授权。

插件目录包含内容完全一致的 `LICENSE` 副本，确保 Codex 独立安装后仍保留许可证条款。
