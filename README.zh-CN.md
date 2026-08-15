# Face3D Modeling Toolkit

[English](README.md) | [简体中文](README.zh-CN.md)

Face3D Modeling Toolkit 是一个 Codex 插件和 Python 工作区，用于可追踪、带质量门槛的多视图
二维到三维重建。它支持人脸和物体的连续模板拟合、局部几何精修、附着特征工作流、固定视角
质量检查，以及不移动已验收几何的外观修改。

工具包采用失败关闭策略：如果生产模板缺失、证据不足、拓扑损坏或视觉候选被拒绝，不会静默
退回到看似合理的基础体或体素外壳。

## 仓库结构

- `plugins/face3d-modeling-toolkit/` — 可分发的 Codex 插件源码。
- `.agents/plugins/marketplace.json` — 仓库本地 Codex marketplace 清单。
- `LICENSE` — 适用于仓库代码和插件分发内容的 Apache License 2.0 条款。
- `src/face3d/` — Python 几何和验证实现。
- `scripts/` — 确定性构建和审计工具。
- `tests/` — Python 测试。
- `viewer/` — 本地模型与标注查看器。
- `docs/GUIDE.zh-CN.md` — 详细安装和技能调用指南。
- `docs/VIRTUAL_ENVIRONMENT.zh-CN.md` — 隔离 Python 环境配置指南。
- 英文文档分别位于 `README.md`、`docs/GUIDE.md` 和 `docs/VIRTUAL_ENVIRONMENT.md`。

## 从这里开始

1. 按照[虚拟环境指南](docs/VIRTUAL_ENVIRONMENT.zh-CN.md)创建隔离环境。
2. 按照[详细指南](docs/GUIDE.zh-CN.md)安装插件并调用技能。
3. 不确定应使用哪条几何路线时，先调用路由技能：

```text
$face3d-modeling-toolkit:face3d-toolkit-router
```

## 边界

- 当任务要求从二维证据重建时，不导入第三方成品三维模型。
- 除非明确声明为独立预览路线，否则 SDF、体素、点云和 Marching Cubes 仅用于质量检查。
- 自动几何门槛通过与用户视觉验收是两个独立状态。
- 私有输入图片和受限工程图不会进入插件包。

可分发插件包在本地生成到 `dist/`。生成模型、证据、虚拟环境、缓存和其他运行产物均不会进入
Git。GitHub Release 可以单独附带对比图片，但它不是插件包的一部分。

标注桥接器与来源无关。使用前必须显式配置锁定输入：

```bash
export FACE3D_ANNOTATION_SOURCE_MODEL=/absolute/path/to/source.glb
export FACE3D_ANNOTATION_SOURCE_SHA256=<64-character-sha256>
export FACE3D_ANNOTATION_SOURCE_VERSION=source-v1
export FACE3D_ANNOTATION_SUBJECT_PROFILE=generic-object
npm --prefix viewer run annotate
```

项目专用实验脚本和生成的标注元数据不会进入公开仓库内容。

## 许可证

仓库代码、文档和可分发插件采用 [Apache License 2.0](LICENSE)。第三方参考图片和商标仍归
各自权利人所有，本项目许可证不会对它们重新授权。
