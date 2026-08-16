# 详细指南

[English](GUIDE.md) | [简体中文](GUIDE.zh-CN.md)

## 目录

1. 安装仓库插件
2. 调用路由和专项技能
3. 选择工作流
4. 运行无贴皮物体路线
5. 验证并打包插件
6. 理解输出状态

## 1. 安装仓库插件

如果要运行仓库内的几何脚本，请先创建 Python 环境，参见
[`VIRTUAL_ENVIRONMENT.zh-CN.md`](VIRTUAL_ENVIRONMENT.zh-CN.md)。

要让 Codex 暴露 App 内置的本地 MCP 工具，先构建并安装 ViewForge Local：

```bash
./local-app/scripts/build_app.sh release
./local-app/scripts/install_app.sh
```

安装脚本会把 App 放到 `~/Applications/ViewForge Local.app`。这是必要步骤，因为已安装插件的
启动器从 Codex 缓存目录运行，无法依赖仓库内的 `dist` 相对路径。App、Tunnel 和 API Key 的完整
设置参见[私有连接指南](../local-app/USER_SETUP.zh-CN.md)。

在克隆仓库的根目录注册本地 marketplace 并安装插件：

```bash
codex plugin marketplace add "$(pwd)"
codex plugin add viewforge-3d-toolkit@viewforge-3d
```

安装后新建一个 Codex 任务，使技能清单重新加载。更新克隆仓库时，先拉取代码，再安装同一个插件
条目，然后重新新建任务。

## 2. 调用路由和专项技能

需要端到端处理，或不确定应选择哪条路线时，调用路由技能：

```text
使用 $viewforge-3d-toolkit:viewforge-3d-router 为这个模型选择安全路线。
```

已经明确处理阶段时，可以直接调用专项技能：

```text
使用 $viewforge-3d-toolkit:reconstruct-3d-from-multiview 根据这些二维图片构建无贴皮物体预览。

使用 $viewforge-3d-toolkit:render-model-preview 为这个已注册的 Blend 或 GLB 渲染固定视角 PNG，并把预览接触表返回对话。

使用 $viewforge-3d-toolkit:build-biological-skeleton 为这个人物或动物建立纯骨架 Armature，不添加权重。

使用 $viewforge-3d-toolkit:animate-biological-skeleton 根据带完整骨架的关键姿势图驱动已验收 Armature，并在无 skin 权重条件下刚性绑定分段肢体。

使用 $viewforge-3d-toolkit:landmark-guided-refinement 拟合已验收的人脸关键点，不改变不可见深度。

使用 $viewforge-3d-toolkit:annotation-region-lowering 将标注区域向内压低，不进行平滑。

使用 $viewforge-3d-toolkit:topology-preserving-smooth 仅平顺已批准区域，并保持拓扑和 UV。

使用 $viewforge-3d-toolkit:blender-manual-polish 指导一次有边界的 Blender 手工修正。

使用 $viewforge-3d-toolkit:same-geometry-skin 修改外观，不移动已验收几何。
```

在 Codex App 中也可以提及插件后直接用自然语言描述任务。完整限定技能名最不容易产生歧义。

## 3. 选择工作流

### 新建重建或来源审计

从 `reconstruct-3d-from-multiview` 开始。清点证据、声明主体配置、选择连续模板或预览路线，并在
拟合前锁定不可变来源。

### 已验收几何需要局部修改

根据实际操作选择关键点精修、标注压低、局部平滑或手工精修。不要把局部修改重新归类为一次
新重建。

### 仅修改外观

只有几何已验收后才能使用 `same-geometry-skin`。外观阶段前后的几何哈希和 UV 哈希必须完全
一致。

### 不添加权重的生物骨架

使用 `build-biological-skeleton`。先生成正面和侧面骨骼标注图，但只把它们视为视觉假设。生产
关节坐标必须来自有语义名称的分段组件，或经过审阅的三维关键点。输出不得包含权重、Armature
modifier、网格父级关系或动画。人物和人形角色使用 `humanoid-v1`；四足动物使用
`quadruped-v1` 并提供显式三维关键点。

### 不使用 skin 的生物动画

骨架位置验收后使用 `animate-biological-skeleton`。基于真实模型的固定正面渲染生成每个关键姿势
和结束姿势，并要求每张图都显示完整骨架。把经过审阅的关节点像素转换为相对根关节的 X/Z
方向，保留 Blender 静止骨长，只记录骨骼四元数旋转，并使用自动限幅 Bezier 插值补齐中间帧。

对于全身移动或道具交互，PoseBone Action 仍然只能记录旋转。角色的整体位移和转向放在父级
Empty 上，每个道具轨迹放在独立 Empty 上。必须在重新打开的 Blend 文件中检查接触距离、道具
位移、根节点转向连续性和最终支撑对位。

如果模型由独立刚性肢体组件组成，为每个组件建立完整骨骼映射，再用 Bone Parent 让模型在没有
顶点组和 Armature modifier 的条件下跟随 Action。连续网格无法用这种方式弯曲；在禁止 skin
权重时，只返回纯骨架动画。分别重新打开审计 Action 和绑定，最后把模型层与完整骨架层合成为
固定正面接触表，等待用户验收。

### 无贴皮物体

使用重建技能及物体模板参考。完成几何质量检查后停止，仅保留中性色因子，并确认导出的 GLB
不包含图片纹理。

## 4. 运行无贴皮物体路线

iPhone 17 示例要求两张被接纳的 Apple 参考图片，且最短边至少为 1024 像素。必须显式提供来源
图片，不得下载或导入现成三维模型。

```bash
source .venv/bin/activate

python scripts/build_iphone17_unskinned.py \
  --hero /absolute/path/to/official-hero.jpg \
  --side /absolute/path/to/official-side.jpg \
  --output dist/iphone17-unskinned-v1
```

输出目录不可覆盖。证据、代码、模板、尺寸、阈值或材质发生变化时，必须使用新的带版本目录。

必需的输出证据包括：

- `models/iphone17-unskinned.glb`；
- `manifest.json`，其中 `external3dImported=false` 且 `skinApplied=false`；
- `evidence/ledger.json`，记录来源 URL、哈希、分辨率门槛和证据权限；
- `qa/geometry-quality.json`，记录拓扑、自相交、GLB 往返和纹理检查；
- 固定正面、斜面、侧面、背面、顶部和底部渲染；
- 覆盖整个不可变运行目录的 `checksums.json`。

未公开的圆角外壳参数和特征轮廓只能视为有边界的二维拟合，不能声称是权威 CAD 测量。固定视角
尚未审阅前，必须保持 `userSignoff=false`。

## 5. 验证并打包插件

先验证每个技能，再验证插件清单：

```bash
python /path/to/skill-creator/scripts/quick_validate.py \
  plugins/viewforge-3d-toolkit/skills/reconstruct-3d-from-multiview

python /path/to/plugin-creator/scripts/validate_plugin.py \
  plugins/viewforge-3d-toolkit
```

生成确定性 ZIP：

```bash
source .venv/bin/activate
python scripts/package_viewforge_plugin.py \
  --plugin plugins/viewforge-3d-toolkit \
  --repository-root . \
  --output dist/viewforge-3d-toolkit-0.5.0.zip
```

ZIP 包含 Apache-2.0 许可证、插件清单、全部技能，以及相互独立的中英文 README、详细指南和
虚拟环境指南。缓存、生成模型、来源证据、对比图片和虚拟环境不会进入 ZIP。

## 6. 理解输出状态

- `preview` — 推断或探索几何，不能用于生产级身份声明。
- `geometry-preview` — 已生成几何并完成自动检查，但仍待视觉验收。
- `automated-gates-passed` — 声明的自动门槛已通过，用户确认仍是独立步骤。
- `user-accepted` — 用户明确接受了这个不可变候选。
- `rejected` — 不得作为下一版本的基线。

不能仅凭截图、命令成功退出或包文件存在，就把一种状态提升成另一种状态。
