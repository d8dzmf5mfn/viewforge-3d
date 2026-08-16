# ViewForge Local：从本地 App 到 ChatGPT 的私有连接指南

本指南用于让每位用户在自己的 Mac 上构建并运行 ViewForge Local，再通过 OpenAI Secure MCP Tunnel 连接到自己的 ChatGPT Developer Mode。

这不是公开发布流程。Secure MCP Tunnel 用于私人连接和开发者测试，不会把本地 MCP 服务直接暴露到公网，也不能用于公开插件分发。

官方资料：

- [Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)
- [OpenAI Platform Tunnel 设置](https://platform.openai.com/settings/organization/tunnels)
- [ChatGPT 插件设置](https://chatgpt.com/#settings/Connectors)
- [tunnel-client 最新版本](https://github.com/openai/tunnel-client/releases/latest)

## 1. 连接结构

```text
ChatGPT Developer Mode
        │
        │ OpenAI 托管的 Tunnel 端点
        ▼
tunnel-client（用户的 Mac）
        │
        │ http://127.0.0.1:8765/mcp
        ▼
ViewForge Local.app
        │
        ├── 本机 Blender
        ├── 用户选择的模型工作区
        └── 本机任务与产物目录
```

必须同时保持以下两个进程运行：

1. `ViewForge Local.app`：提供本地 MCP 服务并调用 Blender。
2. `tunnel-client`：把 ChatGPT 的 MCP 请求转发到本地服务。

## 2. 前置条件

- macOS 14 或更高版本。
- Blender 安装在以下任一标准位置：
  - `/Applications/Blender.app`
  - `~/Applications/Blender.app`
- 用源码构建时需要：
  - Xcode Command Line Tools 或完整 Xcode
  - Swift 6
  - `uv`
  - `ripgrep`
  - 可访问 Python 包源的网络
- OpenAI Platform 中具备相应 Tunnel 权限：
  - 创建或修改 Tunnel：`Tunnels Read + Manage`
  - 运行 Tunnel、在 ChatGPT 中选择 Tunnel：`Tunnels Read + Use`
- ChatGPT 工作区允许并已启用 Developer Mode。

Platform Tunnel 权限与 ChatGPT Developer Mode 权限彼此独立。如果是 Enterprise 或 Edu 工作区，可能需要分别联系 Platform 组织管理员和 ChatGPT 工作区管理员。

## 3. 构建 ViewForge Local.app

### 3.1 获取源码

```bash
git clone https://github.com/d8dzmf5mfn/viewforge-3d.git
cd viewforge-3d
```

如果已经有源码，直接进入仓库根目录。

### 3.2 检查构建工具

```bash
swift --version
uv --version
rg --version
```

缺少 Xcode Command Line Tools 时，可以运行：

```bash
xcode-select --install
```

如果使用 Homebrew，可以安装其余工具：

```bash
brew install uv ripgrep
```

### 3.3 构建 App

在仓库根目录运行：

```bash
./local-app/scripts/build_app.sh release
```

构建结果：

```text
dist/viewforge-local/ViewForge Local.app
```

该构建会把 Python 3.11 运行时、ViewForge MCP 服务和 ViewForge 3D 工具包复制进 App，不依赖仓库的 `.venv`。Blender 不会被打包，仍需单独安装。

当前脚本生成的是本机测试用的临时签名 App。需要分发给其他用户时，应另行完成 Developer ID 签名和公证；不要要求用户绕过 macOS Gatekeeper。

### 3.4 启动 App

```bash
open "dist/viewforge-local/ViewForge Local.app"
```

首次启动后：

1. 点击“选择工作区”。
2. 选择一个专门保存模型、动作输入和输出的目录。
3. 点击“重新检测 Blender”。
4. 等待 MCP 状态显示为 `就绪 · 127.0.0.1:8765`。

本地健康检查：

```bash
curl -fsS http://127.0.0.1:8765/healthz
```

预期返回的 `status` 为 `ready`。如果显示 `needs-configuration`，重新检查工作区、Blender 和 App 内置插件运行时。

## 4. 安装 tunnel-client

不要在指南中固定某个旧版本。始终从 [最新 Release](https://github.com/openai/tunnel-client/releases/latest) 下载。

先确认 Mac 架构：

```bash
uname -m
```

下载对应文件：

| `uname -m` 输出 | Release 文件 |
|---|---|
| `arm64` | `darwin-arm64`，适用于 Apple Silicon |
| `x86_64` | `darwin-amd64`，适用于 Intel Mac |

解压后，把可执行文件安装到 ViewForge Local 能自动检测的位置：

```bash
mkdir -p "$HOME/.local/bin"
install -m 755 "/path/to/extracted/tunnel-client" "$HOME/.local/bin/tunnel-client"
```

把下面一行加入 `~/.zprofile`，然后重新打开 Terminal：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

验证：

```bash
tunnel-client --version
tunnel-client help quickstart
```

不要把整个下载目录或版本号写死到 App 配置中。ViewForge Local 会依次检查：

- `/opt/homebrew/bin/tunnel-client`
- `/usr/local/bin/tunnel-client`
- `~/.local/bin/tunnel-client`

## 5. 创建 OpenAI Tunnel 和运行密钥

### 5.1 创建 Tunnel

1. 打开 [OpenAI Platform Tunnel 设置](https://platform.openai.com/settings/organization/tunnels)。
2. 选择准备使用的 Platform 组织。
3. 创建一个 Tunnel。
4. 将目标 Platform 组织和目标 ChatGPT 工作区关联到该 Tunnel。
5. 记录生成的 Tunnel ID，格式类似 `tunnel_...`。

不要把真实 Tunnel ID 写进公开 README、截图或示例配置。

### 5.2 创建单独的运行密钥

在对应 Platform 组织或项目中创建一个专门供 `tunnel-client` 使用的 API Key。该密钥对应的主体需要 `Tunnels Read + Use`。

这里需要 API Key，是因为 `tunnel-client` 必须向 OpenAI Tunnel 控制平面证明它有权使用该 Tunnel。ViewForge 本地 MCP 不会使用这个密钥调用模型。不要使用 Admin API Key，也不要与其他应用共用长期密钥。

### 5.3 保存到本地工作区

在刚才选择的 ViewForge 工作区根目录新建 `.env.local`：

```dotenv
OPENAI_API_KEY=replace_with_your_runtime_api_key
```

然后把权限限制为仅当前用户可读写：

```bash
chmod 600 "/path/to/your/viewforge-workspace/.env.local"
```

注意：

- 不要把真实密钥直接写在聊天、终端命令参数、截图或 Markdown 中。
- `.env.local` 不应提交到 Git。
- 密钥保存在工作区中，不会被复制进 `ViewForge Local.app`。
- 当前 App 使用环境变量名 `OPENAI_API_KEY` 兼容 `tunnel-client`；这里存放的是专用 Tunnel 运行密钥。

## 6. 创建本地 Tunnel Profile

保持 ViewForge Local 正在运行，然后在 Terminal 中执行。只替换示例 Tunnel ID：

```bash
tunnel-client init \
  --sample sample_mcp_remote_no_auth \
  --profile viewforge-local \
  --tunnel-id tunnel_REPLACE_ME \
  --mcp-server-url http://127.0.0.1:8765/mcp \
  --control-plane-api-key-ref env:OPENAI_API_KEY \
  --health-listen-addr 127.0.0.1:8080
```

这个命令只把环境变量引用写进 Profile，不会把 API Key 本身写进 Profile。Profile 默认位于 `~/.config/tunnel-client`。

如果已经存在同名 Profile，先检查现有配置。只有明确要替换它时才使用 `--force`。

### 6.1 运行诊断

进入包含 `.env.local` 的工作区，再执行：

```bash
set -a
source .env.local
set +a
tunnel-client doctor --profile viewforge-local --explain
unset OPENAI_API_KEY
```

诊断应确认：

- 可以访问 OpenAI Tunnel 控制平面。
- Tunnel ID 存在且当前密钥有使用权限。
- 本地 MCP 地址 `http://127.0.0.1:8765/mcp` 可访问。

## 7. 启动私有连接

推荐从 ViewForge Local 界面启动：

1. 确认 Profile 为 `viewforge-local`。
2. 点击“启动”。
3. 等待“安全隧道”显示为“运行中”。

也可以在 Terminal 前台运行：

```bash
set -a
source .env.local
set +a
tunnel-client run --profile viewforge-local
```

前台运行时不要关闭这个 Terminal。停止时按 `Control-C`，然后执行：

```bash
unset OPENAI_API_KEY
```

默认情况下可以打开本机管理页面检查 Tunnel 状态：

```text
http://127.0.0.1:8080/ui
```

只有当管理页面和 `/readyz` 都显示就绪后，才能认为 Tunnel 已连接。不要把管理页面绑定到非本机地址。

## 8. 在 ChatGPT 中创建私人 App

保持 ViewForge Local 和 tunnel-client 都在运行。

1. 打开 [ChatGPT 插件设置](https://chatgpt.com/#settings/Connectors)。
2. 启用 Developer Mode；如果工作区禁止该功能，联系工作区管理员。
3. 点击加号，创建 Developer Mode App。
4. 名称填写 `ViewForge 3D`。
5. Connection 选择 `Tunnel`。
6. 从列表选择刚创建的 Tunnel，或粘贴自己的 Tunnel ID。
7. MCP 身份验证选择 `No Auth`。
8. 保存并等待 ChatGPT 完成工具发现。

`No Auth` 仅表示 ViewForge MCP 本身没有第二层 OAuth；Tunnel 到 OpenAI 控制平面的连接仍由运行密钥鉴权。

首次只做只读测试：

```text
检查 ViewForge 3D 本地状态，只调用 viewforge_status，不创建任务。
```

结果应显示：

- `ready: true`
- 工作区已配置
- Blender 可用
- 插件运行时可用

## 9. 使用模型时避免泄漏绝对路径

把模型放进所选工作区，例如：

```text
viewforge-workspace/
├── .env.local
└── models/
    └── character.glb
```

在 ChatGPT 中只提供工作区相对路径：

```text
注册 models/character.glb，然后为它创建 humanoid-v1 骨骼。
```

不要输入：

```text
/Users/某个用户名/.../character.glb
```

ViewForge Local 会在本机把相对路径解析到所选工作区，并拒绝访问工作区外的文件。注册完成后应继续使用 `asset_...`、`job_...` 和 `artifact_...` ID，而不是重复传递路径。

隐私边界：

- 模型文件由本地 MCP 和 Blender 读取；除非用户另行上传，工具不会把完整 GLB 或 Blend 文件作为结果发送给 ChatGPT。
- ChatGPT 仍会看到工具调用参数和工具结果，例如相对文件名、资产 ID、文件大小、哈希、任务状态及主动读取的 QA JSON。
- 本地状态、任务文件和日志位于 `~/Library/Application Support/ViewForge Local`，其中可能包含绝对路径。分享日志前必须人工检查并脱敏。
- 不要启用 `tunnel-client` 的不安全原始 HTTP 日志选项。

## 10. 常见问题

### App 显示“未安装 tunnel-client”

确认二进制存在且可执行：

```bash
ls -l "$HOME/.local/bin/tunnel-client"
"$HOME/.local/bin/tunnel-client" --version
```

### App 显示“缺少 API Key”

- 确认 `.env.local` 位于 App 当前选择的工作区根目录。
- 确认变量名是 `OPENAI_API_KEY`。
- 确认文件权限为 `600`。

```bash
stat -f '%Sp %N' "/path/to/your/viewforge-workspace/.env.local"
```

### MCP 一直没有就绪

```bash
curl -fsS http://127.0.0.1:8765/healthz
lsof -nP -iTCP:8765 -sTCP:LISTEN
```

然后在 App 中点击“打开本地记录”，检查 `mcp-server.log`。不要未经检查直接把完整日志发给第三方。

### Tunnel 不出现在 ChatGPT

- 确认 Tunnel 关联了目标 ChatGPT 工作区，而不只是 Platform 组织。
- 确认当前用户有 `Tunnels Read + Use`。
- 确认 ChatGPT Developer Mode 已开启。
- 保持 `tunnel-client run` 正在运行，再重新创建或刷新 App。

### Tunnel 启动后工具调用失败

```bash
tunnel-client doctor --profile viewforge-local --explain
```

同时确认：

- ViewForge Local 没有退出。
- `http://127.0.0.1:8765/healthz` 返回 `ready`。
- Tunnel 管理页的 `/readyz` 为就绪状态。

### 端口 8080 已被占用

重新创建 Profile 时，把 `--health-listen-addr` 改为另一个仅绑定回环地址的端口，例如：

```text
127.0.0.1:8081
```

不要使用 `0.0.0.0`，否则可能把 Tunnel 管理页面暴露给局域网。

## 11. 交给 Coding Agent 的本地 App 构建提示词

把下面整段内容复制给能够读取本地仓库、编辑文件并运行 Terminal 命令的 coding agent。Agent 应在 ViewForge 3D 仓库根目录开始工作。

```text
你正在 ViewForge 3D 仓库根目录工作。请在当前 Mac 上实现、构建并验证一个可运行的 `ViewForge Local.app`，让本机 Blender 和 ViewForge MCP 可以在之后通过 OpenAI Secure MCP Tunnel 接入 ChatGPT Developer Mode。

本次任务范围：只完成本地 App、MCP 运行时及本地验证。不要创建或修改用户的 OpenAI Tunnel，不要申请或读取真实 API Key，不要连接 ChatGPT，不要提交 Git、推送、创建 PR、打包发布或上传任何文件，除非用户之后明确授权。

开始前：

1. 阅读仓库中的 `AGENTS.md`、README、现有 `local-app/`、`src/face3d/local_mcp/`、`plugins/viewforge-3d-toolkit/`、`pyproject.toml` 和相关测试。
2. 运行 `git status --short`，记录已有改动。它们属于用户；不要覆盖、还原、暂存或提交无关文件。
3. 检查当前 Mac 架构、macOS 版本、Swift、uv、ripgrep 和 Blender 是否可用。不要静默安装大型依赖；需要安装时先说明原因并请求用户授权。
4. 优先复用仓库已有架构和构建脚本。如果实现已经存在，只做构建所必需的最小修复；不要重写已经正常工作的模块。

必须达到的本地 App 契约：

- 产品名：`ViewForge Local`。
- 目标系统：macOS 14 或更高版本，SwiftUI 界面。
- 构建输出：`dist/viewforge-local/ViewForge Local.app`。
- MCP transport：Streamable HTTP，且只能监听 `127.0.0.1:8765`。
- MCP URL：`http://127.0.0.1:8765/mcp`。
- 健康检查：`http://127.0.0.1:8765/healthz`；只有工作区、Blender 和插件运行时均有效时才返回 `status: ready`。
- Blender 是外部依赖，优先检测 `/Applications/Blender.app/Contents/MacOS/Blender` 和 `~/Applications/Blender.app/Contents/MacOS/Blender`。
- Blender 任务必须后台运行、禁用脚本自动执行，并产生新的不可变任务目录；不得覆盖输入模型。
- App 本地状态目录：`~/Library/Application Support/ViewForge Local`。
- App 必须允许用户选择工作区、重新检测 Blender、查看 MCP 状态、复制 MCP URL、打开本地记录，并检测/启动名为 `viewforge-local` 的 tunnel-client Profile。
- tunnel-client 只从 `/opt/homebrew/bin/tunnel-client`、`/usr/local/bin/tunnel-client` 或 `~/.local/bin/tunnel-client` 检测。
- App 不得内置 API Key、Tunnel ID、用户名、开发机绝对路径或个人组织/项目 ID。
- 如果 App 支持从工作区 `.env.local` 读取 Tunnel 运行密钥，只允许变量 `OPENAI_API_KEY`，必须拒绝权限宽于 `600` 的文件，并且不得把密钥写入日志、配置、错误消息或进程参数。

必须达到的 MCP 契约：

- 所有可注册资产必须位于用户选择的工作区内；解析符号链接后仍需检查边界，拒绝目录穿越和工作区外文件。
- 支持 `.glb`、`.gltf`、`.blend`、`.json`、`.png`、`.jpg`、`.jpeg`、`.mov` 和 `.mp4`。
- 注册后使用不可猜测或内容寻址的 `asset_...` ID；任务使用 `job_...`；产物使用 `artifact_...`。
- 对外工具结果不得返回本地绝对路径。读取 JSON 产物时必须清理工作区、插件和状态目录路径。
- 写操作生成新产物，不得修改源资产。
- 至少提供并正确标注以下工具：
  - `viewforge_status`
  - `register_local_asset`
  - `list_local_assets`
  - `build_biological_skeleton`
  - `create_bone_animation`
  - `bind_rigid_components`
  - `get_viewforge_job`
  - `list_viewforge_jobs`
  - `list_job_artifacts`
  - `read_json_artifact`
- 骨骼任务支持 `humanoid-v1` 和 `quadruped-v1`。
- 动画任务必须保留根骨骼的合理 X 轴位移，并支持只生成骨骼动画。
- 刚性绑定用于分段方块模型：组件跟随骨骼，但不创建 skin weights，也不做网格形变。
- 同一时间只允许一个 Blender 写任务排队或运行，避免多个进程同时修改同一状态。

如果 `local-app/` 已经完整：

1. 先运行 `swift build --package-path local-app`。
2. 运行仓库已有的本地 MCP、骨骼、动画和绑定测试。
3. 使用 `./local-app/scripts/build_app.sh release` 构建 Release App。
4. 只有测试或构建证明存在问题时才修改源码，然后重新运行失败检查。

如果 `local-app/` 尚不存在或不完整：

1. 创建 Swift Package，最低 macOS 版本为 14，提供 SwiftUI 可执行目标 `ViewForgeLocal`。
2. 创建上述 SwiftUI 控制界面和进程生命周期管理；退出 App 时可靠终止它启动的 MCP 与 tunnel-client 子进程。
3. 在 `src/face3d/local_mcp/` 实现配置、受限资产注册、不可变任务、Blender worker、结构化结果和 MCP server。
4. 在 `local-app/scripts/build_app.sh` 实现可重复构建：
   - 使用 uv 管理的 CPython 3.11；
   - 把 Python 运行时、MCP Python 依赖、`face3d.local_mcp` 和 `viewforge-3d-toolkit` 复制进 App Resources；
   - 修正 App 内 Python 启动器和动态库引用；
   - 删除指向开发机的 Swift toolchain rpath；
   - 对本地测试产物执行 ad-hoc codesign；
   - 不执行 Developer ID 签名、公证、DMG 打包或发布。
5. 在 `plugins/viewforge-3d-toolkit/.mcp.json` 和 `bin/viewforge-local-mcp` 中提供 stdio 回退入口，但不得写死仓库或用户绝对路径。
6. 添加单元测试，覆盖工作区越界、符号链接逃逸、资产 ID、路径脱敏、任务状态、并发写任务限制和工具 annotations。

构建完成后必须验证：

1. `swift build --package-path local-app` 成功。
2. 相关 Python 测试全部成功。
3. `./local-app/scripts/build_app.sh release` 成功并产生预期 `.app`。
4. `codesign --verify --deep --strict "dist/viewforge-local/ViewForge Local.app"` 成功。
5. 扫描 App bundle，确认没有开发机 `/Users/...`、真实 API Key、真实 Tunnel ID、组织 ID、项目 ID、私钥或 `.env.local`。
6. 如果当前权限允许启动 GUI，启动 App、选择一个临时工作区，确认 `/healthz` 返回 `ready`；如果 GUI 权限不足，明确报告未执行，不能把它写成通过。
7. 对 MCP 做一次只读 `viewforge_status` 验证。除非用户提供专用测试模型并明确允许，否则不要创建 Blender 写任务。
8. 构建或测试生成的临时文件只能放在仓库构建目录或系统临时目录；清理时只能删除本次任务明确创建的临时文件。

最终只报告：

- App 相对输出位置；
- 实际运行过的检查及结果；
- 是否仍依赖外部 Blender；
- 未执行或失败的检查与原因；
- 工作区中哪些文件由本次任务修改。

不要在最终回复中输出 API Key、Tunnel ID、本机用户名、绝对工作区路径或未经脱敏的日志。不要仅因为编译成功就声称 ChatGPT Tunnel 已经连接；Tunnel 配置由用户按照本指南后续步骤自行完成。
```

这个提示词刻意把“构建本地 App”和“连接用户自己的 Tunnel”分开。Agent 负责实现与验证本机运行时；用户负责 Platform 权限、运行密钥、Tunnel ID 和 ChatGPT Developer Mode。

## 12. 最终检查清单

- [ ] App 由用户自己构建，或来自已签名、公证的可信发行包。
- [ ] Blender 已安装并被 App 检测到。
- [ ] MCP `/healthz` 返回 `ready`。
- [ ] `tunnel-client` 架构与 Mac 一致。
- [ ] Tunnel 属于用户自己的 Platform 组织并关联正确的 ChatGPT 工作区。
- [ ] 使用独立、最小权限的运行密钥，没有使用 Admin API Key。
- [ ] `.env.local` 权限为 `600`，且未提交到 Git。
- [ ] Tunnel Profile 中只有密钥环境变量引用，没有密钥明文。
- [ ] Tunnel `/readyz` 就绪。
- [ ] ChatGPT 私人 App 可以调用 `viewforge_status`。
- [ ] 对话和工具调用只使用工作区相对路径。

完成以上步骤后，每位用户都拥有一套只在自己 Mac 上运行、由自己 Tunnel 和 API Key 控制的 ViewForge 3D 私人连接。
