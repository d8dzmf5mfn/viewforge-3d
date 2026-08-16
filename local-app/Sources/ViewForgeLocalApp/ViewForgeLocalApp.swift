import AppKit
import Combine
import Foundation
import SwiftUI

private let appName = "ViewForge Local"
private let defaultHost = "127.0.0.1"
private let defaultPort = 8765

private struct LocalConfiguration: Codable {
    var schemaVersion = 1
    var workspaceRoot: String?
    var blenderExecutable: String?
    var pluginRoot: String?
    var mcpHost = defaultHost
    var mcpPort = defaultPort
}

@MainActor
private final class AppController: ObservableObject {
    @Published var workspace = "未选择"
    @Published var blenderStatus = "检查中"
    @Published var modelingStatus = "检查中"
    @Published var mcpStatus = "未启动"
    @Published var tunnelStatus = "未安装 tunnel-client"
    @Published var profile = "viewforge-local"
    @Published var lastError: String?

    private var serverProcess: Process?
    private var serverLog: FileHandle?
    private var tunnelProcess: Process?
    private var tunnelLog: FileHandle?

    private var supportDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(appName, isDirectory: true)
    }

    private var configurationURL: URL {
        supportDirectory.appendingPathComponent("config.json")
    }

    private var resourcesURL: URL {
        Bundle.main.resourceURL ?? Bundle.main.bundleURL.appendingPathComponent("Contents/Resources")
    }

    private var pythonURL: URL {
        resourcesURL.appendingPathComponent("runtime/bin/python3.11")
    }

    private var pythonSourceURL: URL {
        resourcesURL.appendingPathComponent("python")
    }

    private var pluginURL: URL {
        resourcesURL.appendingPathComponent("viewforge-3d-toolkit")
    }

    private var mcpURL: URL {
        URL(string: "http://\(defaultHost):\(defaultPort)/mcp")!
    }

    init() {
        prepareConfiguration()
        refreshLocalState()
    }

    func start() {
        startServer()
        refreshHealth()
    }

    func stop() {
        tunnelProcess?.terminate()
        tunnelProcess = nil
        tunnelLog?.closeFile()
        tunnelLog = nil
        serverProcess?.terminate()
        serverProcess = nil
        serverLog?.closeFile()
        serverLog = nil
    }

    func chooseWorkspace(language: AppLanguage) {
        let localizer = AppLocalizer(language: language)
        let panel = NSOpenPanel()
        panel.title = localizer.text(
            "选择 ViewForge 本地工作区",
            "Choose the ViewForge local workspace"
        )
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = localizer.text("选择", "Choose")
        guard panel.runModal() == .OK, let selected = panel.url else { return }
        var configuration = loadConfiguration()
        configuration.workspaceRoot = selected.resolvingSymlinksInPath().path
        saveConfiguration(configuration)
        refreshLocalState()
        restartServer()
    }

    func detectBlender() {
        var configuration = loadConfiguration()
        configuration.blenderExecutable = defaultBlenderURL()?.path
        saveConfiguration(configuration)
        refreshLocalState()
        restartServer()
    }

    func openSupportDirectory() {
        try? FileManager.default.createDirectory(
            at: supportDirectory,
            withIntermediateDirectories: true
        )
        NSWorkspace.shared.open(supportDirectory)
    }

    func copyMCPURL() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(mcpURL.absoluteString, forType: .string)
    }

    func copyTunnelRunCommand() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(
            "tunnel-client run --profile \(profile)",
            forType: .string
        )
    }

    func openTunnelGuide() {
        guard let url = URL(
            string: "https://developers.openai.com/api/docs/guides/secure-mcp-tunnels"
        ) else { return }
        NSWorkspace.shared.open(url)
    }

    func toggleTunnel() {
        if let process = tunnelProcess, process.isRunning {
            process.terminate()
            tunnelProcess = nil
            tunnelLog?.closeFile()
            tunnelLog = nil
            tunnelStatus = "已停止"
            return
        }
        guard let executable = tunnelClientURL() else {
            tunnelStatus = "未安装 tunnel-client"
            return
        }
        guard let apiKey = openAIAPIKey() else {
            tunnelStatus = "缺少 API Key"
            return
        }
        do {
            let logURL = supportDirectory.appendingPathComponent("tunnel.log")
            try FileManager.default.createDirectory(
                at: supportDirectory,
                withIntermediateDirectories: true
            )
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
            let log = try FileHandle(forWritingTo: logURL)
            let process = Process()
            process.executableURL = executable
            process.arguments = ["run", "--profile", profile]
            var environment = ProcessInfo.processInfo.environment
            environment["OPENAI_API_KEY"] = apiKey
            process.environment = environment
            process.standardOutput = log
            process.standardError = log
            process.terminationHandler = { [weak self] _ in
                Task { @MainActor in
                    self?.tunnelStatus = "已停止；检查 tunnel.log"
                    self?.tunnelProcess = nil
                    self?.tunnelLog?.closeFile()
                    self?.tunnelLog = nil
                }
            }
            try process.run()
            tunnelProcess = process
            tunnelLog = log
            tunnelStatus = "运行中"
        } catch {
            lastError = "无法启动 tunnel-client。"
            tunnelStatus = "启动失败"
        }
    }

    func refreshHealth() {
        guard let url = URL(string: "http://\(defaultHost):\(defaultPort)/healthz") else {
            return
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 1.0
        URLSession.shared.dataTask(with: request) { [weak self] data, response, _ in
            let healthy: Bool
            let modelingReady: Bool
            if let http = response as? HTTPURLResponse,
               http.statusCode == 200,
               let data,
               let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                healthy = payload["status"] as? String == "ready"
                modelingReady = payload["modelingRuntimeAvailable"] as? Bool == true
            } else {
                healthy = false
                modelingReady = false
            }
            Task { @MainActor in
                guard let self else { return }
                if healthy {
                    self.mcpStatus = "就绪 · 127.0.0.1:\(defaultPort)"
                    self.lastError = nil
                } else if self.serverProcess?.isRunning == true {
                    self.mcpStatus = "运行中 · 等待配置"
                } else {
                    self.mcpStatus = "未运行"
                }
                self.modelingStatus = modelingReady ? "完整几何运行时可用" : "几何运行时不可用"
            }
        }.resume()
    }

    private func prepareConfiguration() {
        try? FileManager.default.createDirectory(
            at: supportDirectory,
            withIntermediateDirectories: true
        )
        if !FileManager.default.fileExists(atPath: configurationURL.path) {
            var configuration = LocalConfiguration()
            configuration.workspaceRoot = findDevelopmentRoot()?.path
            configuration.blenderExecutable = defaultBlenderURL()?.path
            configuration.pluginRoot = pluginURL.path
            saveConfiguration(configuration)
            return
        }
        var configuration = loadConfiguration()
        configuration.pluginRoot = pluginURL.path
        if configuration.blenderExecutable == nil {
            configuration.blenderExecutable = defaultBlenderURL()?.path
        }
        saveConfiguration(configuration)
    }

    private func refreshLocalState() {
        let configuration = loadConfiguration()
        if let root = configuration.workspaceRoot,
           FileManager.default.fileExists(atPath: root) {
            workspace = root
        } else {
            workspace = "未选择"
        }
        if let blender = configuration.blenderExecutable,
           FileManager.default.isExecutableFile(atPath: blender) {
            blenderStatus = URL(fileURLWithPath: blender).deletingLastPathComponent()
                .deletingLastPathComponent().deletingLastPathComponent().lastPathComponent
        } else {
            blenderStatus = "未找到 Blender"
        }
        tunnelStatus = tunnelClientURL() == nil ? "未安装 tunnel-client" : "已安装 · 未启动"
    }

    private func startServer() {
        guard serverProcess?.isRunning != true else { return }
        guard FileManager.default.isExecutableFile(atPath: pythonURL.path) else {
            lastError = "App 内缺少 Python 运行时。"
            mcpStatus = "启动失败"
            return
        }
        guard FileManager.default.fileExists(atPath: pythonSourceURL.path) else {
            lastError = "App 内缺少 MCP 服务代码。"
            mcpStatus = "启动失败"
            return
        }
        do {
            let logURL = supportDirectory.appendingPathComponent("mcp-server.log")
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
            let log = try FileHandle(forWritingTo: logURL)
            let process = Process()
            process.executableURL = pythonURL
            process.arguments = [
                "-m", "face3d.local_mcp.server",
                "--transport", "streamable-http",
                "--host", defaultHost,
                "--port", String(defaultPort),
            ]
            var environment = ProcessInfo.processInfo.environment
            environment["PYTHONPATH"] = pythonSourceURL.path
            environment["VIEWFORGE_LOCAL_STATE_DIR"] = supportDirectory.path
            environment["PYTHONNOUSERSITE"] = "1"
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            process.environment = environment
            process.standardInput = FileHandle.nullDevice
            process.standardOutput = log
            process.standardError = log
            process.terminationHandler = { [weak self] process in
                Task { @MainActor in
                    guard let self else { return }
                    self.serverProcess = nil
                    self.serverLog?.closeFile()
                    self.serverLog = nil
                    if process.terminationStatus != 0 {
                        self.mcpStatus = "已停止 · 检查 mcp-server.log"
                    }
                }
            }
            try process.run()
            serverProcess = process
            serverLog = log
            mcpStatus = "启动中"
        } catch {
            lastError = "无法启动本地 MCP 服务。"
            mcpStatus = "启动失败"
        }
    }

    private func restartServer() {
        serverProcess?.terminate()
        serverProcess = nil
        serverLog?.closeFile()
        serverLog = nil
        startServer()
    }

    private func loadConfiguration() -> LocalConfiguration {
        guard let data = try? Data(contentsOf: configurationURL),
              let decoded = try? JSONDecoder().decode(LocalConfiguration.self, from: data) else {
            return LocalConfiguration()
        }
        return decoded
    }

    private func saveConfiguration(_ configuration: LocalConfiguration) {
        guard let data = try? JSONEncoder().encode(configuration) else { return }
        do {
            try data.write(to: configurationURL, options: .atomic)
        } catch {
            lastError = "无法保存本地配置。"
        }
    }

    private func defaultBlenderURL() -> URL? {
        let candidates = [
            URL(fileURLWithPath: "/Applications/Blender.app/Contents/MacOS/Blender"),
            FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Applications/Blender.app/Contents/MacOS/Blender"),
        ]
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    private func tunnelClientURL() -> URL? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let candidates = [
            URL(fileURLWithPath: "/opt/homebrew/bin/tunnel-client"),
            URL(fileURLWithPath: "/usr/local/bin/tunnel-client"),
            home.appendingPathComponent(".local/bin/tunnel-client"),
        ]
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    private func openAIAPIKey() -> String? {
        if let inherited = ProcessInfo.processInfo.environment["OPENAI_API_KEY"],
           !inherited.isEmpty {
            return inherited
        }

        let configuration = loadConfiguration()
        guard let workspaceRoot = configuration.workspaceRoot else {
            lastError = "请先选择包含 .env.local 的工作区。"
            return nil
        }
        let environmentURL = URL(fileURLWithPath: workspaceRoot, isDirectory: true)
            .appendingPathComponent(".env.local")

        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: environmentURL.path)
            if let permissions = attributes[.posixPermissions] as? NSNumber,
               permissions.intValue & 0o077 != 0 {
                lastError = ".env.local 权限过宽；请设为 600。"
                return nil
            }

            let contents = try String(contentsOf: environmentURL, encoding: .utf8)
            for rawLine in contents.components(separatedBy: .newlines) {
                var line = rawLine.trimmingCharacters(in: .whitespaces)
                guard !line.isEmpty, !line.hasPrefix("#") else { continue }
                if line.hasPrefix("export ") {
                    line.removeFirst("export ".count)
                }
                guard let separator = line.firstIndex(of: "=") else { continue }
                let name = line[..<separator].trimmingCharacters(in: .whitespaces)
                guard name == "OPENAI_API_KEY" else { continue }
                var value = String(line[line.index(after: separator)...])
                    .trimmingCharacters(in: .whitespaces)
                if value.count >= 2,
                   (value.first == "\"" && value.last == "\""
                    || value.first == "'" && value.last == "'") {
                    value.removeFirst()
                    value.removeLast()
                }
                if !value.isEmpty {
                    return value
                }
            }
            lastError = ".env.local 中缺少 OPENAI_API_KEY。"
        } catch {
            lastError = "无法读取工作区中的 .env.local。"
        }
        return nil
    }

    private func findDevelopmentRoot() -> URL? {
        var candidate = Bundle.main.bundleURL.deletingLastPathComponent()
        for _ in 0..<6 {
            let manifest = candidate.appendingPathComponent(
                "plugins/viewforge-3d-toolkit/.codex-plugin/plugin.json"
            )
            if FileManager.default.fileExists(atPath: manifest.path) {
                return candidate
            }
            candidate.deleteLastPathComponent()
        }
        return nil
    }
}

private struct StatusRow: View {
    let title: String
    let value: String
    let ready: Bool

    var body: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(ready ? Color.green : Color.orange)
                .frame(width: 9, height: 9)
            Text(title)
                .frame(width: 110, alignment: .leading)
            Text(value)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
        }
    }
}

private struct ContentView: View {
    @EnvironmentObject private var controller: AppController
    @AppStorage(AppLanguage.storageKey) private var languageID = AppLanguage.system.rawValue
    private let timer = Timer.publish(every: 2, on: .main, in: .common).autoconnect()

    private var language: AppLanguage {
        AppLanguage(rawValue: languageID) ?? .system
    }

    private var localizer: AppLocalizer {
        AppLocalizer(language: language)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 5) {
                    Text("ViewForge Local")
                        .font(.system(size: 28, weight: .semibold))
                    Text(localizer.text(
                        "本机多视图建模、Blender 渲染、骨骼与动画 MCP",
                        "Local multiview modeling, Blender rendering, rigging, and animation MCP"
                    ))
                    .foregroundStyle(.secondary)
                }
                Spacer()
                Picker(localizer.text("语言", "Language"), selection: $languageID) {
                    ForEach(AppLanguage.allCases) { option in
                        Text(localizer.languageName(option)).tag(option.rawValue)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(width: 220)
                .accessibilityLabel(localizer.text("语言", "Language"))
            }

            GroupBox(localizer.text("本机运行时", "Local runtime")) {
                VStack(alignment: .leading, spacing: 13) {
                    StatusRow(
                        title: localizer.text("工作区", "Workspace"),
                        value: localizer.runtimeValue(controller.workspace),
                        ready: controller.workspace != "未选择"
                    )
                    StatusRow(
                        title: "Blender",
                        value: localizer.runtimeValue(controller.blenderStatus),
                        ready: controller.blenderStatus != "未找到 Blender"
                    )
                    StatusRow(
                        title: localizer.text("几何运行时", "Geometry runtime"),
                        value: localizer.runtimeValue(controller.modelingStatus),
                        ready: controller.modelingStatus == "完整几何运行时可用"
                    )
                    StatusRow(
                        title: "MCP",
                        value: localizer.runtimeValue(controller.mcpStatus),
                        ready: controller.mcpStatus.hasPrefix("就绪")
                    )
                    HStack {
                        Button(localizer.text("选择工作区", "Choose workspace")) {
                            controller.chooseWorkspace(language: language)
                        }
                        Button(localizer.text("重新检测 Blender", "Detect Blender")) {
                            controller.detectBlender()
                        }
                        Button(localizer.text("复制 MCP URL", "Copy MCP URL")) {
                            controller.copyMCPURL()
                        }
                        Button(localizer.text("打开本地记录", "Open local records")) {
                            controller.openSupportDirectory()
                        }
                    }
                }
                .padding(8)
            }

            GroupBox(localizer.text("ChatGPT 私有连接", "Private ChatGPT connection")) {
                VStack(alignment: .leading, spacing: 12) {
                    StatusRow(
                        title: localizer.text("安全隧道", "Secure tunnel"),
                        value: localizer.runtimeValue(controller.tunnelStatus),
                        ready: controller.tunnelStatus == "运行中"
                    )
                    HStack {
                        Text("Profile")
                        TextField("viewforge-local", text: $controller.profile)
                            .textFieldStyle(.roundedBorder)
                            .frame(maxWidth: 240)
                        Button(controller.tunnelStatus == "运行中"
                            ? localizer.text("停止", "Stop")
                            : localizer.text("启动", "Start")) {
                            controller.toggleTunnel()
                        }
                        Button(localizer.text("复制运行命令", "Copy run command")) {
                            controller.copyTunnelRunCommand()
                        }
                        Button(localizer.text("设置指南", "Setup guide")) {
                            controller.openTunnelGuide()
                        }
                    }
                    Text(localizer.text(
                        "先在 OpenAI Platform 创建 tunnel，并让其转发到 127.0.0.1:8765/mcp。",
                        "Create a tunnel in OpenAI Platform and forward it to 127.0.0.1:8765/mcp."
                    ))
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                .padding(8)
            }

            if let error = controller.lastError {
                Text(localizer.runtimeValue(error))
                    .foregroundStyle(.red)
                    .font(.callout)
            }

            Spacer()
            Text(localizer.text(
                "结果保存在本机；MCP 默认返回 ID，仅在显式读取时返回选定渲染图。",
                "Results stay local; MCP returns IDs by default and images only when explicitly read."
            ))
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding(28)
        .frame(minWidth: 760, minHeight: 540)
        .environment(\.locale, language.locale)
        .onReceive(timer) { _ in controller.refreshHealth() }
    }
}

@MainActor
private final class AppDelegate: NSObject, NSApplicationDelegate {
    let controller = AppController()

    func applicationDidFinishLaunching(_ notification: Notification) {
        controller.start()
    }

    func applicationWillTerminate(_ notification: Notification) {
        controller.stop()
    }
}

@main
private struct ViewForgeLocalApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(delegate.controller)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 800, height: 570)
    }
}
