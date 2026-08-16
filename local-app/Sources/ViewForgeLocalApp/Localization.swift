import Foundation

enum AppLanguage: String, CaseIterable, Identifiable {
    static let storageKey = "viewforge.app-language"

    case system
    case chinese
    case english

    var id: String { rawValue }

    var resolved: AppLanguage {
        guard self == .system else { return self }
        let preferred = Locale.preferredLanguages.first?.lowercased() ?? "en"
        return preferred.hasPrefix("zh") ? .chinese : .english
    }

    var locale: Locale {
        Locale(identifier: resolved == .chinese ? "zh-Hans" : "en")
    }
}

struct AppLocalizer {
    let language: AppLanguage

    func text(_ chinese: String, _ english: String) -> String {
        language.resolved == .chinese ? chinese : english
    }

    func languageName(_ option: AppLanguage) -> String {
        switch option {
        case .system:
            text("自动", "Auto")
        case .chinese:
            "中文"
        case .english:
            "English"
        }
    }

    func runtimeValue(_ value: String) -> String {
        guard language.resolved == .english else { return value }
        let translations = [
            "未选择": "Not selected",
            "检查中": "Checking",
            "未启动": "Not started",
            "未安装 tunnel-client": "tunnel-client not installed",
            "已停止": "Stopped",
            "缺少 API Key": "API key missing",
            "已停止；检查 tunnel.log": "Stopped; check tunnel.log",
            "运行中": "Running",
            "启动失败": "Failed to start",
            "运行中 · 等待配置": "Running · waiting for configuration",
            "未运行": "Not running",
            "完整几何运行时可用": "Full geometry runtime available",
            "几何运行时不可用": "Geometry runtime unavailable",
            "未找到 Blender": "Blender not found",
            "已安装 · 未启动": "Installed · not running",
            "已停止 · 检查 mcp-server.log": "Stopped · check mcp-server.log",
            "启动中": "Starting",
            "App 内缺少 Python 运行时。": "The bundled Python runtime is missing.",
            "App 内缺少 MCP 服务代码。": "The bundled MCP server code is missing.",
            "无法启动 tunnel-client。": "Unable to start tunnel-client.",
            "无法启动本地 MCP 服务。": "Unable to start the local MCP server.",
            "无法保存本地配置。": "Unable to save the local configuration.",
            "请先选择包含 .env.local 的工作区。":
                "Select a workspace containing .env.local first.",
            ".env.local 权限过宽；请设为 600。":
                ".env.local permissions are too broad; set them to 600.",
            ".env.local 中缺少 OPENAI_API_KEY。": ".env.local is missing OPENAI_API_KEY.",
            "无法读取工作区中的 .env.local。": "Unable to read .env.local in the workspace.",
        ]
        if let translated = translations[value] {
            return translated
        }
        if value.hasPrefix("就绪 · ") {
            return value.replacingOccurrences(of: "就绪 · ", with: "Ready · ")
        }
        return value
    }
}
