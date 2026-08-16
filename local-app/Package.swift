// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ViewForgeLocal",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "ViewForgeLocal", targets: ["ViewForgeLocalApp"]),
    ],
    targets: [
        .executableTarget(
            name: "ViewForgeLocalApp",
            path: "Sources/ViewForgeLocalApp"
        ),
    ]
)
