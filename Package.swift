// swift-tools-version: 5.9
import PackageDescription
import Foundation

let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let swiftSDKRelativePath = "sdk/swift"
let ffiXCFrameworkRelativePath = "\(swiftSDKRelativePath)/Generated/MeshLLMFFI.xcframework"
let ffiXCFrameworkPath = "\(repoRoot)/\(ffiXCFrameworkRelativePath)"
let remoteFFIXCFrameworkURL = "https://github.com/Mesh-LLM/mesh-llm/releases/download/v0.75.0/MeshLLMFFI.xcframework.zip"
let remoteFFIXCFrameworkChecksum = "e30e5ecabd2d2a855c1188cf4e3b20aa834fb2394c051e04259666f169ca49ec"
let hasLocalFFIXCFramework = FileManager.default.fileExists(atPath: ffiXCFrameworkPath)
let hasRemoteFFIXCFramework =
    !remoteFFIXCFrameworkURL.contains("__MESH_SWIFT_RELEASE_TAG__")
    && !remoteFFIXCFrameworkChecksum.contains("__MESH_SWIFT_RELEASE_CHECKSUM__")

var meshLLMDependencies: [Target.Dependency] = []
var packageTargets: [Target] = []

if hasLocalFFIXCFramework {
    meshLLMDependencies.append("MeshLLMFFI")
    packageTargets.append(
        .binaryTarget(
            name: "MeshLLMFFI",
            path: ffiXCFrameworkRelativePath
        )
    )
} else if hasRemoteFFIXCFramework {
    meshLLMDependencies.append("MeshLLMFFI")
    packageTargets.append(
        .binaryTarget(
            name: "MeshLLMFFI",
            url: remoteFFIXCFrameworkURL,
            checksum: remoteFFIXCFrameworkChecksum
        )
    )
}

let hasFFIBinaryTarget = hasLocalFFIXCFramework || hasRemoteFFIXCFramework

let package = Package(
    name: "MeshLLM",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        .library(
            name: "MeshLLM",
            targets: ["MeshLLM"]
        ),
    ],
    targets: [
        .target(
            name: "MeshLLM",
            dependencies: meshLLMDependencies,
            path: "sdk/swift/Sources/MeshLLM",
            exclude: hasFFIBinaryTarget ? [] : ["Generated"],
            resources: [
                .copy("Resources/Console"),
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("AppKit", .when(platforms: [.macOS])),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("Foundation"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("SystemConfiguration"),
                .linkedLibrary("c++"),
            ]
        ),
        .testTarget(
            name: "MeshLLMTests",
            dependencies: ["MeshLLM"],
            path: "sdk/swift/Tests/MeshLLMTests"
        ),
    ] + packageTargets
)
