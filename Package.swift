// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "PerthCoreML",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [
        .library(name: "PerthCoreML", targets: ["PerthCoreML"]),
        .executable(name: "perth-cli", targets: ["PerthCLI"]),
    ],
    targets: [
        .target(
            name: "PerthCoreML",
            swiftSettings: [.define("ACCELERATE_NEW_LAPACK")]
        ),
        .executableTarget(name: "PerthCLI", dependencies: ["PerthCoreML"]),
        .testTarget(
            name: "PerthCoreMLTests",
            dependencies: ["PerthCoreML"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
