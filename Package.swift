// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "PlumeriaMathematics",
    products: [
        .library(
            name: "LinearSolvers",
            targets: ["LinearSolvers"]),
    ],
    targets: [
        .target(
            name: "LinearSolvers"),
        .testTarget(
            name: "LinearSolversTests",
            dependencies: ["LinearSolvers"]
        ),
    ]
)
