// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "PlumeriaMathematics",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "PlumeriaTensors", targets: ["Tensors"]),
        .library(name: "PlumeriaLinearSolvers", targets: ["LinearSolvers"]),
    ],
    targets: [
        .systemLibrary(name: "COpenBLAS", path: "Sources/COpenBLAS"),
        .target(name: "OpenBLASWrapper", dependencies: ["COpenBLAS"]),
        .target(name: "AccelerateWrapper", cSettings: [.define("ACCELERATE_NEW_LAPACK")]),
        .target(name: "Tensors", dependencies: ["AccelerateWrapper", "OpenBLASWrapper"]),
        .target(
            name: "LinearSolvers",
            dependencies: ["Tensors", "COpenBLAS"],
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-rpath",
                    "-Xlinker", "plumath-spack/.spack-env/view/lib"
                ])
            ]
        ),
        .testTarget(name: "TensorsTests", dependencies: ["Tensors"]),
        .testTarget(
            name: "LinearSolversTests",
            dependencies: ["LinearSolvers", "COpenBLAS"]
        ),
    ]
)
