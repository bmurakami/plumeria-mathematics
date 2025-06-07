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
            linkerSettings: [.unsafeFlags([
                "/Users/murakami1/.local/spack/opt/spack/darwin-m1/openblas-0.3.29-urtlqs4wcu3shs6ryfazhsk7rknmjklu/lib/libopenblas.a"
            ])]
        ),
        .testTarget(name: "TensorsTests", dependencies: ["Tensors"]),
        .testTarget(
            name: "LinearSolversTests",
            dependencies: ["LinearSolvers", "COpenBLAS"]
        ),
    ]
)
