// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "PlumeriaMathematics",
    products: [
        .library(name: "Tensors", targets: ["Tensors"]),
        .library(name: "LinearSolvers", targets: ["LinearSolvers"]),
    ],
    targets: [
        .systemLibrary(name: "COpenBLAS", path: "Sources/COpenBLAS"),
        .target(name: "Tensors"),
        .target(
            name: "LinearSolvers",
            dependencies: ["COpenBLAS"],
            linkerSettings: [.unsafeFlags(["-L/Users/murakami1/.local/spack/opt/spack/darwin-m1/openblas-0.3.29-urtlqs4wcu3shs6ryfazhsk7rknmjklu"])]
        ),
        .testTarget(name: "TensorsTests", dependencies: ["Tensors"]),
        .testTarget(
            name: "LinearSolversTests",
            dependencies: ["LinearSolvers", "COpenBLAS"],
            linkerSettings: [.unsafeFlags(["-L/Users/murakami1/.local/spack/opt/spack/darwin-m1/openblas-0.3.29-urtlqs4wcu3shs6ryfazhsk7rknmjklu"])]
        ),
    ]
)
