// Package.swift
// swift-tools-version: 6.0

import Foundation
import PackageDescription

let package = Package(
    name: "PlumeriaMathematics",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "PlumeriaMathematics", type: .dynamic, targets: ["PlumeriaMathematics"]),
        .executable(name: "PlumeriaBenchmarks", targets: ["PlumeriaBenchmarks"]),
    ],
    dependencies: [.package(url: "https://github.com/apple/swift-numerics", from: "1.0.0")],
    targets: [
        .systemLibrary(name: "COpenBLAS", path: "Sources/COpenBLAS"),
        .target(name: "AccelerateWrapper", cSettings: [.define("ACCELERATE_NEW_LAPACK")]),
        .target(name: "OpenBLASWrapper", dependencies: ["COpenBLAS"],
                linkerSettings: [.linkedLibrary("gfortran", .when(platforms: [.linux]))]),
        .target(name: "Tensors", dependencies: ["AccelerateWrapper", "OpenBLASWrapper",
            .product(name: "Numerics", package: "swift-numerics")]),
        .target(name: "LinearSolvers", dependencies: ["Tensors"],
                linkerSettings: [.unsafeFlags(["\(Context.packageDirectory)/Sources/COpenBLAS/lib/libopenblas.a"])]),
        .target(name: "PlumeriaMathematics", dependencies: ["Tensors", "LinearSolvers"]),
        .executableTarget(name: "PlumeriaBenchmarks", dependencies: ["PlumeriaMathematics"], exclude: ["README.md"]),
        .testTarget(name: "TensorsTests", dependencies: ["Tensors"]),
        .testTarget(name: "LinearSolversTests", dependencies: ["LinearSolvers"]),
    ]
)
