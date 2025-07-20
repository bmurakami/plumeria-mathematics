// Package.swift
// swift-tools-version: 6.0

import Foundation
import PackageDescription

let package = Package(
    name: "PlumeriaMathematics",
    platforms: [.macOS(.v15)],
    products: [.library(name: "PlumeriaMathematics", targets: ["PlumeriaMathematics"])],
    dependencies: [.package(url: "https://github.com/apple/swift-numerics", from: "1.0.0")],
    targets: [
        .systemLibrary(name: "COpenBLAS", path: "Sources/COpenBLAS"),
        .target(name: "AccelerateWrapper", cSettings: [.define("ACCELERATE_NEW_LAPACK")]),
        .target(name: "OpenBLASWrapper", dependencies: ["COpenBLAS"]),
        .target(name: "Tensors", dependencies: ["AccelerateWrapper", "OpenBLASWrapper",
            .product(name: "Numerics", package: "swift-numerics")]),
        .target(name: "LinearSolvers", dependencies: ["Tensors"],
                linkerSettings: [.unsafeFlags(["\(Context.packageDirectory)/Sources/COpenBLAS/lib/libopenblas.a"])]),
        .target(name: "PlumeriaMathematics", dependencies: ["Tensors", "LinearSolvers"]),
        .testTarget(name: "TensorsTests", dependencies: ["Tensors"]),
        .testTarget(name: "LinearSolversTests", dependencies: ["LinearSolvers"]),
    ]
)
