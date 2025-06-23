// Package.swift
// swift-tools-version: 6.1

import Foundation
import PackageDescription

let package = Package(
    name: "PlumeriaMathematics",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "PlumeriaTensors", targets: ["Tensors"]),
        .library(name: "PlumeriaLinearSolvers", targets: ["LinearSolvers"]),
    ],
    targets: [
        .systemLibrary(name: "COpenBLAS", path: "Sources/COpenBLAS", pkgConfig: "openblas" ),
        .target(name: "AccelerateWrapper", cSettings: [.define("ACCELERATE_NEW_LAPACK")]),
        .target(
            name: "OpenBLASWrapper",
            dependencies: ["COpenBLAS"],
            cSettings: [.unsafeFlags(["-I", ProcessInfo.processInfo.environment["OPENBLAS_INCLUDE_PATH"]!])]
        ),
        .target(name: "Tensors", dependencies: ["AccelerateWrapper", "OpenBLASWrapper"]),
        .target(name: "LinearSolvers", dependencies: ["Tensors"]),
        .testTarget(name: "TensorsTests", dependencies: ["Tensors"]),
        .testTarget(name: "LinearSolversTests", dependencies: ["LinearSolvers"]),
    ]
)
