# Releases

## SwiftPM

To use Plumeria Mathematics in another Swift package, add the GitHub package URL and version:

```swift
.package(url: "https://github.com/bmurakami/plumeria-mathematics.git", from: "0.1.0")
```

Then add the library product to your target:

```swift
.product(name: "PlumeriaMathematics", package: "plumeria-mathematics")
```

## Platform Artifacts

Tagged releases publish archives for every CI-supported platform:

- `plumeria-mathematics-<version>-darwin-m1.tar.gz`: Apple Silicon Macs.
- `plumeria-mathematics-<version>-linux-x86_64.tar.gz`: typical Intel/AMD Linux computers.
- `plumeria-mathematics-<version>-linux-aarch64.tar.gz`: Raspberry Pi 3-5 and generic 64-bit ARM Linux machines.
- `plumeria-mathematics-<version>-linux-neoverse_v2.tar.gz`: AWS Graviton4 and similar Neoverse V2 ARM servers.

Each archive contains:

- `lib/`: the built `libPlumeriaMathematics` dynamic library.
- `Modules/`: Swift module, documentation, and source-info files emitted by the compiler.
- `include/`: generated Swift headers and module maps from the release build.
- `metadata/`: version, platform, Swift compiler version, and host information.

The `.sha256` file next to each archive records the archive checksum.
