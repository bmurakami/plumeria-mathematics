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

## GitHub Releases

Version tags create GitHub releases. They do not publish Plumeria Mathematics binary archives.

SwiftPM source dependencies are the supported release path. Platform-specific native OpenBLAS
dependencies are handled separately by the `BuildScripts/install-pluscilib-*` scripts.

To publish a release:

```bash
git tag 0.1.1
git push origin 0.1.1
```
