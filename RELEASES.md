# Releases

## SwiftPM

To use Plumeria Mathematics (PluMath) in another Swift package, add the GitHub package to your `dependencies`
list in your `Package.swift` file.

```swift
.package(url: "https://github.com/bmurakami/plumeria-mathematics.git", from: "0.1.0")
```

Then declare the library product to your target's `dependencies` list:

```swift
.product(name: "PlumeriaMathematics", package: "plumeria-mathematics")
```

## Plumeria Scientific Libraries

PluMath uses platform-specific binaries from Plumeria Scientific Libraries hosted on GitHub. The
`BuildScripts/install-pluscilib-*` scripts are called by GitHub Actions, which downloads the correct
binary libraries and copies them to PluMath's `Sources` directory.

If PluMath was manually cloned, the build script for your platform must manually be invoked before
building with SwiftPM or Xcode.

## GitHub Releases

Pushing a tag creates a GitHub release. To publish a release:

```bash
git tag 0.1.1
git push origin 0.1.1
```
