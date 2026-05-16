#!/bin/bash
set -euo pipefail

version="$1"
platform="$2"
release_dir="$(swift build -c release --show-bin-path)"
artifact="plumeria-mathematics-${version}-${platform}"
staging="dist/${artifact}"

rm -rf "$staging" "${staging}.tar.gz" "${staging}.sha256"
mkdir -p "$staging/lib" "$staging/Modules" "$staging/include" "$staging/metadata"
find "$release_dir" -maxdepth 1 -type f -name 'libPlumeriaMathematics.*' -exec cp {} "$staging/lib/" \;
find "$release_dir" -maxdepth 1 -type d -name 'libPlumeriaMathematics.*.dSYM' -exec cp -R {} "$staging/lib/" \;
test -n "$(find "$staging/lib" -maxdepth 1 -type f -name 'libPlumeriaMathematics.*' -print -quit)"
cp "$release_dir"/Modules/*.{swiftmodule,swiftdoc,swiftsourceinfo} "$staging/Modules/" 2>/dev/null || true
find "$release_dir" -path '*/include/*' -type f -exec cp {} "$staging/include/" \;
swift --version > "$staging/metadata/swift-version.txt"
uname -a > "$staging/metadata/uname.txt"
printf '%s\n' "$version" > "$staging/metadata/version.txt"
printf '%s\n' "$platform" > "$staging/metadata/platform.txt"
tar -czf "${staging}.tar.gz" -C dist "$artifact"
shasum -a 256 "${staging}.tar.gz" > "${staging}.sha256"
