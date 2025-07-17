#!/bin/bash
set -e

curl -L "https://github.com/bmurakami/plumeria-scientific-libraries/releases/download/test-release/pluscilib-linux-x86_64.zip" -o openblas.zip
unzip -o openblas.zip -d Sources/COpenBLAS
rm openblas.zip
