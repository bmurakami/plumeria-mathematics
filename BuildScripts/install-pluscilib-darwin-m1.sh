#!/bin/bash
set -e

curl -L "https://github.com/bmurakami/plumeria-scientific-libraries/releases/download/test-release/pluscilib-darwin-m1.zip" -o openblas.zip
unzip -o openblas.zip -d Sources/COpenBLAS
rm openblas.zip
cd Sources/COpenBLAS/include
cp -rf openblas/* .
rm -rf openblas
cd -
