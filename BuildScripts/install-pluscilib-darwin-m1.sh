#!/bin/bash
set -e

curl -L "https://github.com/bmurakami/plumeria-scientific-libraries/releases/download/test-release/plumeria-libraries-darwin-m1.zip" -o openblas.zip
unzip -o openblas.zip -d Sources/COpenBLAS
rm openblas.zip
cd Sources/COpenBLAS/include
cp -r openblas/* .
rm -rf openblas
cd -
