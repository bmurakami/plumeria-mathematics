#!/bin/bash
set -e

curl -L "https://github.com/bmurakami/plumeria-scientific-libraries/releases/download/0.1-latest/pluscilib-darwin-m1.tar.gz" -o openblas.tar.gz
tar -xzf openblas.tar.gz -C Sources/COpenBLAS
rm openblas.tar.gz
cd Sources/COpenBLAS/include
cp -rf openblas/* .
rm -rf openblas
cd -
