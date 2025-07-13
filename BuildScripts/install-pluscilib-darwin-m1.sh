#!/bin/bash
set -e

curl -L "https://github.com/bmurakami/plumeria-scientific-libraries/releases/download/test-release/plumeria-libraries-darwin-m1.zip" -o openblas.zip
unzip -o openblas.zip -d Sources/COpenBLAS
rm openblas.zip
ln -s Sources/COpenBLAS/include/openblas/cblas.h Sources/COpenBLAS/include/cblas.h
ln -s Sources/COpenBLAS/include/openblas/lapack.h Sources/COpenBLAS/include/lapack.h
