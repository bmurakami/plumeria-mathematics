#!/bin/bash
set -e

curl -L "https://github.com/bmurakami/plumeria-scientific-libraries/releases/download/0.1-latest/pluscilib-linux-neoverse_v2.tar.gz" -o openblas.tar.gz
tar -xzf openblas.tar.gz -C Sources/COpenBLAS
rm openblas.tar.gz
