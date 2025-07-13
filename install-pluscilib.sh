#!/bin/bash

architecture=${1:-"darwin-m1"}
curl -L "https://github.com/bmurakami/plumeria-scientific-libraries/releases/download/test-release/plumeria-libraries-${architecture}.zip" -o openblas.zip
unzip -o openblas.zip -d Sources/COpenBLAS
rm openblas.zip
