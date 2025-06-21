#!/bin/zsh
set -e

spack env activate --dir plumath-spack
spack install
spack env view regenerate
sed "s|PLUMATH_PATH|$(pwd)|g" Sources/COpenBLAS/template.modulemap > Sources/COpenBLAS/module.modulemap
