#!/bin/zsh

spack env activate --dir plumath-spack
spack concretize
spack install  
spack load openblas

export OPENBLAS_INCLUDE_PATH=$(pkg-config --variable=includedir openblas)
