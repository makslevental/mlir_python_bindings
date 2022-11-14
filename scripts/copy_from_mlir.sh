#!/bin/bash

set -e -x

LLVM_PROJECT_SRC_DIR=$1

function sedinplace {
  if ! sed --version 2>&1 | grep -i gnu >/dev/null; then
    sed -i '' "$@"
  else
    sed -i "$@"
  fi
}

cp -R $LLVM_PROJECT_SRC_DIR/mlir/python/mlir/. ../python/mlir
cp -R $LLVM_PROJECT_SRC_DIR/mlir/lib/Bindings/Python/. ../cpp/lib/
mkdir -p ../cpp/include/mlir/Bindings/Python
cp -R $LLVM_PROJECT_SRC_DIR/mlir/include/mlir/Bindings/Python ../cpp/include/mlir/Bindings/
mkdir -p ../cpp/include/mlir-c/Bindings/Python
cp -R $LLVM_PROJECT_SRC_DIR/mlir/include/mlir-c/Bindings/Python ../cpp/include/mlir-c/Bindings/
cp -R $LLVM_PROJECT_SRC_DIR/mlir/python/CMakeLists.txt .

sedinplace "s/MLIRPython/StandaloneMLIRPython/g" CMakeLists.txt
sedinplace "s/DEPENDS LinalgOdsGen)/#DEPENDS LinalgOdsGen\n)/g" CMakeLists.txt
sedinplace "s/include(AddStandaloneMLIRPython)//g" CMakeLists.txt
# https://stackoverflow.com/a/53541941/9045206
sedinplace 's/^[^#]*set(PYTHON_SOURCE_DIR/#&/' CMakeLists.txt
# `q`uit after hitting the reset line
sedinplace '/# CMakeLists RESET/q' ../CMakeLists.txt

cat CMakeLists.txt >> ../CMakeLists.txt
