#!/bin/bash

set -e -x

LLVM_PROJECT_SRC_DIR=$1

cp -R $LLVM_PROJECT_SRC_DIR/mlir/python/mlir/. ../python/mlir
cp -R $LLVM_PROJECT_SRC_DIR/mlir/lib/Bindings/Python/. ../cpp/lib
cp -R $LLVM_PROJECT_SRC_DIR/mlir/include/mlir/Bindings/Python/. ../cpp/include
cp $LLVM_PROJECT_SRC_DIR/mlir/include/mlir-c/Bindings/Python/Interop.h ../cpp/include
