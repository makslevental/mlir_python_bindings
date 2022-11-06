#!/bin/bash

set -e -x

PATH="$PATH_":$PATH

# The absolute path to the directory of this script.
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PATH_=$PATH bash "$HERE/check_path.sh"

PYTHON_LOC=${PYTHON_LOC:-$(which python3)}

echo "$PYTHON_LOC"
echo "$(which python3)"

python3 -m pip install pybind11==2.10.1 numpy wheel PyYAML dataclasses ninja==1.10.2 cmake==3.24.0 -U --force

git submodule sync
git submodule update --init --recursive --depth 1

mkdir -p build_mlir
INSTALL_DIR=$HERE/../install_mlir
mkdir -p $INSTALL_DIR
pushd build_mlir

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_BUILD_TESTS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_BUILD_UTILS=OFF \
  -DLLVM_INCLUDE_UTILS=OFF \
  -DLLVM_BUILD_RUNTIMES=OFF \
  -DLLVM_INCLUDE_RUNTIMES=OFF \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_BUILD_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_BUILD_TOOLS=ON \
  -DLLVM_INCLUDE_TOOLS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DPython3_EXECUTABLE="$PYTHON_LOC" \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
  ../llvm-project/llvm

ninja install
cp bin/mlir-tblgen $INSTALL_DIR/bin/mlir-tblgen

popd

echo "thank you come agaimo!"