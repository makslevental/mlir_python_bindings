#!/usr/bin/env bash

set -e -x

git submodule sync
git submodule update --init --recursive --depth 1

mkdir -p build_mlir
mkdir -p install_mlir
cd build_mlir

function sedinplace {
    if ! sed --version 2>&1 | grep -i gnu > /dev/null; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

## https://stackoverflow.com/a/17201375/9045206
sedinplace -e "/set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)/a\\
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)" ../llvm-project/llvm/CMakeLists.txt

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
  -DCMAKE_INSTALL_PREFIX=../install_mlir \
  ../llvm-project/llvm

# building the in-tree modules is the minimal build such that we can build the standalones
ninja MLIRPythonModules \
  mlir-tblgen \
  tools/mlir/all \
  lib/Support/all \
  lib/Demangle/all \
  lib/all \
  cmake/modules/all \
  include/llvm/all

# https://stackoverflow.com/a/45128493/9045206
ninja tools/mlir/install
ninja lib/Support/install
ninja lib/Demangle/install
ninja lib/install
ninja cmake/modules/install
ninja include/llvm/install
#ninja install
cp bin/mlir-tblgen ../install_mlir/bin/mlir-tblgen
cp ../LLVMExports.cmake ../install_mlir/lib/cmake/llvm
cp ../LLVMExports-release.cmake ../install_mlir/lib/cmake/llvm

ARCH=(`uname -m | tr '[A-Z]' '[a-z]'`)
if [ x"$ARCH" == x"arm64" ]; then
  sedinplace 's/X86/AArch64/g' ../install_mlir/lib/cmake/llvm/LLVMExports.cmake
  sedinplace 's/"rt;/"System;/g' ../install_mlir/lib/cmake/llvm/LLVMExports.cmake
  sedinplace -e "/add_executable(sancov IMPORTED)/a\\
  add_library(LLVMAArch64Utils STATIC IMPORTED)\\
  set_target_properties(LLVMAArch64Utils PROPERTIES INTERFACE_LINK_LIBRARIES \"LLVMSupport;LLVMCore\")" ../install_mlir/lib/cmake/llvm/LLVMExports.cmake

  sedinplace 's/X86/AArch64/g' ../install_mlir/lib/cmake/llvm/LLVMExports-release.cmake
  sedinplace -e "/list(APPEND _cmake_import_check_targets LLVMOrcJIT )/a\\
  set_property(TARGET LLVMAArch64Utils APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)\\
  set_target_properties(LLVMAArch64Utils PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE \"CXX\" IMPORTED_LOCATION_RELEASE \"\${_IMPORT_PREFIX}/lib/libLLVMAArch64Utils.a\")\\
  list(APPEND _cmake_import_check_targets LLVMAArch64Utils)\\
  list(APPEND _cmake_import_check_files_for_LLVMAArch64Utils \"\${_IMPORT_PREFIX}/lib/libLLVMAArch64Utils.a\")" ../install_mlir/lib/cmake/llvm/LLVMExports-release.cmake
fi

cp -R ../llvm-project/llvm/include/* ../install_mlir/include
cp -R include/llvm/Config/. ../install_mlir/include/llvm/Config


echo "thank you come agaimo!"