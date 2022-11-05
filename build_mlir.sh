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
#sedinplace -e "/set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)/a\\
#set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)" ../llvm-project/llvm/CMakeLists.txt

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
#ninja MLIRPythonModules mlir-tblgen tools/mlir/tools/mlir-tblgen/all lib/Support/all lib/all cmake/modules/all lib/Demangle/all lib/TableGen/all utils/TableGen/all
## https://stackoverflow.com/a/45128493/9045206
#ninja tools/mlir/python/install
#ninja tools/mlir/tools/mlir-tblgen/install
#ninja tools/mlir/install
#ninja lib/Support/install
#ninja lib/Demangle/install
#ninja lib/TableGen/install
#ninja utils/TableGen/install
#ninja lib/install
#ninja cmake/modules/install
ninja install
cp bin/mlir-tblgen ../install_mlir/bin/mlir-tblgen

echo "thank you come agaimo!"