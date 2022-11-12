# MLIR Python bindings

The MLIR python bindings are usually [bundled with MLIR](https://github.com/llvm/llvm-project/blob/main/mlir/docs/Bindings/Python.md).
It's a little tedious getting everything hooked up, and then using them, since you have to set `PYTHONPATH` (no in-tree wheels currently).
This repo just refactors things a little and builds them as wheels (based on the [standalone example](https://github.com/llvm/llvm-project/blob/main/mlir/examples/standalone/README.md) and the [CIRCT wheels](https://github.com/llvm/circt/blob/main/lib/Bindings/Python/setup.py)).

# Building

Just 

```shell
pip install . -vvvvv
```

# Demo

The linalg-ish tutorial @ `tutorial/linalg_tut.py` is just a minimal-working example of taking a `linalg` operation (a `matmul`), lowers it to the `affine` dialect, and then unrolls it (and performs naive store-load forwarding).

```shell
$ python tutorial/linalg_tut.py

# before

module {
  func.func @demo(%arg0: tensor<4x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<4x8xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<4x16xf32>) -> tensor<4x16xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%arg1 : tensor<16x8xf32>) -> tensor<16x8xf32>
    %2 = linalg.init_tensor [4, 8] : tensor<4x8xf32>
    %3 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%0, %1 : tensor<4x16xf32>, tensor<16x8xf32>) outs(%2 : tensor<4x8xf32>) -> tensor<4x8xf32>
    return %3 : tensor<4x8xf32>
  }
}

# after

module {
  func.func @demo(%arg0: memref<4x16xf32>, %arg1: memref<16x8xf32>) -> memref<4x8xf32> {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x16xf32>
    affine.for %arg2 = 0 to 4 {
      affine.store %cst, %0[%arg2, %c0_1] : memref<4x16xf32>
      %3 = affine.apply #map0(%c0_1)
      affine.store %cst, %0[%arg2, %3] : memref<4x16xf32>
      %4 = affine.apply #map1(%c0_1)
      affine.store %cst, %0[%arg2, %4] : memref<4x16xf32>
      %5 = affine.apply #map2(%c0_1)
      affine.store %cst, %0[%arg2, %5] : memref<4x16xf32>
      %6 = affine.apply #map3(%c0_1)
      affine.store %cst, %0[%arg2, %6] : memref<4x16xf32>
      %7 = affine.apply #map4(%c0_1)
      affine.store %cst, %0[%arg2, %7] : memref<4x16xf32>
      %8 = affine.apply #map5(%c0_1)
      ...
```

# Updating

```shell
LLVM_BUILD_DIR=$(pwd)/build_mlir ./copy_from_mlir.sh
```
