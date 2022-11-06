# MLIR Python bindings

The MLIR python bindings are usually [bundled with MLIR](https://github.com/llvm/llvm-project/blob/main/mlir/docs/Bindings/Python.md).
It's a little tedious getting everything hooked up, and then using them, since you have to set `PYTHONPATH` (no in-tree wheels currently).
This repo just refactors things a little and builds them as wheels (based on the [standalone example](https://github.com/llvm/llvm-project/blob/main/mlir/examples/standalone/README.md) and the [CIRCT wheels](https://github.com/llvm/circt/blob/main/lib/Bindings/Python/setup.py)).

# Building

You do need `llvm-project` installed somewhere; if you don't then just run `scripts/build_mlir.sh`.
Then you can just pip install:

```shell
LLVM_INSTALL_DIR=$(pwd)/llvm_install pip install . -vvvvv
```

Alternatively

```shell
LLVM_INSTALL_DIR=$(pwd)/install_mlir pip wheel . -vvvvv
```

# Demo

```shell
$ python scripts/demo.py

<Dialect arith (class mlir.dialects._arith_ops_gen._Dialect)>
<Dialect bufferization (class mlir.dialects._bufferization_ops_gen._Dialect)>
<Dialect builtin (class mlir.dialects._builtin_ops_gen._Dialect)>
<Dialect cf (class mlir.dialects._cf_ops_gen._Dialect)>
<Dialect complex (class mlir.dialects._complex_ops_gen._Dialect)>
<Dialect func (class mlir.dialects._func_ops_gen._Dialect)>
<Dialect math (class mlir.dialects._math_ops_gen._Dialect)>
<Dialect memref (class mlir.dialects._memref_ops_gen._Dialect)>
<Dialect ml_program (class mlir.dialects._ml_program_ops_gen._Dialect)>
<Dialect pdl (class mlir.dialects._pdl_ops_gen._Dialect)>
<Dialect quant (class mlir._mlir_libs._mlir.ir.Dialect)>
<Dialect scf (class mlir.dialects._scf_ops_gen._Dialect)>
<Dialect shape (class mlir.dialects._shape_ops_gen._Dialect)>
<Dialect sparse_tensor (class mlir.dialects._sparse_tensor_ops_gen._Dialect)>
<Dialect tensor (class mlir.dialects._tensor_ops_gen._Dialect)>
<Dialect tosa (class mlir.dialects._tosa_ops_gen._Dialect)>
<Dialect vector (class mlir.dialects._vector_ops_gen._Dialect)>
```

# Updating

```shell
LLVM_BUILD_DIR=$(pwd)/build_mlir ./copy_from_mlir.sh
```
