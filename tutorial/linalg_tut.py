import numpy as np
from mlir._mlir_libs._mlir.ir import (
    Context,
    Location,
    Module,
    InsertionPoint,
    RankedTensorType,
    F64Type,
)
from mlir.dialects import func, linalg

from refbackend import (
    RefBackendLinalgOnTensorsBackend,
)


def test_matmul():
    with Context(), Location.unknown():
        module = Module.create()
        f32 = F64Type.get()
        with InsertionPoint(module.body):

            @func.FuncOp.from_py_func(
                RankedTensorType.get((4, 16), f32),
                RankedTensorType.get((16, 8), f32),
            )
            def main(lhs, rhs):
                out = linalg.InitTensorOp([4, 8], f32)
                return linalg.matmul(lhs, rhs, outs=[out])

    print(module)
    module = RefBackendLinalgOnTensorsBackend.compile(module)
    print(module)

    invoker = RefBackendLinalgOnTensorsBackend.load(module)
    argument1 = np.random.uniform(low=0.0, high=5, size=(4, 16))
    argument2 = np.random.uniform(low=0.0, high=5, size=(16, 8))
    res = invoker.main(argument1, argument2)
    print(res)

    print(argument1 @ argument2)


test_matmul()
