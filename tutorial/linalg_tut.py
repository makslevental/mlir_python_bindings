from mlir._mlir_libs._mlir.ir import (
    Context,
    Location,
    Module,
    F32Type,
    InsertionPoint,
    RankedTensorType,
    OpView,
    FunctionType,
    SymbolTable,
)
from mlir.dialects import func, arith, linalg

from compiler_utils import (
    run_pipeline_with_repro_report,
    LOWERING_PIPELINE,
    ONE_SHOT_BUFFERIZATION_PIPELINE,
    unrolling_pipeline,
    traverse_op_region_block_iterators,
)

with Context(), Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    with InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            RankedTensorType.get((4, 16), f32), RankedTensorType.get((16, 8), f32)
        )
        def demo(arg0, arg1):
            one = arith.ConstantOp(F32Type.get(), 1.0)
            lhs = linalg.fill(one, outs=[arg0])
            rhs = linalg.fill(one, outs=[arg1])
            init = linalg.InitTensorOp([4, 8], f32)
            return linalg.matmul(lhs, rhs, outs=init)


print(module)

pipeline = ",".join(
    ONE_SHOT_BUFFERIZATION_PIPELINE + LOWERING_PIPELINE + unrolling_pipeline(10)
)

run_pipeline_with_repro_report(module, pipeline)

# print(module)


def affine_store_load_forwarding(affine_for):
    stores_to_forward = set()
    loads_to_delete = set()
    ops = list(affine_for.operation.regions[0].blocks[0].operations)
    for i, op in reversed(list(enumerate(ops))):
        if op.operation.name == "affine.load":
            loads_to_delete.add(op)
            buffer, *idxs = op.operands
            j = i - 1
            while j >= 0:
                other_op = ops[j]
                if other_op.operation.name == "affine.store":
                    val, other_buffer, *other_idxs = other_op.operands
                    if idxs == other_idxs:
                        stores_to_forward.add((val, other_op, op))
                        break
                j -= 1
    if stores_to_forward:
        for i, (val, store, load) in enumerate(stores_to_forward):
            load.operation.replace_all_uses_with(val.owner)
            store.operation.erase()
            load.operation.erase()


def handler(mlir_op):
    if mlir_op.operation.name == "affine.for":
        affine_store_load_forwarding(mlir_op)


with Context(), Location.unknown():
    traverse_op_region_block_iterators(module.operation, handler)

print(module)
