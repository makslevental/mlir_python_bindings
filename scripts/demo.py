import inspect
from typing import Optional, Sequence
import ast

from mlir._mlir_libs._mlir.ir import (
    InsertionPoint,
    IndexType,
    Location,
    F32Type,
    MemRefType,
    Type, FunctionType, Value, OpView, Operation, TypeAttr, FlatSymbolRefAttr,
)
from mlir._mlir_libs._mlir.passmanager import PassManager
from mlir.dialects import func, scf, arith
from mlir.dialects._func_ops_gen import FuncOp
from mlir.ir import Context
from mlir.dialects import linalg, affine_ as affine
import mlir.dialects.memref as memref


def from_py_func(
    *inputs: Type, results: Optional[Sequence[Type]] = None, name: Optional[str] = None
):
    def decorator(f):
        # Introspect the callable for optional features.
        sig = inspect.signature(f)
        has_arg_func_op = False
        for param in sig.parameters.values():
            if param.kind == param.VAR_KEYWORD:
                has_arg_func_op = True
            if param.name == "func_op" and (
                param.kind == param.POSITIONAL_OR_KEYWORD
                or param.kind == param.KEYWORD_ONLY
            ):
                has_arg_func_op = True

        # Emit the FuncOp.
        implicit_return = results is None
        symbol_name = name or f.__name__
        function_type = FunctionType.get(
            inputs=inputs, results=[] if implicit_return else results
        )
        func_op = FuncOp(name=symbol_name, type=function_type)
        with InsertionPoint(func_op.add_entry_block()):
            func_args = func_op.entry_block.arguments
            func_kwargs = {}
            if has_arg_func_op:
                func_kwargs["func_op"] = func_op
            return_values = f(*func_args, **func_kwargs)
            if not implicit_return:
                return_types = list(results)
                assert return_values is None, (
                    "Capturing a python function with explicit `results=` "
                    "requires that the wrapped function returns None."
                )
            else:
                # Coerce return values, add ReturnOp and rewrite func type.
                if return_values is None:
                    return_values = []
                elif isinstance(return_values, tuple):
                    return_values = list(return_values)
                elif isinstance(return_values, Value):
                    # Returning a single value is fine, coerce it into a list.
                    return_values = [return_values]
                elif isinstance(return_values, OpView):
                    # Returning a single operation is fine, coerce its results a list.
                    return_values = return_values.operation.results
                elif isinstance(return_values, Operation):
                    # Returning a single operation is fine, coerce its results a list.
                    return_values = return_values.results
                else:
                    return_values = list(return_values)
                func.ReturnOp(return_values)
                # Recompute the function type.
                return_types = [v.type for v in return_values]
                function_type = FunctionType.get(inputs=inputs, results=return_types)
                func_op.attributes["function_type"] = TypeAttr.get(function_type)

        def emit_call_op(*call_args):
            call_op = func.CallOp(
                return_types, FlatSymbolRefAttr.get(symbol_name), call_args
            )
            if return_types is None:
                return None
            elif len(return_types) == 1:
                return call_op.result
            else:
                return call_op.results

        wrapped = emit_call_op
        wrapped.__name__ = f.__name__
        wrapped.func_op = func_op
        return wrapped

    return decorator


def icst(x):
    return arith.ConstantOp.create_index(x)


def fcst(x):
    return arith.ConstantOp(f32, x)


with Context() as ctx, Location.unknown():
    f32 = F32Type.get()
    index_type = IndexType.get()

    @from_py_func(MemRefType.get((12, 12), f32))
    def simple_loop(mem):
        lb = icst(0)
        ub = icst(42)
        step = icst(2)
        tens = linalg.InitTensorOp([3, 4], f32)
        loop1 = scf.ForOp(lb, ub, step, [])
        with InsertionPoint(loop1.body):
            loop2 = scf.ForOp(lb, ub, step, [])
            with InsertionPoint(loop2.body):
                loop3 = scf.ForOp(lb, ub, step, [])
                with InsertionPoint(loop3.body):
                    f = fcst(1.14)
                    memref.StoreOp(
                        f, mem, [loop2.induction_variable, loop3.induction_variable]
                    )
                    scf.YieldOp([])
                scf.YieldOp([])
            scf.YieldOp([])
        return

    # fn = simple_loop()
    simple_loop.func_op.print()

    pass_man = PassManager.parse('async-to-async-runtime')