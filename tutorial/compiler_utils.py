# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
import ast
import os
import re
import sys
import tempfile
from io import StringIO

from mlir._mlir_libs._mlir.ir import (
    Context,
    Location,
    InsertionPoint,
    Operation,
    Type,
    IntegerType,
)
from mlir._mlir_libs._mlir.passmanager import PassManager
from mlir.dialects import arith, memref

ONE_SHOT_BUFFERIZATION_PIPELINE = [
    "func.func(linalg-init-tensor-to-alloc-tensor)",
    "one-shot-bufferize",
    "func-bufferize",
    "arith-bufferize",
    "func.func(finalizing-bufferize)",
]

LOWERING_PIPELINE = [
    # Lower to LLVM
    "convert-scf-to-cf",
    # "func.func(refback-expand-ops-for-llvm)",
    "func.func(arith-expand)",
    "func.func(convert-math-to-llvm)",
    # Handle some complex mlir::math ops (e.g. atan2)
    "convert-math-to-libm",
    "convert-linalg-to-llvm",
    "convert-memref-to-llvm",
    "func.func(convert-arith-to-llvm)",
    "convert-func-to-llvm",
    "convert-cf-to-llvm",
    "reconcile-unrealized-casts",
]


def unrolling_pipeline(unroll_factor):
    return [
        # f"func.func(affine-loop-unroll{{unroll-full unroll-full-threshold={unroll_factor}}})",
        f"func.func(affine-loop-unroll{{unroll-factor={unroll_factor} unroll-up-to-factor=1}})"
        if unroll_factor < 100
        else f"func.func(affine-loop-unroll{{unroll-full unroll-full-threshold={unroll_factor}}})",
        # f"func.func(affine-loop-unroll{{unroll-factor={unroll_factor} unroll-up-to-factor=0}})",
        # f"func.func(affine-loop-unroll-jam{{unroll-jam-factor={unroll_factor}}})",
    ]


def run_pipeline_with_repro_report(module, pipeline: str, description: str = None):
    try:
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        asm_for_error_report = module.operation.get_asm(
            large_elements_limit=10, enable_debug_info=True
        )
        # Lower module in place to make it ready for compiler backends.
        with module.context:
            pm = PassManager.parse(pipeline)
            pm.run(module)
    except Exception as e:
        filename = os.path.join(tempfile.gettempdir(), "tmp.mlir")
        with open(filename, "w") as f:
            f.write(asm_for_error_report)
        debug_options = "-mlir-print-ir-after-all -mlir-disable-threading"
        description = description or f"tmp compile"

        message = f"""\
            {description} failed with the following diagnostics:
            {sys.stderr.getvalue()}

            For MLIR developers, the error can be reproduced with:
            $ mlir-opt -pass-pipeline='{pipeline}' {filename}
            Add '{debug_options}' to get the IR dump for debugging purpose.
            """
        trimmed_message = "\n".join([m.lstrip() for m in message.split("\n")])
        raise Exception(trimmed_message) from None
    finally:
        sys.stderr = original_stderr


def traverse_op_region_block_iterators(op, handler):
    for i, region in enumerate(op.regions):
        for j, block in enumerate(region):
            for k, child_op in enumerate(block):
                res = handler(child_op)
                if res is not None and isinstance(res, Exception):
                    return res
                res = traverse_op_region_block_iterators(child_op, handler)
                if res is not None and isinstance(res, Exception):
                    return res


def parse_attrs_to_dict(attrs):
    d = {}
    for named_attr in attrs:
        if named_attr.name in {"lpStartTime", "value"}:
            d[named_attr.name] = ast.literal_eval(
                str(named_attr.attr).split(":")[0].strip()
            )
        elif named_attr.name in {"opr"}:
            d[named_attr.name] = ast.literal_eval(str(named_attr.attr))
        else:
            d[named_attr.name] = ast.literal_eval(str(named_attr.attr).replace('"', ""))
    return d


def affine_store_load_forwarding(module):
    def handler(affine_for):
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

    with Context(), Location.unknown():
        traverse_op_region_block_iterators(
            module.operation,
            lambda op: handler(op) if op.operation.name == "affine.for" else None,
        )


def make_i32_int(x):
    return arith.ConstantOp(IntegerType.get_signless(32), x).result


def promote_alloc(module):
    def parse_memref_type_str(s):
        shape = list(map(int, re.findall(r"(\d)+x", s)))
        typ = re.findall(r"x([a-z].*)>", s)[0]
        typ = Type.parse(typ, Context())
        operand_segment_sizes = []
        for s in shape:
            segment = []
            for i in range(s):
                segment.append(add_dummy_value())
            operand_segment_sizes.append(segment)

        return operand_segment_sizes

    memrefs_to_erase = set()

    def handler(mem_alloc):
        res_type = mem_alloc.results.types[0]
        # operand_segment_sizes = parse_memref_type_str(str(res_type))
        entry_block = module.body.operations[0].regions[0].blocks[0]
        with InsertionPoint.at_block_begin(entry_block), Location.unknown():
            op = memref.AllocaOp(res_type, [], [])
            # print(op)
            mem_alloc.operation.replace_all_uses_with(op.operation)
            memrefs_to_erase.add(mem_alloc)

    with Context(), Location.unknown():
        traverse_op_region_block_iterators(
            module.operation,
            lambda op: handler(op) if op.operation.name == "memref.alloc" else None,
        )

        for memref_to_erase in memrefs_to_erase:
            memref_to_erase.operation.erase()

        traverse_op_region_block_iterators(
            module.operation,
            lambda op: op.operation.erase() or Exception("done")
            if op.operation.name == "memref.dealloc"
            else None,
        )


def add_dummy_value():
    return Operation.create(
        "custom.value", results=[IntegerType.get_signless(32)]
    ).result
