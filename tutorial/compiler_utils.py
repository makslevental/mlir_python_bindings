# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
import ast
from io import StringIO
import os
import sys
import tempfile

from mlir._mlir_libs._mlir.passmanager import PassManager

ONE_SHOT_BUFFERIZATION_PIPELINE = [
    "func.func(linalg-init-tensor-to-alloc-tensor)",
    "one-shot-bufferize",
    "func-bufferize",
    "arith-bufferize",
    "func.func(finalizing-bufferize)",
]

LOWERING_PIPELINE = ["func.func(convert-linalg-to-affine-loops)"]


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

            For Torch-MLIR developers, the error can be reproduced with:
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
                handler(child_op)
                traverse_op_region_block_iterators(child_op, handler)


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
