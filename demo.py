from mlir import ir

ctx = ir.Context()

dialects = """arith
bufferization
builtin
cf
complex
func
math
memref
ml_program
pdl
quant
scf
shape
sparse_tensor
tensor
tosa
vector""".splitlines()
for dialect in dialects:
    print(ctx.dialects[dialect])
