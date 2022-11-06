include(AddMLIRPython)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

################################################################################
# Structural groupings.
################################################################################

declare_mlir_python_sources(StandaloneMLIRPythonSources)
declare_mlir_python_sources(StandaloneMLIRPythonSources.Dialects
  ADD_TO_PARENT StandaloneMLIRPythonSources)

################################################################################
# Pure python sources and generated code
################################################################################

declare_mlir_python_sources(StandaloneMLIRPythonSources.Core
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  ADD_TO_PARENT StandaloneMLIRPythonSources
  SOURCES
    _mlir_libs/__init__.py
    ir.py
    passmanager.py
    dialects/_ods_common.py

    # The main _mlir module has submodules: include stubs from each.
    _mlir_libs/_mlir/__init__.pyi
    _mlir_libs/_mlir/ir.pyi
    _mlir_libs/_mlir/passmanager.pyi
)

declare_mlir_python_sources(StandaloneMLIRPythonSources.ExecutionEngine
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  ADD_TO_PARENT StandaloneMLIRPythonSources
  SOURCES
    execution_engine.py
    _mlir_libs/_mlirExecutionEngine.pyi
  SOURCES_GLOB
    runtime/*.py
)

declare_mlir_python_sources(StandaloneMLIRPythonCAPI.HeaderSources
  ROOT_DIR "${MLIR_SOURCE_DIR}/include"
  SOURCES_GLOB "mlir-c/*.h"
)

################################################################################
# Dialect bindings
################################################################################

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/AsyncOps.td
  SOURCES_GLOB dialects/async_dialect/*.py
  DIALECT_NAME async_dialect)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/BufferizationOps.td
  SOURCES
    dialects/bufferization.py
    dialects/_bufferization_ops_ext.py
  DIALECT_NAME bufferization)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/BuiltinOps.td
  SOURCES
    dialects/builtin.py
    dialects/_builtin_ops_ext.py
  DIALECT_NAME builtin)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/ComplexOps.td
  SOURCES
    dialects/complex.py
  DIALECT_NAME complex)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/ControlFlowOps.td
  SOURCES
    dialects/cf.py
  DIALECT_NAME cf)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/FuncOps.td
  SOURCES
    dialects/func.py
    dialects/_func_ops_ext.py
  DIALECT_NAME func)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/GPUOps.td
  SOURCES_GLOB dialects/gpu/*.py
  DIALECT_NAME gpu)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/LinalgOps.td
  SOURCES
    dialects/_linalg_ops_ext.py
  SOURCES_GLOB
    dialects/linalg/*.py
  DIALECT_NAME linalg)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/TransformOps.td
  SOURCES
    dialects/_transform_ops_ext.py
    dialects/transform/__init__.py
  DIALECT_NAME transform)

declare_mlir_dialect_extension_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/SCFLoopTransformOps.td
  SOURCES
    dialects/_loop_transform_ops_ext.py
    dialects/transform/loop.py
  DIALECT_NAME transform
  EXTENSION_NAME loop_transform)

declare_mlir_dialect_extension_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/LinalgStructuredTransformOps.td
  SOURCES
    dialects/_structured_transform_ops_ext.py
    dialects/transform/structured.py
  DIALECT_NAME transform
  EXTENSION_NAME structured_transform)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/MathOps.td
  SOURCES dialects/math.py
  DIALECT_NAME math)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/ArithmeticOps.td
  SOURCES
    dialects/arith.py
    dialects/_arith_ops_ext.py
  DIALECT_NAME arith)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/MemRefOps.td
  SOURCES
    dialects/memref.py
    dialects/_memref_ops_ext.py
  DIALECT_NAME memref)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/MLProgramOps.td
  SOURCES
    dialects/ml_program.py
    dialects/_ml_program_ops_ext.py
  DIALECT_NAME ml_program)

declare_mlir_python_sources(
  StandaloneMLIRPythonSources.Dialects.quant
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  SOURCES
    dialects/quant.py
    _mlir_libs/_mlir/dialects/quant.pyi)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/PDLOps.td
  SOURCES
    dialects/pdl.py
    dialects/_pdl_ops_ext.py
    _mlir_libs/_mlir/dialects/pdl.pyi
  DIALECT_NAME pdl)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/SCFOps.td
  SOURCES
    dialects/scf.py
    dialects/_scf_ops_ext.py
  DIALECT_NAME scf)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/ShapeOps.td
  SOURCES dialects/shape.py
  DIALECT_NAME shape)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/SparseTensorOps.td
  SOURCES dialects/sparse_tensor.py
  DIALECT_NAME sparse_tensor)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/TensorOps.td
  SOURCES dialects/tensor.py
  DIALECT_NAME tensor)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/TosaOps.td
  SOURCES dialects/tosa.py
  DIALECT_NAME tosa)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  # TD_FILE dialects/VectorOps.td
  SOURCES dialects/vector.py
  DIALECT_NAME vector)
