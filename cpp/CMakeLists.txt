################################################################################
# Python extensions.
# The sources for these are all in lib/Bindings/Python, but since they have to
# be rebuilt for each package and integrate with the source setup here, we
# just reference them here instead of having ordered, cross package target
# dependencies.
################################################################################

set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")
include_directories("include")
declare_mlir_python_extension(StandaloneMLIRPythonExtension.Core
  MODULE_NAME _mlir
  ADD_TO_PARENT StandaloneMLIRPythonSources.Core
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    MainModule.cpp
    IRAffine.cpp
    IRAttributes.cpp
    IRCore.cpp
    IRInterfaces.cpp
    IRModule.cpp
    IRTypes.cpp
    PybindUtils.cpp
    Pass.cpp

    # Headers must be included explicitly so they are installed.
    Globals.h
    IRModule.h
    Pass.h
    PybindUtils.h
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIDebug
    MLIRCAPIIR
    MLIRCAPIInterfaces

    # Dialects
    MLIRCAPIFunc
)

# This extension exposes an API to register all dialects, extensions, and passes
# packaged in upstream MLIR and it is used for the upstream "mlir" Python
# package. Downstreams will likely want to provide their own and not depend
# on this one, since it links in the world.
# Note that this is not added to any top-level source target for transitive
# inclusion: It must be included explicitly by downstreams if desired. Note that
# this has a very large impact on what gets built/packaged.
declare_mlir_python_extension(StandaloneMLIRPythonExtension.RegisterEverything
  MODULE_NAME _mlirRegisterEverything
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    RegisterEverything.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIConversion
    MLIRCAPITransforms
    MLIRCAPIRegisterEverything
)

declare_mlir_python_extension(StandaloneMLIRPythonExtension.Dialects.Linalg.Pybind
  MODULE_NAME _mlirDialectsLinalg
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects.linalg
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectLinalg.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPILinalg
)

declare_mlir_python_extension(StandaloneMLIRPythonExtension.Dialects.Quant.Pybind
  MODULE_NAME _mlirDialectsQuant
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects.quant
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectQuant.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIQuant
)

declare_mlir_python_extension(StandaloneMLIRPythonExtension.Dialects.PDL.Pybind
  MODULE_NAME _mlirDialectsPDL
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects.pdl
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectPDL.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIPDL
)

declare_mlir_python_extension(StandaloneMLIRPythonExtension.Dialects.SparseTensor.Pybind
  MODULE_NAME _mlirDialectsSparseTensor
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects.sparse_tensor
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectSparseTensor.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPISparseTensor
)

declare_mlir_python_extension(StandaloneMLIRPythonExtension.AsyncDialectPasses
  MODULE_NAME _mlirAsyncPasses
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects.async_dialect
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    AsyncPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIAsync
)

if(MLIR_ENABLE_EXECUTION_ENGINE)
  declare_mlir_python_extension(StandaloneMLIRPythonExtension.ExecutionEngine
    MODULE_NAME _mlirExecutionEngine
    ADD_TO_PARENT StandaloneMLIRPythonSources.ExecutionEngine
    ROOT_DIR "${PYTHON_SOURCE_DIR}"
    SOURCES
      ExecutionEngineModule.cpp
    PRIVATE_LINK_LIBS
      LLVMSupport
    EMBED_CAPI_LINK_LIBS
      MLIRCAPIExecutionEngine
  )
endif()

declare_mlir_python_extension(StandaloneMLIRPythonExtension.GPUDialectPasses
  MODULE_NAME _mlirGPUPasses
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects.gpu
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    GPUPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIGPU
)

declare_mlir_python_extension(StandaloneMLIRPythonExtension.LinalgPasses
  MODULE_NAME _mlirLinalgPasses
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects.linalg
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    LinalgPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPILinalg
)

declare_mlir_python_extension(StandaloneMLIRPythonExtension.SparseTensorDialectPasses
  MODULE_NAME _mlirSparseTensorPasses
  ADD_TO_PARENT StandaloneMLIRPythonSources.Dialects.sparse_tensor
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    SparseTensorPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPISparseTensor
)
