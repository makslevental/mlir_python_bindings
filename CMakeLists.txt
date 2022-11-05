cmake_minimum_required(VERSION 3.13.4)

if (POLICY CMP0068)
    cmake_policy(SET CMP0068 NEW)
    set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif ()

if (POLICY CMP0075)
    cmake_policy(SET CMP0075 NEW)
endif ()

if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif ()

if (POLICY CMP0116)
    cmake_policy(SET CMP0116 NEW)
endif ()

project(mlir-python-bindings LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

##################################### Bindings

include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()
mlir_detect_pybind11_install()

set(MY_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}) # --src-root
include(AddMLIRPython)
add_subdirectory(python)
add_subdirectory(cpp)

add_mlir_python_common_capi_library(StandaloneMLIRPythonCAPI
        INSTALL_COMPONENT StandaloneMLIRPythonModules
        INSTALL_DESTINATION python_packages/mlir_core/mlir/_mlir_libs
        OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/mlir_core/mlir/_mlir_libs"
        RELATIVE_INSTALL_ROOT "../../../.."
        DECLARED_HEADERS
        StandaloneMLIRPythonCAPI.HeaderSources
        DECLARED_SOURCES
        StandaloneMLIRPythonSources
        StandaloneMLIRPythonExtension.RegisterEverything
        )

add_mlir_python_modules(StandaloneMLIRPythonModules
        ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/mlir_core/mlir"
        INSTALL_PREFIX "python_packages/mlir_core/mlir"
        DECLARED_SOURCES
        StandaloneMLIRPythonSources
        StandaloneMLIRPythonExtension.RegisterEverything
        COMMON_CAPI_LINK_LIBS
        StandaloneMLIRPythonCAPI
        )
