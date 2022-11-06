#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import subprocess
from distutils.command.build import build as _build

from setuptools import find_namespace_packages, setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


# Build phase discovery is unreliable. Just tell it what phases to run.
class CustomBuild(_build):
    def run(self):
        self.run_command("build_py")
        self.run_command("build_ext")
        self.run_command("build_scripts")


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


def do_tblgen(llvm_build_dir, here, cmake_install_dir):
    td_files = """dialects/AsyncOps.td
    dialects/BufferizationOps.td
    dialects/BuiltinOps.td
    dialects/ComplexOps.td
    dialects/ControlFlowOps.td
    dialects/FuncOps.td
    dialects/GPUOps.td
    dialects/LinalgOps.td
    dialects/TransformOps.td
    dialects/SCFLoopTransformOps.td
    dialects/LinalgStructuredTransformOps.td
    dialects/MathOps.td
    dialects/ArithOps.td
    dialects/MemRefOps.td
    dialects/MLProgramOps.td
    dialects/PDLOps.td
    dialects/SCFOps.td
    dialects/ShapeOps.td
    dialects/SparseTensorOps.td
    dialects/TensorOps.td
    dialects/TosaOps.td
    dialects/VectorOps.td""".split()

    dialect_names = """async_dialect
    bufferization
    builtin
    complex
    cf
    func
    gpu
    linalg
    transform
    transform
    transform
    math
    arith
    memref
    ml_program
    pdl
    scf
    shape
    sparse_tensor
    tensor
    tosa
    vector
    """.split()

    for td_file, dialect_name in zip(td_files, dialect_names):
        subprocess.check_call(
            [
                f"{llvm_build_dir}/bin/mlir-tblgen",
                "-gen-python-op-bindings",
                f"-bind-dialect={dialect_name}",
                "-I",
                "cpp/include",
                "-I",
                f"{llvm_build_dir}/include",
                f"python/mlir/{td_file}",
                "-o",
                f"{cmake_install_dir}/python_packages/mlir_core/mlir/dialects/_{dialect_name}_ops_gen.py",
            ],
            cwd=here,
        )


class CMakeBuild(build_py):
    def run(self):
        target_dir = self.build_lib
        here = os.path.abspath(os.path.dirname(__file__))
        cmake_build_dir = os.path.join(here, "build")
        cmake_install_dir = os.path.join(cmake_build_dir, "install")
        llvm_install_dir = os.getenv("LLVM_INSTALL_DIR")
        cmake_args = [
            "-DCMAKE_BUILD_TYPE=Release",  # not used on MSVC, but no harm
            f"-DCMAKE_INSTALL_PREFIX={cmake_install_dir}",
            f"-DCMAKE_PREFIX_PATH={llvm_install_dir}",
        ]

        build_args = []
        os.makedirs(cmake_build_dir, exist_ok=True)
        if os.path.exists(cmake_install_dir):
            shutil.rmtree(cmake_install_dir)
        cmake_cache_file = os.path.join(cmake_build_dir, "CMakeCache.txt")
        if os.path.exists(cmake_cache_file):
            os.remove(cmake_cache_file)
        subprocess.check_call(
            ["cmake", "-G Ninja", here] + cmake_args, cwd=cmake_build_dir
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"] + build_args,
            cwd=cmake_build_dir,
        )
        do_tblgen(llvm_install_dir, here, cmake_install_dir)
        shutil.copytree(
            os.path.join(cmake_install_dir, "python_packages", "mlir_core"),
            target_dir,
            symlinks=False,
            dirs_exist_ok=True,
        )


class NoopBuildExtension(build_ext):
    def build_extension(self, ext):
        pass


packages = find_namespace_packages(
    where="python",
    include=[
        "mlir",
        "mlir.*",
    ],
)

setup(
    name="mlir_python_bindings",
    include_package_data=True,
    ext_modules=[
        CMakeExtension("mlir._mlir_libs._mlir"),
    ],
    cmdclass={
        "build": CustomBuild,
        "built_ext": NoopBuildExtension,
        "build_py": CMakeBuild,
    },
    zip_safe=False,
    packages=packages,
    package_dir={"": "python"},
)
