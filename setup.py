"""Builds the optional compiled kernel ddma._core (C++/pybind11).

If no C++ compiler is available the build is skipped and ddma uses its NumPy
implementation, which gives identical results (only slower).

The kernel is built with OpenMP (threads) when the compiler supports it; otherwise it is
built without, and runs on one thread with the same results. -ffp-contract=off keeps the
floating-point operations exactly as written (no fused multiply-add), so results do not
depend on how the compiler splits loops.
"""

import os

from setuptools import setup
from setuptools.command.build_ext import build_ext

try:
    from pybind11.setup_helpers import Pybind11Extension

    ext_modules = [
        Pybind11Extension(
            "ddma._core",
            ["cpp/python_binding.cpp"],
            include_dirs=["cpp"],
            cxx_std=17,
        )
    ]
except ImportError:  # pragma: no cover
    ext_modules = []


class OptionalBuildExt(build_ext):
    def run(self):
        try:
            super().run()
        except Exception as exc:  # pragma: no cover
            print(f"WARNING: compiled kernel not built ({exc}); using the NumPy backend.")

    def build_extension(self, ext):
        ctype = self.compiler.compiler_type
        if ctype == "msvc":
            base, omp, omp_link = ["/O2", "/fp:precise"], ["/openmp"], []
        else:
            base = ["-O3", "-ffp-contract=off", "-Wno-unknown-pragmas"]
            omp, omp_link = ["-fopenmp"], ["-fopenmp"]
        compile_args, link_args = list(ext.extra_compile_args), list(ext.extra_link_args)
        if os.environ.get("DDMA_NO_OPENMP"):
            omp, omp_link = [], []
        try:  # with OpenMP
            ext.extra_compile_args = compile_args + base + omp
            ext.extra_link_args = link_args + omp_link
            super().build_extension(ext)
            return
        except Exception as exc:  # pragma: no cover
            if not omp:
                print(f"WARNING: could not build {ext.name} ({exc}); using the NumPy backend.")
                return
            print(f"NOTE: building {ext.name} without OpenMP ({exc}); kernels run on one thread.")
        try:  # without OpenMP
            ext.extra_compile_args = compile_args + base
            ext.extra_link_args = link_args
            super().build_extension(ext)
        except Exception as exc:  # pragma: no cover
            print(f"WARNING: could not build {ext.name} ({exc}); using the NumPy backend.")


setup(ext_modules=ext_modules, cmdclass={"build_ext": OptionalBuildExt})
