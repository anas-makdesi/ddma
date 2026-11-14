"""Builds the optional compiled kernel ddma._core (C++/pybind11).

If no C++ compiler is available the build is skipped and ddma uses its NumPy
implementation, which gives identical results (only slower).
"""

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
            extra_compile_args=["-O3"],
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
        try:
            super().build_extension(ext)
        except Exception as exc:  # pragma: no cover
            print(f"WARNING: could not build {ext.name} ({exc}); using the NumPy backend.")


setup(ext_modules=ext_modules, cmdclass={"build_ext": OptionalBuildExt})
