"""Builds the optional compiled kernel ddma._core (C++/pybind11).

If no C++ compiler is available the build is skipped and ddma uses its NumPy
implementation, which gives identical results (only slower).

The kernel is built with OpenMP (threads) when the compiler supports it: GCC and Clang
(-fopenmp), Microsoft's compiler (/openmp), and Apple's compiler when Homebrew's libomp is
installed (brew install libomp). Otherwise it is built without, and runs on one thread with
the same results. -ffp-contract=off keeps the
floating-point operations exactly as written (no fused multiply-add), so results do not
depend on how the compiler splits loops.
"""

import os
import subprocess
import sys

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


def _libomp_prefix():
    """Folder of the OpenMP library for Apple's compiler (Homebrew ``libomp``), or None.

    ``LIBOMP_PREFIX`` overrides the search; otherwise ``brew --prefix libomp`` and the usual
    Homebrew folders (Apple Silicon, Intel) are tried.
    """
    candidates = [os.environ.get("LIBOMP_PREFIX")]
    try:
        out = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True,
                             timeout=20)
        if out.returncode == 0:
            candidates.append(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    candidates += ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "include", "omp.h")):
            return c
    return None


class OptionalBuildExt(build_ext):
    def run(self):
        try:
            super().run()
        except Exception as exc:  # pragma: no cover
            print(f"WARNING: compiled kernel not built ({exc}); using the NumPy backend.")

    def _openmp_flag_sets(self):
        """(compile flags, link flags) to try in order, best first; the last one has no
        OpenMP. Each compiler switches OpenMP on differently; the C++ code is the same."""
        if os.environ.get("DDMA_NO_OPENMP"):
            return [([], [])]
        if self.compiler.compiler_type == "msvc":  # Microsoft's compiler (Windows)
            return [(["/openmp"], []), ([], [])]
        sets = [(["-fopenmp"], ["-fopenmp"])]  # GCC, and Clang with OpenMP (Linux, ...)
        if sys.platform == "darwin":  # Apple's compiler: needs Homebrew's libomp
            prefix = _libomp_prefix()
            if prefix:
                sets.append((["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"],
                             [f"-L{prefix}/lib", "-lomp", f"-Wl,-rpath,{prefix}/lib"]))
        return sets + [([], [])]

    def build_extension(self, ext):
        if self.compiler.compiler_type == "msvc":
            base = ["/O2", "/fp:precise"]
        else:
            base = ["-O3", "-ffp-contract=off", "-Wno-unknown-pragmas"]
        compile_args, link_args = list(ext.extra_compile_args), list(ext.extra_link_args)
        sets = self._openmp_flag_sets()
        for i, (omp, omp_link) in enumerate(sets):
            ext.extra_compile_args = compile_args + base + omp
            ext.extra_link_args = link_args + omp_link
            try:
                super().build_extension(ext)
                if not omp and len(sets) > 1:
                    hint = (" (on macOS: brew install libomp, then reinstall)"
                            if sys.platform == "darwin" else "")
                    print(f"NOTE: {ext.name} built without OpenMP; kernels run on one "
                          f"thread{hint}.")
                return
            except Exception as exc:  # pragma: no cover
                if i == len(sets) - 1:
                    print(f"WARNING: could not build {ext.name} ({exc}); "
                          "using the NumPy backend.")


setup(ext_modules=ext_modules, cmdclass={"build_ext": OptionalBuildExt})
