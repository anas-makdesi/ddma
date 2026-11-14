// pybind11 module ddma._core: compiled kernels (optional; ddma falls back to NumPy).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ddma_core.hpp"
#include "ddma_growth.hpp"
#include "ddma_synthesis.hpp"

#include <algorithm>
#include <type_traits>

namespace py = pybind11;
using arr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using arr8 = py::array_t<std::int8_t, py::array::c_style | py::array::forcecast>;

template <class T>
static py::tuple fit_sigma_t(arr Z, arr Y, std::vector<arr> edges, arr theta, arr8 signs) {
  if (Z.ndim() != 2 || Y.ndim() != 2 || Z.shape(0) != Y.shape(0))
    throw std::invalid_argument("Z must be (K, n) and Y (K, m)");
  const auto K = Z.shape(0), n = Z.shape(1), m = Y.shape(1);
  if (std::ptrdiff_t(edges.size()) != n) throw std::invalid_argument("one edge array per input");
  std::vector<const double*> e;
  std::vector<std::ptrdiff_t> ne, shape;
  for (auto& a : edges) {
    e.push_back(a.data());
    ne.push_back(a.size());
    shape.push_back(a.size() - 1);
  }
  shape.push_back(m);
  py::array_t<T> lower(shape), upper(shape);
  {
    py::gil_scoped_release release;
    ddma::fit_sigma<T>(Z.data(), Y.data(), K, n, m, e, ne, theta.data(), signs.data(),
                       lower.mutable_data(), upper.mutable_data());
  }
  return py::make_tuple(lower, upper);
}

static py::tuple fit_sigma(arr Z, arr Y, std::vector<arr> edges, arr theta, arr8 signs,
                           bool single) {
  return single ? fit_sigma_t<float>(Z, Y, edges, theta, signs)
                : fit_sigma_t<double>(Z, Y, edges, theta, signs);
}

// ---------------------------------------------------------------- synthesis
// first/last: (nq, np, n) C-order arrays of uint8/uint16/int32; enabled: (nq, np) uint8.
using arru8 = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
using shape_t = std::vector<std::ptrdiff_t>;

template <class T>
static ddma::PairBoxes<T> view(const py::array& first, const py::array& last, const arru8& en) {
  if (first.ndim() != 3 || last.ndim() != 3 || en.ndim() != 2)
    throw std::invalid_argument("first/last must be (nq, np, n) and enabled (nq, np)");
  for (int d = 0; d < 3; ++d)
    if (first.shape(d) != last.shape(d)) throw std::invalid_argument("first/last shapes differ");
  if (en.shape(0) != first.shape(0) || en.shape(1) != first.shape(1))
    throw std::invalid_argument("enabled must be (nq, np)");
  const auto nq = first.shape(0), np_ = first.shape(1), n = first.shape(2);
  ddma::PairBoxes<T> a{static_cast<const T*>(first.data()), static_cast<const T*>(last.data()),
                       en.data(), nq, np_, n, np_ * n, n, 1, np_, 1, {}};
  for (std::ptrdiff_t j = 0; j < n; ++j) a.dim.push_back(j);
  return a;
}

template <class T, class F>
static auto with_type(py::array first, py::array last, F f) {
  using A = py::array_t<T, py::array::c_style | py::array::forcecast>;
  A fa = A::ensure(first), la = A::ensure(last);
  return f(fa, la);
}

template <class F>
static auto dispatch(py::array first, py::array last, F f) {
  const char k = first.dtype().kind();
  const auto sz = first.dtype().itemsize();
  if (k == 'u' && sz == 1) return with_type<std::uint8_t>(first, last, f);
  if (k == 'u' && sz == 2) return with_type<std::uint16_t>(first, last, f);
  return with_type<std::int32_t>(first, last, f);
}

static py::array_t<std::uint8_t> box_all_in(py::array first, py::array last, arru8 en,
                                            shape_t shape, arru8 W) {
  py::array_t<std::uint8_t> good({en.shape(0), en.shape(1)});
  dispatch(first, last, [&](auto& fa, auto& la) {
    using T = typename std::remove_reference_t<decltype(fa)>::value_type;
    auto a = view<T>(fa, la, en);
    if (W.size() != a.nq) throw std::invalid_argument("W must have one entry per state");
    py::gil_scoped_release release;
    ddma::box_all_in<T>(a, shape, W.data(), good.mutable_data());
    return 0;
  });
  return good;
}

static py::tuple safety(py::array first, py::array last, arru8 en, shape_t shape, arru8 safe,
                        py::object fixed_obj) {
  py::array_t<std::uint8_t> W(safe.size()), allowed({en.shape(0), en.shape(1)});
  arru8 fixed;
  const std::uint8_t* fixed_ptr = nullptr;
  if (!fixed_obj.is_none()) {
    fixed = fixed_obj.cast<arru8>();
    if (fixed.size() != safe.size()) throw std::invalid_argument("fixed must match safe");
    fixed_ptr = fixed.data();
  }
  std::copy(safe.data(), safe.data() + safe.size(), W.mutable_data());
  std::ptrdiff_t rounds = dispatch(first, last, [&](auto& fa, auto& la) {
    using T = typename std::remove_reference_t<decltype(fa)>::value_type;
    auto a = view<T>(fa, la, en);
    if (safe.size() != a.nq) throw std::invalid_argument("safe must have one entry per state");
    py::gil_scoped_release release;
    return ddma::safety<T>(a, shape, W.mutable_data(), allowed.mutable_data(), 100000,
                           fixed_ptr);
  });
  return py::make_tuple(W, allowed, rounds);
}

static py::tuple reach_avoid(py::array first, py::array last, arru8 en, shape_t shape,
                             arru8 target, arru8 avoid) {
  const auto nq = en.shape(0);
  py::array_t<std::uint8_t> R(nq), allowed({en.shape(0), en.shape(1)});
  py::array_t<double> value(nq);
  std::ptrdiff_t rounds = dispatch(first, last, [&](auto& fa, auto& la) {
    using T = typename std::remove_reference_t<decltype(fa)>::value_type;
    auto a = view<T>(fa, la, en);
    if (target.size() != nq || avoid.size() != nq)
      throw std::invalid_argument("target/avoid must have one entry per state");
    py::gil_scoped_release release;
    return ddma::reach_avoid<T>(a, shape, target.data(), avoid.data(), R.mutable_data(),
                                value.mutable_data(), allowed.mutable_data());
  });
  return py::make_tuple(R, value, allowed, rounds);
}

// ---------------------------------------------------------------- growth cones
struct GridArgs {
  std::vector<const double*> e;
  std::vector<std::ptrdiff_t> ne;
};

static GridArgs grid_args(const arr& Z, const arr& y, const std::vector<arr>& edges,
                          const arr& gl, const arr& gh) {
  if (Z.ndim() != 2 || y.ndim() != 1 || Z.shape(0) != y.shape(0))
    throw std::invalid_argument("Z must be (K, n) and y (K,)");
  const auto n = Z.shape(1);
  if (std::ptrdiff_t(edges.size()) != n || gl.size() != n || gh.size() != n)
    throw std::invalid_argument("one edge array and one derivative bound per input");
  GridArgs g;
  for (auto& a : edges) {
    g.e.push_back(a.data());
    g.ne.push_back(a.size());
  }
  return g;
}

static py::tuple growth_vertices(arr Z, arr y, std::vector<arr> edges, arr gl, arr gh,
                                 double theta) {
  auto g = grid_args(Z, y, edges, gl, gh);
  py::array_t<double> lo(g.ne), up(g.ne);
  {
    py::gil_scoped_release release;
    ddma::growth_vertices(Z.data(), y.data(), Z.shape(0), Z.shape(1), g.e, g.ne, gl.data(),
                          gh.data(), theta, lo.mutable_data(), up.mutable_data());
  }
  return py::make_tuple(lo, up);
}

static py::tuple growth_local(arr Z, arr y, std::vector<arr> edges, arr gl, arr gh,
                              double theta, bool slab) {
  auto g = grid_args(Z, y, edges, gl, gh);
  std::vector<std::ptrdiff_t> shape;
  for (auto s : g.ne) shape.push_back(s - 1);
  py::array_t<double> lo(shape), up(shape);
  {
    py::gil_scoped_release release;
    ddma::growth_local(Z.data(), y.data(), Z.shape(0), Z.shape(1), g.e, g.ne, gl.data(),
                       gh.data(), theta, slab, lo.mutable_data(), up.mutable_data());
  }
  return py::make_tuple(lo, up);
}

PYBIND11_MODULE(_core, mod) {
  mod.doc() = "Compiled kernels for ddma";
  mod.def("fit_sigma", &fit_sigma, "Fixed-partition simulating map (lower, upper)",
          py::arg("Z"), py::arg("Y"), py::arg("edges"), py::arg("theta"), py::arg("signs"),
          py::arg("single") = false);
  mod.def("box_all_in", &box_all_in, "good[q, p]: p enabled at q, all successors in W");
  mod.def("safety", &safety, "Maximal controlled invariant (W, allowed, rounds)",
          py::arg("first"), py::arg("last"), py::arg("enabled"), py::arg("shape"),
          py::arg("safe"), py::arg("fixed") = py::none());
  mod.def("reach_avoid", &reach_avoid, "Reach-avoid (R, value, allowed, rounds)");
  mod.def("growth_vertices", &growth_vertices, "Growth-cone bounds at the grid vertices (lo, up)",
          py::arg("Z"), py::arg("y"), py::arg("edges"), py::arg("g_lo"), py::arg("g_hi"),
          py::arg("theta"));
  mod.def("set_num_threads", &ddma::set_num_threads,
          "Threads used by the kernels (0 = OpenMP default: OMP_NUM_THREADS or all cores)",
          py::arg("n"));
  mod.def("get_num_threads", &ddma::num_threads, "Threads the kernels will use");
  mod.def("openmp_available", &ddma::openmp_available, "Whether the kernel was built with OpenMP");
  mod.def("_set_scatter_mode", [](int m) { ddma::scatter_mode() = m; },
          "Internal (benchmarks/tests): 0 auto, 1 direct, 2 bucketed, 3 private");
  mod.def("_set_chunk_entries", [](std::ptrdiff_t c) { ddma::chunk_entries() = c; },
          "Internal (benchmarks/tests): values written per round in the bucketed mode");
  mod.def("growth_local", &growth_local, "Per-cell cone bounds, 'cell' or 'slab' (lo, up)",
          py::arg("Z"), py::arg("y"), py::arg("edges"), py::arg("g_lo"), py::arg("g_hi"),
          py::arg("theta"), py::arg("slab"));
}
