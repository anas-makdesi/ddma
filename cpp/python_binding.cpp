// pybind11 module ddma._core: compiled kernels (optional; ddma falls back to NumPy).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ddma_abstraction.hpp"
#include "ddma_core.hpp"
#include "ddma_growth.hpp"
#include "ddma_monitor.hpp"
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

// ---------------------------------------------------------------- abstraction
// The spec is a dict built by ddma.abstraction (see ddma_abstraction.hpp):
//   mode, n, nq, np, edges, gathers [(offq, offp, lo, hi, outs, target)], gm, gp, g
//   (each None or (qmid, qrad, pmid | None, prad | None)), g_lo, g_hi, box_lo, box_hi
//   (None or per-pair arrays), assumption (None or (a_lo, a_hi, c_lo, c_hi)).
using arri64 = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

namespace {
struct AbsHolder {
  ddma::AbsSpec s;
  std::vector<py::object> keep;  // arrays the spec points into
};
}  // namespace

static const double* keep_double(AbsHolder& h, py::handle o) {
  arr a = arr::ensure(o);
  if (!a) throw std::invalid_argument("expected a float array");
  h.keep.push_back(a);
  return a.data();
}

static ddma::AbsAffine affine_spec(AbsHolder& h, py::handle o) {
  ddma::AbsAffine a;
  if (o.is_none()) return a;
  auto t = o.cast<py::tuple>();
  a.qmid = keep_double(h, t[0]);
  a.qrad = keep_double(h, t[1]);
  if (!t[2].is_none()) {
    a.pmid = keep_double(h, t[2]);
    a.prad = keep_double(h, t[3]);
  }
  return a;
}

static void abs_spec(AbsHolder& h, const py::dict& d) {
  auto& s = h.s;
  s.mode = d["mode"].cast<int>();
  s.n = d["n"].cast<int>();
  s.nq = d["nq"].cast<std::ptrdiff_t>();
  s.np = d["np"].cast<std::ptrdiff_t>();
  for (auto e : d["edges"].cast<py::list>()) {
    s.edges.push_back(keep_double(h, e));
    s.ne.push_back(py::len(e));
  }
  if (std::ptrdiff_t(s.edges.size()) != s.n) throw std::invalid_argument("one edge array per state");
  for (auto item : d["gathers"].cast<py::list>()) {
    auto t = item.cast<py::tuple>();
    ddma::AbsGather g;
    arri64 oq = arri64::ensure(t[0]), op = arri64::ensure(t[1]);
    if (oq.size() != s.nq || op.size() != s.np)
      throw std::invalid_argument("offsets must have one entry per state / input cell");
    h.keep.push_back(oq);
    h.keep.push_back(op);
    g.offq = oq.data();
    g.offp = op.data();
    for (int k = 2; k <= 3; ++k) {
      py::array a = py::array::ensure(t[k], py::array::c_style);
      const char kind = a.dtype().kind();
      const auto sz = a.dtype().itemsize();
      if (kind != 'f' || (sz != 4 && sz != 8)) throw std::invalid_argument("bounds must be float32/64");
      if (k == 3 && (sz == 4) != g.f32) throw std::invalid_argument("lower/upper dtypes differ");
      g.f32 = sz == 4;
      h.keep.push_back(a);
      (k == 2 ? g.lo : g.hi) = a.data();
    }
    g.outs = t[4].cast<std::vector<int>>();
    g.m = int(g.outs.size());
    g.target = t[5].cast<int>();
    for (int o : g.outs)
      if (o < 0 || o >= s.n) throw std::invalid_argument("output index out of range");
    s.gathers.push_back(std::move(g));
  }
  s.gm = affine_spec(h, d["gm"]);
  s.gp = affine_spec(h, d["gp"]);
  s.g = affine_spec(h, d["g"]);
  if (!d["g_lo"].is_none()) {
    s.g_lo = keep_double(h, d["g_lo"]);
    s.g_hi = keep_double(h, d["g_hi"]);
  }
  if (!d["box_lo"].is_none()) {
    s.box_lo = keep_double(h, d["box_lo"]);
    s.box_hi = keep_double(h, d["box_hi"]);
  }
  if (s.mode == 1 && !(s.gm.set() && s.gp.set())) throw std::invalid_argument("model needs gm, gp");
  if (s.mode == 2 && !s.box_lo) throw std::invalid_argument("mode 2 needs box_lo, box_hi");
  if (!d["assumption"].is_none()) {
    auto t = d["assumption"].cast<py::tuple>();
    s.assume = true;
    s.a_lo = t[0].cast<std::vector<double>>();
    s.a_hi = t[1].cast<std::vector<double>>();
    s.c_lo = t[2].cast<std::vector<std::ptrdiff_t>>();
    s.c_hi = t[3].cast<std::vector<std::ptrdiff_t>>();
  }
  for (auto k : s.ne)
    if (k < 2) throw std::invalid_argument("every axis needs at least one cell");
  s.prepare();
}

// Fill the rows qs of enabled (nq, np) and first/last (nq, np, n) in place.
static void abstraction_rows(py::dict spec, arri64 qs, py::array enabled, py::array first,
                             py::array last) {
  AbsHolder h;
  abs_spec(h, spec);
  const auto& s = h.s;
  auto check = [&](const py::array& a, int nd) {
    if (!(a.flags() & py::array::c_style) || !a.writeable() || a.ndim() != nd ||
        a.shape(0) != s.nq || a.shape(1) != s.np || (nd == 3 && a.shape(2) != s.n))
      throw std::invalid_argument("outputs must be writeable C-ordered (nq, np[, n]) arrays");
  };
  check(enabled, 2);
  check(first, 3);
  check(last, 3);
  if (enabled.dtype().itemsize() != 1) throw std::invalid_argument("enabled must be bool");
  if (!first.dtype().is(last.dtype())) throw std::invalid_argument("first/last dtypes differ");
  for (std::ptrdiff_t i = 0; i < qs.size(); ++i)
    if (qs.data()[i] < 0 || qs.data()[i] >= s.nq) throw std::invalid_argument("state index out of range");
  if (s.g_lo && std::ptrdiff_t(py::len(spec["g_lo"])) != qs.size() * s.np)
    throw std::invalid_argument("g_lo must have one row per pair");
  if (s.box_lo && std::ptrdiff_t(py::len(spec["box_lo"])) != qs.size() * s.np)
    throw std::invalid_argument("box_lo must have one row per pair");
  auto* en = static_cast<std::uint8_t*>(enabled.mutable_data());
  const char k = first.dtype().kind();
  const auto sz = first.dtype().itemsize();
  py::gil_scoped_release release;
  if (k == 'u' && sz == 1)
    ddma::abstraction_rows(s, qs.data(), qs.size(), en, static_cast<std::uint8_t*>(first.mutable_data()),
                           static_cast<std::uint8_t*>(last.mutable_data()));
  else if (k == 'u' && sz == 2)
    ddma::abstraction_rows(s, qs.data(), qs.size(), en, static_cast<std::uint16_t*>(first.mutable_data()),
                           static_cast<std::uint16_t*>(last.mutable_data()));
  else if (k == 'i' && sz == 4)
    ddma::abstraction_rows(s, qs.data(), qs.size(), en, static_cast<std::int32_t*>(first.mutable_data()),
                           static_cast<std::int32_t*>(last.mutable_data()));
  else
    throw std::invalid_argument("first/last must be uint8, uint16 or int32");
}

// Transitions of the pairs (Q[k], P[k]): (enabled (k,) bool, first, last (k, n) of dtype).
static py::tuple abstraction_pairs(py::dict spec, arri64 Q, arri64 P, py::dtype dtype) {
  AbsHolder h;
  abs_spec(h, spec);
  const auto& s = h.s;
  const auto k = Q.size();
  if (P.size() != k) throw std::invalid_argument("Q and P must have the same length");
  for (std::ptrdiff_t i = 0; i < k; ++i)
    if (Q.data()[i] < 0 || Q.data()[i] >= s.nq || P.data()[i] < 0 || P.data()[i] >= s.np)
      throw std::invalid_argument("cell index out of range");
  if (s.g_lo && std::ptrdiff_t(py::len(spec["g_lo"])) != k) throw std::invalid_argument("g_lo must have one row per pair");
  if (s.box_lo && std::ptrdiff_t(py::len(spec["box_lo"])) != k) throw std::invalid_argument("box_lo must have one row per pair");
  py::array_t<bool> en(k);
  py::array first(dtype, std::vector<std::ptrdiff_t>{k, s.n}), last(dtype, std::vector<std::ptrdiff_t>{k, s.n});
  auto* e = reinterpret_cast<std::uint8_t*>(en.mutable_data());
  const char kd = dtype.kind();
  const auto sz = dtype.itemsize();
  {
    py::gil_scoped_release release;
    if (kd == 'u' && sz == 1)
      ddma::abstraction_pairs(s, Q.data(), P.data(), k, e, static_cast<std::uint8_t*>(first.mutable_data()),
                              static_cast<std::uint8_t*>(last.mutable_data()));
    else if (kd == 'u' && sz == 2)
      ddma::abstraction_pairs(s, Q.data(), P.data(), k, e, static_cast<std::uint16_t*>(first.mutable_data()),
                              static_cast<std::uint16_t*>(last.mutable_data()));
    else if (kd == 'i' && sz == 4)
      ddma::abstraction_pairs(s, Q.data(), P.data(), k, e, static_cast<std::int32_t*>(first.mutable_data()),
                              static_cast<std::int32_t*>(last.mutable_data()));
    else
      throw std::invalid_argument("dtype must be uint8, uint16 or int32");
  }
  return py::make_tuple(en, first, last);
}

// ---------------------------------------------------------------- assumption monitor
// Shared arguments: lo, hi (m, n) slope bounds; theta (m,); tol; Z (K, n), Y (K, m).
using arri64m = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using arru8m = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

struct MonArgs {
  ddma::mon::Spec s;
  ddma::mon::View v;
};

static MonArgs mon_args(const arr& lo, const arr& hi, const arr& theta, double tol, const arr& Z,
                        const arr& Y) {
  if (lo.ndim() != 2 || hi.ndim() != 2 || lo.shape(0) != hi.shape(0) || lo.shape(1) != hi.shape(1))
    throw std::invalid_argument("lo and hi must be (m, n)");
  const int m = int(lo.shape(0)), n = int(lo.shape(1));
  if (theta.size() != m) throw std::invalid_argument("theta must have m entries");
  if (Z.ndim() != 2 || Y.ndim() != 2 || Z.shape(1) != n || Y.shape(1) != m || Z.shape(0) != Y.shape(0))
    throw std::invalid_argument("Z must be (K, n) and Y (K, m)");
  MonArgs a;
  a.s = {m, n, lo.data(), hi.data(), theta.data(), tol};
  a.v = {Z.data(), Y.data(), Z.shape(0), n, 1, m, 1};
  return a;
}

// Need as (any (m,) bool, width_r (m,), hi (m, n), hi_out (m, n) bool, lo, lo_out).
static py::tuple need_tuple(const ddma::mon::Need& N) {
  const auto m = N.m, n = N.n;
  py::array_t<bool> any(m), hio({m, n}), loo({m, n});
  py::array_t<double> w(m), hv({m, n}), lv({m, n});
  for (int i = 0; i < m; ++i) {
    any.mutable_data()[i] = N.any[i];
    w.mutable_data()[i] = N.width_r[i];
  }
  for (std::size_t t = 0; t < N.hi.size(); ++t) {
    hv.mutable_data()[t] = N.hi[t];
    lv.mutable_data()[t] = N.lo[t];
    hio.mutable_data()[t] = N.hi_out[t];
    loo.mutable_data()[t] = N.lo_out[t];
  }
  return py::make_tuple(any, w, hv, hio, lv, loo);
}

// Level 2: (need, worst (m,), witness (m,) stored index or -1).
static py::tuple monitor_list(arr lo, arr hi, arr theta, double tol, arr Z, arr Y, arr z, arr y,
                              arri64m idx, arru8m status) {
  auto a = mon_args(lo, hi, theta, tol, Z, Y);
  if (z.size() != a.s.n || y.size() != a.s.m) throw std::invalid_argument("z must have n and y m entries");
  if (status.size() < a.v.K) throw std::invalid_argument("status must have K entries");
  ddma::mon::Need N(a.s.m, a.s.n);
  std::vector<ddma::mon::Worst> W(a.s.m);
  {
    py::gil_scoped_release release;
    ddma::mon::check_list(a.s, a.v, z.data(), y.data(), idx.data(), idx.size(), status.data(), N, W.data());
  }
  py::array_t<double> wg(a.s.m);
  arri64m wk(a.s.m);
  for (int i = 0; i < a.s.m; ++i) {
    wg.mutable_data()[i] = W[i].gap;
    wk.mutable_data()[i] = W[i].k;
  }
  return py::make_tuple(need_tuple(N), wg, wk);
}

static ddma::mon::Grid grid_view(const py::tuple& g, int n, int m, std::int64_t K) {
  if (g.size() != 10) throw std::invalid_argument("grid must be a tuple of 10 arrays");
  ddma::mon::Grid G;
  G.n = n;
  G.m = m;
  // The tuple keeps the arrays alive; they must already have the right type and layout
  // (no conversion: a converted copy would be freed when this function returns).
  auto i64 = [&](int k) {
    py::array x = g[k].cast<py::array>();
    if (!py::isinstance<arri64m>(x) || !(x.flags() & py::array::c_style))
      throw std::invalid_argument("grid arrays 0, 7, 8, 9 must be C-contiguous int64");
    return arri64m::ensure(x);
  };
  auto f64 = [&](int k) {
    py::array x = g[k].cast<py::array>();
    if (!py::isinstance<arr>(x) || !(x.flags() & py::array::c_style))
      throw std::invalid_argument("grid arrays 1 to 6 must be C-contiguous float64");
    return arr::ensure(x);
  };
  auto cdim = i64(0), cnt = i64(7), head = i64(8), next = i64(9);
  auto glo = f64(1), ginv = f64(2), zlo = f64(3), zhi = f64(4), ylo = f64(5), yhi = f64(6);
  if (cdim.size() != n) throw std::invalid_argument("grid: one cell count per input");
  std::int64_t C = 1;
  for (int j = 0; j < n; ++j) C *= cdim.data()[j];
  if (cnt.size() < C || head.size() < C || zlo.size() < C * n || zhi.size() < C * n ||
      ylo.size() < C * m || yhi.size() < C * m || next.size() < K)
    throw std::invalid_argument("grid arrays too small");
  G.C = C;
  G.cdim = cdim.data();
  G.glo = glo.data();
  G.ginv = ginv.data();
  G.zlo = zlo.data();
  G.zhi = zhi.data();
  G.ylo = ylo.data();
  G.yhi = yhi.data();
  G.cnt = cnt.data();
  G.head = head.data();
  G.next = next.data();
  for (std::int64_t c = 0; c < C; ++c)
    if (G.head[c] >= K) throw std::invalid_argument("grid list points beyond the data");
  return G;
}

// Level 3: (need, worst (P, m), witness (P, m), inrange (R, 3) int64 [q, k, i], gaps (R,), pairs).
static py::tuple monitor_full(arr lo, arr hi, arr theta, double tol, arr Z, arr Y, arru8m status,
                              std::int64_t first, std::int64_t last, py::object grid, bool prune) {
  auto a = mon_args(lo, hi, theta, tol, Z, Y);
  if (status.size() < a.v.K) throw std::invalid_argument("status must have K entries");
  ddma::mon::Grid G;
  const ddma::mon::Grid* gp = nullptr;
  if (!grid.is_none()) {
    G = grid_view(grid.cast<py::tuple>(), a.s.n, a.s.m, a.v.K);
    gp = &G;
  }
  ddma::mon::FullOut o;
  {
    py::gil_scoped_release release;
    o = ddma::mon::check_full(a.s, a.v, status.data(), first, last, gp, prune);
  }
  const std::int64_t P = last - first, m = a.s.m, R = std::int64_t(o.inrange.size());
  py::array_t<double> wg({P, m}), gaps(R);
  arri64m wk({P, m}), ir({R, std::int64_t(3)});
  for (std::size_t r = 0; r < o.worst.size(); ++r) {
    wg.mutable_data()[r] = o.worst[r].gap;
    wk.mutable_data()[r] = o.worst[r].k;
  }
  for (std::int64_t r = 0; r < R; ++r) {
    ir.mutable_data()[3 * r] = o.inrange[r].q;
    ir.mutable_data()[3 * r + 1] = o.inrange[r].k;
    ir.mutable_data()[3 * r + 2] = o.inrange[r].i;
    gaps.mutable_data()[r] = o.inrange[r].gap;
  }
  return py::make_tuple(need_tuple(o.need), wg, wk, ir, gaps, o.pairs);
}

// check_data: (need, count (m,), worst (m,), pair (m, 2) [later, earlier], conflicts (K,), pairs).
static py::tuple monitor_all(arr lo, arr hi, arr theta, double tol, arr Z, arr Y, bool prune) {
  auto a = mon_args(lo, hi, theta, tol, Z, Y);
  ddma::mon::AllOut o;
  {
    py::gil_scoped_release release;
    o = ddma::mon::check_all(a.s, a.v, prune);
  }
  const std::int64_t m = a.s.m, K = a.v.K;
  arri64m cnt(m), pr({m, std::int64_t(2)}), conf(K);
  py::array_t<double> wg(m);
  for (int i = 0; i < m; ++i) {
    cnt.mutable_data()[i] = o.count[i];
    wg.mutable_data()[i] = o.worst[i].gap;
    pr.mutable_data()[2 * i] = o.worst[i].a;
    pr.mutable_data()[2 * i + 1] = o.worst[i].b;
  }
  std::copy(o.conflicts.begin(), o.conflicts.end(), conf.mutable_data());
  return py::make_tuple(need_tuple(o.need), cnt, wg, pr, conf, o.pairs);
}

// Grid of the points: (cdim, glo, ginv, zlo, zhi, ylo, yhi, cnt, head, next, cell).
static py::tuple monitor_grid(arr Z, arr Y) {
  if (Z.ndim() != 2 || Y.ndim() != 2 || Z.shape(0) != Y.shape(0))
    throw std::invalid_argument("Z must be (K, n) and Y (K, m)");
  const int n = int(Z.shape(1)), m = int(Y.shape(1));
  ddma::mon::View v{Z.data(), Y.data(), Z.shape(0), n, 1, m, 1};
  ddma::mon::GridData G;
  {
    py::gil_scoped_release release;
    G = ddma::mon::build_grid(v, n, m);
  }
  auto vec_i = [](const std::vector<std::int64_t>& x) {
    arri64m a(std::int64_t(x.size()));
    std::copy(x.begin(), x.end(), a.mutable_data());
    return a;
  };
  auto mat_d = [](const std::vector<double>& x, std::int64_t r, std::int64_t c) {
    py::array_t<double> a({r, c});
    std::copy(x.begin(), x.end(), a.mutable_data());
    return a;
  };
  py::array_t<double> glo(n), ginv(n);
  std::copy(G.glo.begin(), G.glo.end(), glo.mutable_data());
  std::copy(G.ginv.begin(), G.ginv.end(), ginv.mutable_data());
  return py::make_tuple(vec_i(G.cdim), glo, ginv, mat_d(G.zlo, G.C, n), mat_d(G.zhi, G.C, n),
                        mat_d(G.ylo, G.C, m), mat_d(G.yhi, G.C, m), vec_i(G.cnt), vec_i(G.head),
                        vec_i(G.next), vec_i(G.cell));
}

PYBIND11_MODULE(_core, mod) {
  mod.def("monitor_list", &monitor_list, "Monitor level 2: one point against a list");
  mod.def("monitor_full", &monitor_full, "Monitor level 3: points [first, last) against all earlier ones");
  mod.def("monitor_all", &monitor_all, "check_data: all pairs");
  mod.def("monitor_grid", &monitor_grid, "Cell grid of the points (for pruning)");
  mod.def("abstraction_rows", &abstraction_rows,
          "Fill the transitions of the state cells qs (all inputs) in place");
  mod.def("abstraction_pairs", &abstraction_pairs,
          "Transitions of the pairs (Q[k], P[k]): (enabled, first, last)");
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
  mod.def("_set_chunk_entries", [](std::ptrdiff_t c) { ddma::chunk_entries() = c > 0 ? c : (std::ptrdiff_t(1) << 20); },
          "Internal (benchmarks/tests): values written per round in the bucketed mode");
  mod.def("growth_local", &growth_local, "Per-cell cone bounds, 'cell' or 'slab' (lo, up)",
          py::arg("Z"), py::arg("y"), py::arg("edges"), py::arg("g_lo"), py::arg("g_hi"),
          py::arg("theta"), py::arg("slab"));
}
