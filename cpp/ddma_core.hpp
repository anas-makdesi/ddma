// ddma_core.hpp -- header-only C++ kernels shared by the Python (pybind11) and MATLAB (MEX)
// interfaces of ddma.
//
// fit_sigma: tightest interval-valued simulating map on a fixed rectangular partition
// (Makdesi, Girard, Fribourg, "Data-driven models of monotone systems", IEEE TAC 2024,
// Theorem 2 and Proposition 5). One pass over the data (locate + scatter), then one
// running min / max sweep per axis: O(|K| n log K_i + |Q| n m) time. Both phases use
// threads (OpenMP, optional; see ddma_parallel.hpp); results are bit-identical for any
// number of threads.
//
// Conventions (identical to the Python implementation in ddma/partitioned.py):
//  * edges[j]: strictly increasing breakpoints of axis j (K_j + 1 values, K_j cells);
//    cells are [e_j[c], e_j[c+1]) and the last one is closed.
//  * signs[i*n + j] in {-1, 0, +1}: monotonicity of output i in input j.
//  * outputs lower/upper have C-order layout (cell, output): index q*m + i.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "ddma_parallel.hpp"

namespace ddma {

struct Axis {
  const double* e;  // edges
  std::ptrdiff_t k; // number of cells (edges has k + 1 values)
  bool uniform;
  double lo, scale; // for uniform axes: index ~ (z - lo) * scale
};

inline Axis make_axis(const double* e, std::ptrdiff_t n_edges) {
  Axis a{e, n_edges - 1, true, e[0], 0.0};
  if (a.k < 1) throw std::invalid_argument("an axis needs at least two edges");
  const double h = (e[n_edges - 1] - e[0]) / double(a.k);
  for (std::ptrdiff_t c = 0; c < a.k; ++c) {
    if (!(e[c + 1] > e[c])) throw std::invalid_argument("edges must be strictly increasing");
    if (std::fabs((e[c + 1] - e[c]) - h) > 1e-12 * std::fabs(h)) a.uniform = false;
  }
  a.scale = double(a.k) / (e[n_edges - 1] - e[0]);
  return a;
}

// Same result as numpy.searchsorted(e, z, side): number of edges < z (left) or <= z (right).
inline std::ptrdiff_t search(const Axis& a, double z, bool right) {
  const double* e = a.e;
  const std::ptrdiff_t ne = a.k + 1;
  if (!a.uniform) {
    return right ? std::upper_bound(e, e + ne, z) - e : std::lower_bound(e, e + ne, z) - e;
  }
  double g = std::floor((z - a.lo) * a.scale) + 1.0;
  std::ptrdiff_t r = g < 0 ? 0 : (g > double(ne) ? ne : std::ptrdiff_t(g));
  if (right) {
    while (r > 0 && e[r - 1] > z) --r;
    while (r < ne && e[r] <= z) ++r;
  } else {
    while (r > 0 && e[r - 1] >= z) --r;
    while (r < ne && e[r] < z) ++r;
  }
  return r;
}

// Outward rounding to the storage type T (exact for double; for float the result is a
// lower bound of x (down) or an upper bound (up), so bounds stay sound).
template <class T> inline T round_down(double x) {
  T f = static_cast<T>(x);
  if (double(f) > x) f = std::nextafter(f, -std::numeric_limits<T>::infinity());
  return f;
}
template <class T> inline T round_up(double x) {
  T f = static_cast<T>(x);
  if (double(f) < x) f = std::nextafter(f, std::numeric_limits<T>::infinity());
  return f;
}

// Running min (upper) / max (lower) along every axis, for all outputs at once.
// rev[i * n + ax] = 1: suffix accumulation (from the last cell back) for output i on axis ax.
// For axis ax the array is (outer, S, inner, m); each grid line (o, r, i) is an independent
// running min/max over its S cells. Threads share out blocks of lines (o, r-range): every
// line is processed with the same operations in the same order, whatever the thread
// count. The innermost loop runs over contiguous memory (r, i), so it vectorizes.
template <class T, class Op>
inline void sweep(T* a, const std::vector<std::ptrdiff_t>& shape, std::ptrdiff_t m,
                  const std::vector<char>& rev, Op op) {
  const std::ptrdiff_t n = std::ptrdiff_t(shape.size());
  std::ptrdiff_t N = 1;
  for (auto s : shape) N *= s;
  std::ptrdiff_t inner = N;
  for (std::ptrdiff_t ax = 0; ax < n; ++ax) {
    const std::ptrdiff_t S = shape[ax];
    inner /= S;
    const std::ptrdiff_t outer = N / (S * inner);
    if (S < 2) continue;
    std::vector<std::ptrdiff_t> fwd, bwd;  // outputs swept forward / backward on this axis
    for (std::ptrdiff_t i = 0; i < m; ++i) (rev[i * n + ax] ? bwd : fwd).push_back(i);
    const std::ptrdiff_t rch = std::max<std::ptrdiff_t>(1, 4096 / m);  // lines per task
    const std::ptrdiff_t nr = (inner + rch - 1) / rch;
    const std::ptrdiff_t L = inner * m;  // distance between neighbouring cells on this axis
    parallel_tasks(outer * nr, N * m, [&](std::ptrdiff_t t) {
      const std::ptrdiff_t o = t / nr, r0 = (t % nr) * rch, r1 = std::min(inner, r0 + rch);
      T* base = a + o * S * L;
      auto pass = [&](const std::vector<std::ptrdiff_t>& outs, bool backward) {
        if (outs.empty()) return;
        for (std::ptrdiff_t step = 1; step < S; ++step) {
          const std::ptrdiff_t c = backward ? S - 1 - step : step;
          T* cur = base + c * L;
          const T* src = backward ? cur + L : cur - L;  // the neighbour already done
          if (std::ptrdiff_t(outs.size()) == m) {   // all outputs: one contiguous range
            for (std::ptrdiff_t x = r0 * m; x < r1 * m; ++x) cur[x] = op(cur[x], src[x]);
          } else {
            for (std::ptrdiff_t r = r0; r < r1; ++r)
              for (auto i : outs) cur[r * m + i] = op(cur[r * m + i], src[r * m + i]);
          }
        }
      };
      pass(fwd, false);
      pass(bwd, true);
    });
  }
}

// Z: K x n (row-major), Y: K x m (row-major). lower/upper: preallocated N*m, of type
// double (exact) or float (half the memory; bounds rounded outward, still sound).
//
// Each data point gives, for each output i, one upper-bound value (written at the cell
// "above" the point in the order of output i) and one lower-bound value (at the cell
// "below"); the sweeps then spread them over the grid. The writing is done by
// minmax_scatter (ddma_parallel.hpp), which uses threads and works block by block.
template <class T>
inline void fit_sigma(const double* Z, const double* Y, std::ptrdiff_t K, std::ptrdiff_t n,
                      std::ptrdiff_t m, const std::vector<const double*>& edges,
                      const std::vector<std::ptrdiff_t>& n_edges, const double* theta,
                      const std::int8_t* signs, T* lower, T* upper) {
  std::vector<Axis> ax;
  std::vector<std::ptrdiff_t> shape(n), stride(n, 1);
  for (std::ptrdiff_t j = 0; j < n; ++j) {
    ax.push_back(make_axis(edges[j], n_edges[j]));
    shape[j] = ax[j].k;
  }
  for (std::ptrdiff_t j = n - 2; j >= 0; --j) stride[j] = stride[j + 1] * shape[j + 1];
  std::ptrdiff_t N = 1;
  for (auto s : shape) N *= s;
  const T INF = std::numeric_limits<T>::infinity();
  parallel_fill(lower, N * m, -INF);
  parallel_fill(upper, N * m, INF);

  // One generator per thread, with its own scratch arrays.
  auto make_gen = [&]() {
    // scratch: ub, lb (cell indices), ub_ok, lb_ok (flags), one value per axis
    // everything the generator reads, captured by value (kept in registers)
    return [Z, Y, n, m, theta, signs, axes = ax.data(), shp = shape.data(),
            str = stride.data(), buf = std::vector<std::ptrdiff_t>(2 * n),
            ok_buf = std::vector<char>(2 * n)](std::ptrdiff_t k, auto& emit_up,
                                               auto& emit_lo) mutable {
      std::ptrdiff_t* ub = buf.data();
      std::ptrdiff_t* lb = ub + n;
      char* ub_ok = ok_buf.data();
      char* lb_ok = ub_ok + n;
      const double* z = Z + k * n;
      for (std::ptrdiff_t j = 0; j < n; ++j) {
        const std::ptrdiff_t last = shp[j] - 1;
        std::ptrdiff_t u = search(axes[j], z[j], true) - 2;   // last cell with upper edge <= z
        std::ptrdiff_t l = search(axes[j], z[j], false);      // first cell with lower edge >= z
        ub_ok[j] = u >= 0;
        lb_ok[j] = l <= last;
        ub[j] = std::min(u, last);
        lb[j] = std::max<std::ptrdiff_t>(l, 0);
      }
      for (std::ptrdiff_t i = 0; i < m; ++i) {
        const std::int8_t* s = signs + i * n;
        const double y = Y[k * m + i];
        // upper bound: corner "above" in the order of output i
        bool ok = true;
        std::ptrdiff_t q = 0;
        for (std::ptrdiff_t j = 0; j < n && ok; ++j) {
          if (s[j] > 0) { ok = ub_ok[j]; q += ub[j] * str[j]; }
          else if (s[j] < 0) { ok = lb_ok[j]; q += lb[j] * str[j]; }
          else q += (shp[j] - 1) * str[j];
        }
        if (ok) emit_up(q * m + i, round_up<T>(y + theta[i]));
        // lower bound: corner "below"
        ok = true;
        q = 0;
        for (std::ptrdiff_t j = 0; j < n && ok; ++j) {
          if (s[j] > 0) { ok = lb_ok[j]; q += lb[j] * str[j]; }
          else if (s[j] < 0) { ok = ub_ok[j]; q += ub[j] * str[j]; }
          else q += (shp[j] - 1) * str[j];
        }
        if (ok) emit_lo(q * m + i, round_down<T>(y - theta[i]));
      }
    };
  };
  minmax_scatter<T, true>(K, N * m, upper, lower, m, make_gen);

  auto mn = [](T a, T b) { return a < b ? a : b; };
  auto mx = [](T a, T b) { return a > b ? a : b; };
  std::vector<char> rev_u(m * n), rev_l(m * n);
  for (std::ptrdiff_t x = 0; x < m * n; ++x) {
    rev_u[x] = signs[x] >= 0;
    rev_l[x] = signs[x] <= 0;
  }
  sweep(upper, shape, m, rev_u, mn);
  sweep(lower, shape, m, rev_l, mx);
}

}  // namespace ddma
