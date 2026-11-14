// ddma_core.hpp -- header-only C++ kernels shared by the Python (pybind11) and MATLAB (MEX)
// interfaces of ddma.
//
// fit_sigma: tightest interval-valued simulating map on a fixed rectangular partition
// (Makdesi, Girard, Fribourg, "Data-driven models of monotone systems", IEEE TAC 2024,
// Theorem 2 and Proposition 5). One fused pass over the data (locate + scatter), then one
// running min / max sweep per axis: O(|K| n log K_i + |Q| n m) time, no temporaries.
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

// Running min (upper) / max (lower) along every axis. `reverse` = suffix accumulation.
// Blocked loops: for axis ax, the array is (outer, S, inner) with inner = stride; the inner
// loop is contiguous (step m), so it vectorizes and needs no index arithmetic.
template <class T, class Op>
inline void sweep(T* a, const std::vector<std::ptrdiff_t>& shape, std::ptrdiff_t m,
                  std::ptrdiff_t out_i, const std::vector<char>& reverse, Op op) {
  const std::ptrdiff_t n = std::ptrdiff_t(shape.size());
  std::ptrdiff_t N = 1;
  for (auto s : shape) N *= s;
  std::ptrdiff_t inner = N;
  for (std::ptrdiff_t ax = 0; ax < n; ++ax) {
    const std::ptrdiff_t S = shape[ax];
    inner /= S;
    const std::ptrdiff_t outer = N / (S * inner);
    if (S < 2) continue;
    for (std::ptrdiff_t o = 0; o < outer; ++o) {
      T* base = a + (o * S * inner) * m + out_i;
      if (reverse[ax]) {
        for (std::ptrdiff_t c = S - 2; c >= 0; --c) {
          T* cur = base + c * inner * m;
          const T* nxt = cur + inner * m;
          for (std::ptrdiff_t r = 0; r < inner; ++r) cur[r * m] = op(cur[r * m], nxt[r * m]);
        }
      } else {
        for (std::ptrdiff_t c = 1; c < S; ++c) {
          T* cur = base + c * inner * m;
          const T* prv = cur - inner * m;
          for (std::ptrdiff_t r = 0; r < inner; ++r) cur[r * m] = op(cur[r * m], prv[r * m]);
        }
      }
    }
  }
}

// Z: K x n (row-major), Y: K x m (row-major). lower/upper: preallocated N*m, of type
// double (exact) or float (half the memory; bounds rounded outward, still sound).
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
  std::fill(lower, lower + N * m, -INF);
  std::fill(upper, upper + N * m, INF);

  std::vector<std::ptrdiff_t> ub(n), lb(n);
  std::vector<char> ub_ok(n), lb_ok(n);
  for (std::ptrdiff_t k = 0; k < K; ++k) {
    const double* z = Z + k * n;
    for (std::ptrdiff_t j = 0; j < n; ++j) {
      const std::ptrdiff_t last = shape[j] - 1;
      std::ptrdiff_t u = search(ax[j], z[j], true) - 2;   // last cell with upper edge <= z
      std::ptrdiff_t l = search(ax[j], z[j], false);      // first cell with lower edge >= z
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
        if (s[j] > 0) { ok = ub_ok[j]; q += ub[j] * stride[j]; }
        else if (s[j] < 0) { ok = lb_ok[j]; q += lb[j] * stride[j]; }
        else q += (shape[j] - 1) * stride[j];
      }
      if (ok) { T& x = upper[q * m + i]; x = std::min(x, round_up<T>(y + theta[i])); }
      // lower bound: corner "below"
      ok = true;
      q = 0;
      for (std::ptrdiff_t j = 0; j < n && ok; ++j) {
        if (s[j] > 0) { ok = lb_ok[j]; q += lb[j] * stride[j]; }
        else if (s[j] < 0) { ok = ub_ok[j]; q += ub[j] * stride[j]; }
        else q += (shape[j] - 1) * stride[j];
      }
      if (ok) { T& x = lower[q * m + i]; x = std::max(x, round_down<T>(y - theta[i])); }
    }
  }
  auto mn = [](T a, T b) { return a < b ? a : b; };
  auto mx = [](T a, T b) { return a > b ? a : b; };
  for (std::ptrdiff_t i = 0; i < m; ++i) {
    std::vector<char> rev_u(n), rev_l(n);
    for (std::ptrdiff_t j = 0; j < n; ++j) {
      rev_u[j] = signs[i * n + j] >= 0;
      rev_l[j] = signs[i * n + j] <= 0;
    }
    sweep(upper, shape, m, i, rev_u, mn);
    sweep(lower, shape, m, i, rev_l, mx);
  }
}

}  // namespace ddma
