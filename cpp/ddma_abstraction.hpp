// ddma_abstraction.hpp -- building the symbolic abstraction (transitions) in C++.
//
// For every state-input cell pair (q, p) the abstraction stores whether p is enabled at q
// and the box of successor cells first..last. Per pair this takes:
//   1. the successor box [lo, hi] of f(x, u) + g(x, u) + W over the cell pair:
//        map:    lo = sigma_lo(q, p) + g_lo(q, p)              (hi likewise)
//        model:  lo = max(gm_lo + sm_lo, gp_lo - sp_hi)        (two-matrix model, thesis
//                hi = min(gm_hi + sm_hi, gp_hi - sp_lo)         Ch. 3, approach 1)
//        boxes:  lo, hi given (other models: computed in Python)
//      sigma, sm and sp are read from their grids at a flat index offq[q] + offp[p]
//      (offsets precomputed in Python from the features each map uses); g, gm and gp are
//      affine interval bounds stored by parts, mid = qmid[q] + pmid[p] and
//      rad = qrad[q] + prad[p], or (any other g) given per pair;
//   2. clipping to an assumption box (assume-guarantee contracts);
//   3. the checks (bounded, non-empty, inside X) and the cell ranges (binary search).
// The arithmetic is exactly that of the NumPy code (same operations in the same order,
// built with -ffp-contract=off), so both give bit-identical abstractions. Pairs are
// independent: threads split the state cells and write disjoint parts of the output, so
// the result does not depend on the number of threads.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <vector>

#include "ddma_parallel.hpp"

namespace ddma {

// One map read at (offq[q] + offp[p]) * fstride + c * cstride, written to output outs[c]
// (Python grids (cells, m): fstride = m, cstride = 1; MATLAB (cells x m): 1, n_cells).
struct AbsGather {
  const std::int64_t* offq = nullptr;
  const std::int64_t* offp = nullptr;
  const void* lo = nullptr;
  const void* hi = nullptr;
  bool f32 = false;  // grids stored as float32 (compact maps)
  int m = 0;
  std::ptrdiff_t fstride = 0, cstride = 1;  // fstride 0 means m
  std::vector<int> outs;
  int target = 0;  // 0: sigma (map), 1: sigma minus, 2: sigma plus (model)
};

// Affine interval bounds by parts: mid = qmid[q] + pmid[p], rad = qrad[q] + prad[p],
// bounds mid -/+ rad. pmid == nullptr: no input part (mid = qmid, rad = qrad).
struct AbsAffine {
  const double* qmid = nullptr;
  const double* qrad = nullptr;
  const double* pmid = nullptr;
  const double* prad = nullptr;
  bool set() const { return qmid != nullptr; }
};

// Cell index along one axis: np.searchsorted(e, v, "right") - 1 clipped to [0, ne - 2],
// for v in [e[0], e[ne - 1]]. A table of buckets of equal width gives a first guess (the
// number of edges <= the bucket's left end); a short scan then makes it exact, so the
// result never depends on the rounding of the bucket computation. O(1) per value instead
// of a binary search.
struct AxisSearch {
  const double* e = nullptr;
  std::ptrdiff_t ne = 0;
  double e0 = 0, scale = 0;
  std::vector<std::ptrdiff_t> table;

  void init(const double* edges, std::ptrdiff_t n_edges) {
    e = edges;
    ne = n_edges;
    e0 = e[0];
    const std::ptrdiff_t nb = 4 * (ne - 1);
    const double width = e[ne - 1] - e[0];
    table.assign(std::size_t(nb) + 1, 0);
    scale = width > 0 ? double(nb) / width : 0.0;
    for (std::ptrdiff_t b = 0; b <= nb; ++b) {
      const double left = e0 + (width * double(b)) / double(nb);
      table[b] = std::upper_bound(e, e + ne, left) - e;
    }
  }
  std::ptrdiff_t cell(double v) const {
    double t = (v - e0) * scale;
    std::ptrdiff_t b = t > 0 ? std::ptrdiff_t(t) : 0;
    if (b >= std::ptrdiff_t(table.size())) b = std::ptrdiff_t(table.size()) - 1;
    std::ptrdiff_t k = table[b];  // guess for the number of edges <= v
    while (k < ne && e[k] <= v) ++k;
    while (k > 0 && e[k - 1] > v) --k;
    return std::min<std::ptrdiff_t>(std::max<std::ptrdiff_t>(k - 1, 0), ne - 2);
  }
};

struct AbsSpec {
  int n = 0;  // state dimension
  std::ptrdiff_t nq = 0, np = 0;
  int mode = 0;  // 0 map, 1 two-matrix model, 2 boxes given
  std::vector<AbsGather> gathers;
  AbsAffine gm, gp;               // model only
  AbsAffine g;                    // user g in parts (affine_bounds), or
  const double* g_lo = nullptr;   // user g per pair position (pos * n + j)
  const double* g_hi = nullptr;
  const double* box_lo = nullptr;  // mode 2: successor boxes per pair position
  const double* box_hi = nullptr;
  // per-pair arrays (g_lo, box_lo, ...) are read at pos * pair_stride + j * dim_stride
  // (Python (k, n): n, 1; MATLAB k x n: 1, k); pair_stride 0 means n.
  std::ptrdiff_t pair_stride = 0, dim_stride = 1;
  std::vector<const double*> edges;  // state partition
  std::vector<std::ptrdiff_t> ne;
  bool assume = false;
  std::vector<double> a_lo, a_hi;
  std::vector<std::ptrdiff_t> c_lo, c_hi;
  std::vector<AxisSearch> search;  // filled by prepare()

  void prepare() {
    search.resize(edges.size());
    for (std::size_t j = 0; j < edges.size(); ++j) search[j].init(edges[j], ne[j]);
  }
};

namespace detail {

inline double rd(const void* p, bool f32, std::ptrdiff_t i) {
  return f32 ? double(static_cast<const float*>(p)[i]) : static_cast<const double*>(p)[i];
}

// np.maximum / np.minimum: NaN if either argument is NaN.
inline double nan_max(double a, double b) {
  if (std::isnan(a) || std::isnan(b)) return std::nan("");
  return a < b ? b : a;
}
inline double nan_min(double a, double b) {
  if (std::isnan(a) || std::isnan(b)) return std::nan("");
  return b < a ? b : a;
}

inline void affine(const AbsAffine& a, int n, std::ptrdiff_t q, std::ptrdiff_t p, double* lo,
                   double* hi) {
  for (int j = 0; j < n; ++j) {
    double mid = a.qmid[q * n + j], rad = a.qrad[q * n + j];
    if (a.pmid) {
      mid = mid + a.pmid[p * n + j];
      rad = rad + a.prad[p * n + j];
    }
    lo[j] = mid - rad;
    hi[j] = mid + rad;
  }
}

// Successor box of the pair (q, p) at position pos; lo/hi/b1..b4 are scratch of size n.
inline void pair_box(const AbsSpec& s, std::ptrdiff_t q, std::ptrdiff_t p, std::ptrdiff_t pos,
                     double* lo, double* hi, double* b1, double* b2, double* b3, double* b4) {
  const int n = s.n;
  const std::ptrdiff_t ps = s.pair_stride ? s.pair_stride : n, ds = s.dim_stride;
  if (s.mode == 2) {
    for (int j = 0; j < n; ++j) {
      lo[j] = s.box_lo[pos * ps + j * ds];
      hi[j] = s.box_hi[pos * ps + j * ds];
    }
    return;
  }
  // user g (lo in b3, hi in b4), if any
  const bool has_g = s.g.set() || s.g_lo != nullptr;
  if (s.g.set()) {
    affine(s.g, n, q, p, b3, b4);
  } else if (s.g_lo) {
    for (int j = 0; j < n; ++j) {
      b3[j] = s.g_lo[pos * ps + j * ds];
      b4[j] = s.g_hi[pos * ps + j * ds];
    }
  }
  if (s.mode == 0) {
    for (const auto& gt : s.gathers) {
      const std::ptrdiff_t f = (gt.offq[q] + gt.offp[p]) * (gt.fstride ? gt.fstride : gt.m);
      for (int c = 0; c < gt.m; ++c) {
        lo[gt.outs[c]] = rd(gt.lo, gt.f32, f + c * gt.cstride);
        hi[gt.outs[c]] = rd(gt.hi, gt.f32, f + c * gt.cstride);
      }
    }
    if (has_g)
      for (int j = 0; j < n; ++j) {
        lo[j] = lo[j] + b3[j];
        hi[j] = hi[j] + b4[j];
      }
    return;
  }
  // mode 1: two-matrix model. sm in lo/hi, sp in b1/b2, then gm, gp.
  for (const auto& gt : s.gathers) {
    double* L = gt.target == 1 ? lo : b1;
    double* H = gt.target == 1 ? hi : b2;
    const std::ptrdiff_t f = (gt.offq[q] + gt.offp[p]) * (gt.fstride ? gt.fstride : gt.m);
    for (int c = 0; c < gt.m; ++c) {
      L[gt.outs[c]] = rd(gt.lo, gt.f32, f + c * gt.cstride);
      H[gt.outs[c]] = rd(gt.hi, gt.f32, f + c * gt.cstride);
    }
  }
  for (int j = 0; j < n; ++j) {
    double gm_lo, gm_hi, gp_lo, gp_hi;
    {
      double mid = s.gm.qmid[q * n + j], rad = s.gm.qrad[q * n + j];
      if (s.gm.pmid) {
        mid = mid + s.gm.pmid[p * n + j];
        rad = rad + s.gm.prad[p * n + j];
      }
      gm_lo = mid - rad;
      gm_hi = mid + rad;
    }
    {
      double mid = s.gp.qmid[q * n + j], rad = s.gp.qrad[q * n + j];
      if (s.gp.pmid) {
        mid = mid + s.gp.pmid[p * n + j];
        rad = rad + s.gp.prad[p * n + j];
      }
      gp_lo = mid - rad;
      gp_hi = mid + rad;
    }
    if (has_g) {
      gm_lo = gm_lo + b3[j];
      gm_hi = gm_hi + b4[j];
      gp_lo = gp_lo + b3[j];
      gp_hi = gp_hi + b4[j];
    }
    const double sm_lo = lo[j], sm_hi = hi[j], sp_lo = b1[j], sp_hi = b2[j];
    lo[j] = nan_max(gm_lo + sm_lo, gp_lo - sp_hi);
    hi[j] = nan_min(gm_hi + sm_hi, gp_hi - sp_lo);
  }
}

// Steps 2 and 3: enabled flag and cell range of one pair. Returns ok.
template <class T>
inline bool box_cells(const AbsSpec& s, double* lo, double* hi, T* first, T* last) {
  const int n = s.n;
  if (s.assume)
    for (int j = 0; j < n; ++j) {
      if (!std::isnan(lo[j])) lo[j] = std::min(std::max(lo[j], s.a_lo[j]), s.a_hi[j]);
      if (!std::isnan(hi[j])) hi[j] = std::min(std::max(hi[j], s.a_lo[j]), s.a_hi[j]);
    }
  bool ok = true;
  for (int j = 0; j < n && ok; ++j) {
    const double* e = s.edges[j];
    const std::ptrdiff_t k = s.ne[j];
    ok = (lo[j] <= hi[j]) && std::isfinite(lo[j]) && std::isfinite(hi[j]) && (lo[j] >= e[0]) &&
         (hi[j] <= e[k - 1]);
  }
  if (!ok) {
    for (int j = 0; j < n; ++j) first[j] = last[j] = T(0);
    return false;
  }
  for (int j = 0; j < n; ++j) {
    std::ptrdiff_t f = s.search[j].cell(lo[j]);
    std::ptrdiff_t l = s.search[j].cell(hi[j]);
    if (s.assume) {
      f = std::min(std::max(f, s.c_lo[j]), s.c_hi[j]);
      l = std::min(std::max(l, s.c_lo[j]), s.c_hi[j]);
    }
    first[j] = T(f);
    last[j] = T(l);
  }
  return true;
}

}  // namespace detail

// Rows mode: all inputs of the state cells qs[0..r). Outputs are the full (nq, np[, n])
// arrays; per-pair inputs (g_lo, box_lo) are indexed by position i * np + p.
template <class T>
void abstraction_rows(const AbsSpec& s, const std::int64_t* qs, std::ptrdiff_t r,
                      std::uint8_t* enabled, T* first, T* last) {
  const std::ptrdiff_t np = s.np;
  const int n = s.n;
  const int nt = (r * np > (std::ptrdiff_t(1) << 15) && r > 1) ? num_threads() : 1;
  ErrorBox err;
#pragma omp parallel num_threads(nt)
  {
    std::vector<double> buf(6 * std::size_t(n));
    double *lo = buf.data(), *hi = lo + n, *b1 = hi + n, *b2 = b1 + n, *b3 = b2 + n,
           *b4 = b3 + n;
#pragma omp for schedule(static)
    for (std::ptrdiff_t i = 0; i < r; ++i) {
      err.run([&] {
        const std::ptrdiff_t q = qs[i];
        for (std::ptrdiff_t p = 0; p < np; ++p) {
          detail::pair_box(s, q, p, i * np + p, lo, hi, b1, b2, b3, b4);
          const std::ptrdiff_t o = q * np + p;
          enabled[o] = detail::box_cells<T>(s, lo, hi, first + o * n, last + o * n);
        }
      });
    }
  }
  err.rethrow();
}

// Pairs mode: the pairs (Q[k], P[k]); outputs (k,) and (k, n).
template <class T>
void abstraction_pairs(const AbsSpec& s, const std::int64_t* Q, const std::int64_t* P,
                       std::ptrdiff_t k, std::uint8_t* enabled, T* first, T* last) {
  const int n = s.n;
  const int nt = k > (std::ptrdiff_t(1) << 15) ? num_threads() : 1;
  ErrorBox err;
#pragma omp parallel num_threads(nt)
  {
    std::vector<double> buf(6 * std::size_t(n));
    double *lo = buf.data(), *hi = lo + n, *b1 = hi + n, *b2 = b1 + n, *b3 = b2 + n,
           *b4 = b3 + n;
#pragma omp for schedule(static)
    for (std::ptrdiff_t i = 0; i < k; ++i) {
      err.run([&] {
        detail::pair_box(s, Q[i], P[i], i, lo, hi, b1, b2, b3, b4);
        enabled[i] = detail::box_cells<T>(s, lo, hi, first + i * n, last + i * n);
      });
    }
  }
  err.rethrow();
}

// Column-major outputs (MATLAB). Grid: the pairs (qs[i], ps[c]) for i < r, c < k, at
// position i + r * c; outputs enabled (r x k) and first/last (r x k x n), 0-based cells.
// paired: the pairs (qs[i], ps[i]) for i < r (k = 1); outputs r x 1 and r x n.
template <class T>
void abstraction_colmajor(const AbsSpec& s, const std::int64_t* qs, std::ptrdiff_t r,
                          const std::int64_t* ps, std::ptrdiff_t k, bool paired,
                          std::uint8_t* enabled, T* first, T* last) {
  const int n = s.n;
  if (paired) k = 1;
  const std::ptrdiff_t rk = r * k;
  const int nt = rk > (std::ptrdiff_t(1) << 15) ? num_threads() : 1;
  ErrorBox err;
#pragma omp parallel num_threads(nt)
  {
    std::vector<double> buf(6 * std::size_t(n));
    std::vector<T> fb(n), lb(n);
    double *lo = buf.data(), *hi = lo + n, *b1 = hi + n, *b2 = b1 + n, *b3 = b2 + n,
           *b4 = b3 + n;
#pragma omp for schedule(static)
    for (std::ptrdiff_t pos = 0; pos < rk; ++pos) {
      err.run([&] {
        const std::ptrdiff_t i = pos % r, c = paired ? pos : pos / r;
        detail::pair_box(s, qs[i], ps[c], pos, lo, hi, b1, b2, b3, b4);
        enabled[pos] = detail::box_cells<T>(s, lo, hi, fb.data(), lb.data());
        for (int j = 0; j < n; ++j) {
          first[pos + rk * j] = fb[j];
          last[pos + rk * j] = lb[j];
        }
      });
    }
  }
  err.rethrow();
}

}  // namespace ddma
