// ddma_monitor.hpp -- pair checks of the assumption monitor and of check_data.
//
// Assumptions: per output i and input j a slope interval [lo_ij, hi_ij] (+-inf allowed)
// and a disturbance width theta_i. Two data points a, b are consistent iff, for every i,
//
//     y_a,i - y_b,i <= theta_i + sum_j s_ij(z_a,j - z_b,j),
//     s_ij(t) = hi_ij t (t > 0),  lo_ij t (t < 0),  0 (t = 0).                       (*)
//
// For a new point q and a stored point k, gap_new is (*) for "q above k" and gap_old for
// "k above q"; a gap > tol is a violation (a proof that some assumption is wrong).
//
// Three kernels, all built on the same per-pair arithmetic (pair_gaps):
//   check_list : one point against a list of stored points        (level 2, random sample)
//   check_full : points [first, last) each against all earlier ones (level 3, flush)
//   check_all  : all pairs of a data set                            (check_data)
//
// Pruning with a cell grid (optional, exact). Each cell keeps the box of the points stored
// in it (smallest and largest z in each direction) and the smallest and largest y of each
// output. From these, an upper bound of the gap of every pair with that cell follows, by
// taking each term at its worst. If the bound is <= tol in both directions for every
// output, no pair with that cell can violate (*) and the cell is skipped.
// Why no rounding margin is needed: the bound is computed with the same rounded operations
// in the same order as the pair gap, with each input replaced by its worst value. Rounded
// subtraction, multiplication by a fixed slope and addition are monotone (a larger input
// never gives a smaller rounded result), so the computed gap of any pair is <= the
// computed bound. A skipped pair therefore has a computed gap <= tol: it would not have been
// a violation. So pruning changes only the work, never the result.
//
// Results are bit-identical for any number of threads: the per-pair arithmetic is fixed,
// and partial results are combined only with max, min, logical or and integer counts; ties
// in the worst gap go to the smaller index. They are also identical to the NumPy code
// (ddma._monitor_kernels) and the MATLAB code (+ddma/+internal/monitor_*.m), which use the
// same arithmetic in the same order.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "ddma_parallel.hpp"

namespace ddma {
namespace mon {

using idx_t = std::int64_t;
constexpr double kInf = std::numeric_limits<double>::infinity();

// The assumptions. lo, hi: m x n, row-major; theta: m.
struct Spec {
  int m = 0, n = 0;
  const double* lo = nullptr;
  const double* hi = nullptr;
  const double* theta = nullptr;
  double tol = 1e-9;
};

// Read-only view of K points with any memory layout: z(k, j) = Z[k*zk + j*zj].
// NumPy (K, n) C-order: zk = n, zj = 1. MATLAB K x n: zk = 1, zj = K (or the capacity).
struct View {
  const double* Z = nullptr;
  const double* Y = nullptr;
  idx_t K = 0;
  idx_t zk = 0, zj = 0, yk = 0, yi = 0;
  double z(idx_t k, int j) const { return Z[k * zk + j * zj]; }
  double y(idx_t k, int i) const { return Y[k * yk + i * yi]; }
};

// s(t) of (*). Both products are formed first, then one is selected: this branches less
// than selecting first, and gives the same value (an unused inf * 0 = NaN is discarded).
inline double cone(double t, double lo, double hi) {
  const double pa = hi * t, pb = lo * t;
  return t > 0 ? pa : (t < 0 ? pb : 0.0);
}

// Smallest value of s over [a, b]: s is linear on each side of 0, so the smallest value is
// at an end of the interval, or at 0 if the interval contains it.
inline double cone_min(double a, double b, double lo, double hi) {
  double v = std::min(cone(a, lo, hi), cone(b, lo, hi));
  if (a <= 0 && b >= 0) v = std::min(v, 0.0);
  return v;
}

// What each single assumption would need (see AssumptionMonitor.needed in Python):
//   width_r : largest gap of a violating pair (width needed = theta + width_r);
//   hi, hi_out : largest slope needed for the upper bound, or "ruled out" (a violating pair
//                with dz_j <= 0 cannot be fixed by that bound); lo, lo_out likewise.
struct Need {
  int m = 0, n = 0;
  std::vector<std::uint8_t> any;  // output i has a violating pair
  std::vector<double> width_r, hi, lo;
  std::vector<std::uint8_t> hi_out, lo_out;
  Need() = default;
  Need(int m_, int n_)
      : m(m_), n(n_), any(m_, 0), width_r(m_, -kInf), hi(std::size_t(m_) * n_, -kInf),
        lo(std::size_t(m_) * n_, kInf), hi_out(std::size_t(m_) * n_, 0),
        lo_out(std::size_t(m_) * n_, 0) {}
  void merge(const Need& o) {
    for (int i = 0; i < m; ++i) {
      any[i] |= o.any[i];
      width_r[i] = std::max(width_r[i], o.width_r[i]);
    }
    for (std::size_t t = 0; t < hi.size(); ++t) {
      hi[t] = std::max(hi[t], o.hi[t]);
      lo[t] = std::min(lo[t], o.lo[t]);
      hi_out[t] |= o.hi_out[t];
      lo_out[t] |= o.lo_out[t];
    }
  }
};

// Gaps of output i between point q (zq: n values, yq: its value of output i) and stored
// point k. Same operations, same order as the NumPy and MATLAB code:
//   d_j = zq_j - z_k,j;  tn = sum_j s(d_j);  to = sum_j s(-d_j)  (left to right, from 0)
//   dy = yq - y_k,i;     gap_new = (dy - theta) - tn;  gap_old = (-dy - theta) - to.
inline void pair_gaps(const Spec& s, int i, const double* zq, double yq, const View& v, idx_t k,
                      double& gn, double& go) {
  const double* lo = s.lo + std::size_t(i) * s.n;
  const double* hi = s.hi + std::size_t(i) * s.n;
  double tn = 0.0, to = 0.0;
  for (int j = 0; j < s.n; ++j) {
    const double d = zq[j] - v.z(k, j);
    tn = tn + cone(d, lo[j], hi[j]);
    to = to + cone(-d, lo[j], hi[j]);
  }
  const double dy = yq - v.y(k, i);
  gn = (dy - s.theta[i]) - tn;
  go = (-dy - s.theta[i]) - to;
}

// A violating pair, written as "a above b": du = z_a - z_b (n values), dyu = y_a,i - y_b,i,
// gap > tol. Updates what each assumption would need.
inline void accumulate(const Spec& s, int i, const double* du, double dyu, double gap, Need& N) {
  const int n = s.n;
  const double* lo = s.lo + std::size_t(i) * n;
  const double* hi = s.hi + std::size_t(i) * n;
  const double th = s.theta[i];
  N.any[i] = 1;
  N.width_r[i] = std::max(N.width_r[i], gap);
  double tot = 0.0;
  for (int j = 0; j < n; ++j) tot = tot + cone(du[j], lo[j], hi[j]);
  for (int j = 0; j < n; ++j) {
    const std::size_t t = std::size_t(i) * n + j;
    const double rest = (dyu - th) - (tot - cone(du[j], lo[j], hi[j]));
    if (du[j] > 0)
      N.hi[t] = std::max(N.hi[t], rest / du[j]);
    else
      N.hi_out[t] = 1;
    if (du[j] < 0)
      N.lo[t] = std::min(N.lo[t], rest / du[j]);
    else
      N.lo_out[t] = 1;
  }
}

// Evaluate the pair (q, k) for every output. Calls on_viol(i, g) for each output with
// g = max(gap_new, gap_old) > tol, after updating N with each violating direction.
template <class OnViol>
inline void eval_pair(const Spec& s, const double* zq, const double* yq, const View& v, idx_t k,
                      Need& N, double* du, OnViol&& on_viol) {
  for (int i = 0; i < s.m; ++i) {
    double gn, go;
    pair_gaps(s, i, zq, yq[i], v, k, gn, go);
    if (!(gn > s.tol) && !(go > s.tol)) continue;
    const double dy = yq[i] - v.y(k, i);
    for (int j = 0; j < s.n; ++j) du[j] = zq[j] - v.z(k, j);
    if (gn > s.tol) accumulate(s, i, du, dy, gn, N);
    if (go > s.tol) {
      for (int j = 0; j < s.n; ++j) du[j] = -du[j];
      accumulate(s, i, du, -dy, go, N);
    }
    on_viol(i, gn > go ? gn : go);
  }
}

// ------------------------------------------------------------------ cell grid
// Cells: c_j per input over [lo_j, lo_j + c_j / inv_j]; points outside go to the edge
// cells (harmless: a cell's box is that of its actual points). Members of cell c form a
// linked list: head[c], next[k] (-1 ends it). Bounds per cell: zlo, zhi (C x n),
// ylo, yhi (C x m), row-major; cnt: number of points.
struct Grid {
  int n = 0, m = 0;
  idx_t C = 0;
  const idx_t* cdim = nullptr;   // cells per input (n)
  const double* glo = nullptr;   // n
  const double* ginv = nullptr;  // n (0 when c_j = 1)
  const double* zlo = nullptr;
  const double* zhi = nullptr;
  const double* ylo = nullptr;
  const double* yhi = nullptr;
  const idx_t* cnt = nullptr;
  const idx_t* head = nullptr;
  const idx_t* next = nullptr;
};

// Cell of a point (flat index, last input fastest).
inline idx_t cell_of(const idx_t* cdim, const double* glo, const double* ginv, int n,
                     const double* z) {
  idx_t c = 0;
  for (int j = 0; j < n; ++j) {
    idx_t t = 0;
    if (cdim[j] > 1) {
      const double u = std::floor((z[j] - glo[j]) * ginv[j]);
      t = u >= 0 ? (u < double(cdim[j]) ? idx_t(u) : cdim[j] - 1) : 0;  // NaN -> 0
    }
    c = c * cdim[j] + t;
  }
  return c;
}

// Number of cells per input for K points: about K / 128 cells in all (at most 8192),
// spread evenly over the inputs whose data have some spread.
inline std::vector<idx_t> grid_shape(idx_t K, int n, const double* zmin, const double* zmax) {
  std::vector<idx_t> c(n, 1);
  int ne = 0;
  for (int j = 0; j < n; ++j) ne += zmax[j] > zmin[j];
  if (ne == 0) return c;
  const double target = std::min(8192.0, std::max(1.0, double(K) / 128.0));
  idx_t per = std::max<idx_t>(1, idx_t(std::floor(std::pow(target, 1.0 / ne))));
  auto total = [&](idx_t p) {
    double t = 1;
    for (int j = 0; j < ne; ++j) t *= double(p);
    return t;
  };
  while (total(per + 1) <= target) ++per;  // guard against pow rounding down
  while (per > 1 && total(per) > target) --per;
  for (int j = 0; j < n; ++j)
    if (zmax[j] > zmin[j]) c[j] = per;
  return c;
}

// Storage of a grid built here (check_data, or the monitor's index).
struct GridData {
  int n = 0, m = 0;
  idx_t C = 0;
  std::vector<idx_t> cdim, cell, cnt, head, next;
  std::vector<double> glo, ginv, zlo, zhi, ylo, yhi;
  Grid view() const {
    Grid g;
    g.n = n;
    g.m = m;
    g.C = C;
    g.cdim = cdim.data();
    g.glo = glo.data();
    g.ginv = ginv.data();
    g.zlo = zlo.data();
    g.zhi = zhi.data();
    g.ylo = ylo.data();
    g.yhi = yhi.data();
    g.cnt = cnt.data();
    g.head = head.data();
    g.next = next.data();
    return g;
  }
};

inline GridData build_grid(const View& v, int n, int m) {
  GridData G;
  G.n = n;
  G.m = m;
  const idx_t K = v.K;
  std::vector<double> zmin(n, kInf), zmax(n, -kInf);
  for (idx_t k = 0; k < K; ++k)
    for (int j = 0; j < n; ++j) {
      const double z = v.z(k, j);
      if (z < zmin[j]) zmin[j] = z;
      if (z > zmax[j]) zmax[j] = z;
    }
  G.cdim = grid_shape(K, n, zmin.data(), zmax.data());
  G.glo.assign(n, 0.0);
  G.ginv.assign(n, 0.0);
  G.C = 1;
  for (int j = 0; j < n; ++j) {
    G.C *= G.cdim[j];
    if (G.cdim[j] > 1) {
      G.glo[j] = zmin[j];
      G.ginv[j] = double(G.cdim[j]) / (zmax[j] - zmin[j]);
    }
  }
  const idx_t C = G.C;
  G.zlo.assign(std::size_t(C) * n, kInf);
  G.zhi.assign(std::size_t(C) * n, -kInf);
  G.ylo.assign(std::size_t(C) * m, kInf);
  G.yhi.assign(std::size_t(C) * m, -kInf);
  G.cnt.assign(C, 0);
  G.head.assign(C, -1);
  G.next.assign(K, -1);
  G.cell.assign(K, 0);
  std::vector<double> z(n);
  for (idx_t k = 0; k < K; ++k) {
    for (int j = 0; j < n; ++j) z[j] = v.z(k, j);
    const idx_t c = cell_of(G.cdim.data(), G.glo.data(), G.ginv.data(), n, z.data());
    G.cell[k] = c;
    for (int j = 0; j < n; ++j) {
      double& a = G.zlo[std::size_t(c) * n + j];
      double& b = G.zhi[std::size_t(c) * n + j];
      a = std::min(a, z[j]);
      b = std::max(b, z[j]);
    }
    for (int i = 0; i < m; ++i) {
      const double y = v.y(k, i);
      double& a = G.ylo[std::size_t(c) * m + i];
      double& b = G.yhi[std::size_t(c) * m + i];
      a = std::min(a, y);
      b = std::max(b, y);
    }
    ++G.cnt[c];
    G.next[k] = G.head[c];
    G.head[c] = k;
  }
  return G;
}

// Bound of one direction, "a above b", with d = z_a - z_b in [dlo, dhi] per input and
// y_a,i - y_b,i <= dyub. True if some pair could violate (bound > tol).
inline bool bound_exceeds(const Spec& s, int i, const double* dlo, const double* dhi,
                          double dyub) {
  const double* lo = s.lo + std::size_t(i) * s.n;
  const double* hi = s.hi + std::size_t(i) * s.n;
  double tot = 0.0;
  for (int j = 0; j < s.n; ++j) tot = tot + cone_min(dlo[j], dhi[j], lo[j], hi[j]);
  return (dyub - s.theta[i]) - tot > s.tol;
}

// Can point q (zq, yq) violate (*) with some point of the box [zlo, zhi] x [ylo, yhi]?
inline bool point_box(const Spec& s, const double* zq, const double* yq, const double* zlo,
                      const double* zhi, const double* ylo, const double* yhi, double* a,
                      double* b) {
  for (int i = 0; i < s.m; ++i) {
    for (int j = 0; j < s.n; ++j) {  // q above k: d = zq - z_k
      a[j] = zq[j] - zhi[j];
      b[j] = zq[j] - zlo[j];
    }
    if (bound_exceeds(s, i, a, b, yq[i] - ylo[i])) return true;
    for (int j = 0; j < s.n; ++j) {  // k above q: d = z_k - zq
      a[j] = zlo[j] - zq[j];
      b[j] = zhi[j] - zq[j];
    }
    if (bound_exceeds(s, i, a, b, yhi[i] - yq[i])) return true;
  }
  return false;
}

// Can some point of box A violate (*) with some point of box B?
inline bool box_box(const Spec& s, const double* zla, const double* zha, const double* yla,
                    const double* yha, const double* zlb, const double* zhb, const double* ylb,
                    const double* yhb, double* a, double* b) {
  for (int i = 0; i < s.m; ++i) {
    for (int j = 0; j < s.n; ++j) {  // a above b
      a[j] = zla[j] - zhb[j];
      b[j] = zha[j] - zlb[j];
    }
    if (bound_exceeds(s, i, a, b, yha[i] - ylb[i])) return true;
    for (int j = 0; j < s.n; ++j) {  // b above a
      a[j] = zlb[j] - zha[j];
      b[j] = zhb[j] - zla[j];
    }
    if (bound_exceeds(s, i, a, b, yhb[i] - yla[i])) return true;
  }
  return false;
}

// ------------------------------------------------------------------ worst gap
// Largest gap > tol with its witness; ties go to the smaller witness index.
struct Worst {
  double gap = -kInf;
  idx_t k = -1;
  void offer(double g, idx_t kk) {
    if (g > gap || (g == gap && kk < k)) {
      gap = g;
      k = kk;
    }
  }
};

inline void load_point(const View& v, idx_t q, int n, int m, double* z, double* y) {
  for (int j = 0; j < n; ++j) z[j] = v.z(q, j);
  for (int i = 0; i < m; ++i) y[i] = v.y(q, i);
}

// ------------------------------------------------------------------ level 2: a list
// Point (zq, yq) against the stored points idx[0..L) (repeats allowed). Points with
// status 2 (quarantined) are skipped. worst: m entries (witness = stored index).
inline void check_list(const Spec& s, const View& v, const double* zq, const double* yq,
                       const idx_t* idx, idx_t L, const std::uint8_t* status, Need& N,
                       Worst* worst) {
  std::vector<double> du(s.n);
  for (idx_t t = 0; t < L; ++t) {
    const idx_t k = idx[t];
    if (k < 0 || k >= v.K) throw std::out_of_range("sample index out of range");
    if (status && status[k] == 2) continue;
    eval_pair(s, zq, yq, v, k, N, du.data(), [&](int i, double g) { worst[i].offer(g, k); });
  }
}

// ------------------------------------------------------------------ level 3: full check
// Each point q in [first, last) against every stored point k < q (all statuses).
// Every violating pair updates N. For the blame:
//   k <  first: if status[k] == 0 (clean), offered to worst[(q - first) * m + i];
//   k >= first: recorded in `inrange` (q, k, i, gap), sorted by (q, k, i): whether it
//               counts depends on k's own outcome, decided in arrival order by the caller.
// Quarantined k < first only feed N. With a grid (and prune), cells that cannot violate
// with q are skipped.
struct InRange {
  idx_t q, k;
  int i;
  double gap;
};

struct FullOut {
  Need need;
  std::vector<Worst> worst;  // (last - first) * m
  std::vector<InRange> inrange;
  idx_t pairs = 0;  // pairs evaluated
};

inline FullOut check_full(const Spec& s, const View& v, const std::uint8_t* status, idx_t first,
                          idx_t last, const Grid* grid, bool prune) {
  if (first < 0 || last > v.K || first > last) throw std::out_of_range("bad range of points");
  const int m = s.m, n = s.n;
  const idx_t P = last - first;
  FullOut out;
  out.need = Need(m, n);
  out.worst.assign(std::size_t(P) * m, Worst());
  if (P == 0) return out;
  // work items: (point, block). Grid: blocks of cells; no grid: blocks of stored points.
  const idx_t bsize = grid ? 256 : 8192;
  const idx_t span = grid ? grid->C : last;  // what the blocks cover
  const idx_t nb = (span + bsize - 1) / bsize;
  const idx_t items = P * nb;
  double work = double(P) * double(last);
  const int nt = work > double(1 << 16) ? num_threads() : 1;
  std::vector<Need> needs(nt, Need(m, n));
  std::vector<std::vector<Worst>> worsts(nt, std::vector<Worst>(std::size_t(P) * m));
  std::vector<std::vector<InRange>> ins(nt);
  std::vector<idx_t> pairs(nt, 0);
  ErrorBox err;
#pragma omp parallel num_threads(nt)
  {
    const int t = thread_id();
    std::vector<double> zq(n), yq(m), du(n), a(n), b(n);
    idx_t my_pairs = 0;  // local: neighbouring counters of other threads would share a cache line
#pragma omp for schedule(dynamic, 1)
    for (idx_t it = 0; it < items; ++it) {
      err.run([&] {
        const idx_t q = first + it / nb, blk = it % nb;
        load_point(v, q, n, m, zq.data(), yq.data());
        Worst* w = &worsts[t][std::size_t(q - first) * m];
        auto pair = [&](idx_t k) {
          ++my_pairs;
          eval_pair(s, zq.data(), yq.data(), v, k, needs[t], du.data(), [&](int i, double g) {
            if (k < first) {
              if (!status || status[k] == 0) w[i].offer(g, k);
            } else {
              ins[t].push_back({q, k, i, g});
            }
          });
        };
        const idx_t c0 = blk * bsize, c1 = std::min(span, c0 + bsize);
        if (!grid) {
          for (idx_t k = c0; k < std::min(c1, q); ++k) pair(k);
          return;
        }
        for (idx_t c = c0; c < c1; ++c) {
          if (grid->cnt[c] == 0) continue;
          if (prune && !point_box(s, zq.data(), yq.data(), grid->zlo + c * n, grid->zhi + c * n,
                                  grid->ylo + c * m, grid->yhi + c * m, a.data(), b.data()))
            continue;
          for (idx_t k = grid->head[c]; k >= 0; k = grid->next[k])
            if (k < q) pair(k);
        }
      });
    }
    pairs[t] = my_pairs;
  }
  err.rethrow();
  for (int t = 0; t < nt; ++t) {
    out.need.merge(needs[t]);
    for (std::size_t r = 0; r < out.worst.size(); ++r)
      if (worsts[t][r].k >= 0) out.worst[r].offer(worsts[t][r].gap, worsts[t][r].k);
    out.inrange.insert(out.inrange.end(), ins[t].begin(), ins[t].end());
    out.pairs += pairs[t];
  }
  std::sort(out.inrange.begin(), out.inrange.end(), [](const InRange& x, const InRange& y) {
    if (x.q != y.q) return x.q < y.q;
    if (x.k != y.k) return x.k < y.k;
    return x.i < y.i;
  });
  return out;
}

// ------------------------------------------------------------------ check_data: all pairs
struct WorstPair {
  double gap = -kInf;
  idx_t a = -1, b = -1;  // a > b (a = the later point)
  void offer(double g, idx_t aa, idx_t bb) {
    if (g > gap || (g == gap && (aa < a || (aa == a && bb < b)))) {
      gap = g;
      a = aa;
      b = bb;
    }
  }
};

struct AllOut {
  Need need;
  std::vector<idx_t> count;      // m: violating pairs per output
  std::vector<WorstPair> worst;  // m
  std::vector<idx_t> conflicts;  // K: violating partners per point (any output)
  idx_t pairs = 0;
};

// Every unordered pair of the K points once. Each pair is evaluated with the later point
// (larger index) as q, as in the monitor. With prune, the points are grouped by cell
// (sorted copy, for contiguous memory); a pair of cells is skipped if their boxes cannot
// violate, then each point against the other cell's box, then the pairs.
inline AllOut check_all(const Spec& s, const View& v, bool prune) {
  const int m = s.m, n = s.n;
  const idx_t K = v.K;
  AllOut out;
  out.need = Need(m, n);
  out.count.assign(m, 0);
  out.worst.assign(m, WorstPair());
  out.conflicts.assign(K, 0);
  if (K < 2) return out;

  // Points sorted by cell (one cell with everything when not pruning).
  GridData G;
  std::vector<idx_t> order(K), start;
  idx_t C = 1;
  if (prune) {
    G = build_grid(v, n, m);
    C = G.C;
    start.assign(C + 1, 0);
    for (idx_t k = 0; k < K; ++k) ++start[G.cell[k] + 1];
    for (idx_t c = 0; c < C; ++c) start[c + 1] += start[c];
    std::vector<idx_t> pos(start.begin(), start.end() - 1);
    for (idx_t k = 0; k < K; ++k) order[pos[G.cell[k]]++] = k;  // stable: increasing k
  } else {  // no grid: blocks of consecutive points, so that threads share the work
    for (idx_t k = 0; k < K; ++k) order[k] = k;
    const idx_t bs = 1024;
    C = (K + bs - 1) / bs;
    start.resize(C + 1);
    for (idx_t c = 0; c <= C; ++c) start[c] = std::min(K, c * bs);
  }
  std::vector<double> Zs(std::size_t(K) * n), Ys(std::size_t(K) * m);
  for (idx_t r = 0; r < K; ++r) {
    for (int j = 0; j < n; ++j) Zs[std::size_t(r) * n + j] = v.z(order[r], j);
    for (int i = 0; i < m; ++i) Ys[std::size_t(r) * m + i] = v.y(order[r], i);
  }
  View sv{Zs.data(), Ys.data(), K, n, 1, m, 1};
  std::vector<idx_t> cells;
  for (idx_t c = 0; c < C; ++c)
    if (start[c + 1] > start[c]) cells.push_back(c);
  const idx_t NC = idx_t(cells.size());

  // Each pair is evaluated with the point of the outer loop as q. This is exact whichever
  // point plays q: swapping the two points only negates each difference (exact in floating
  // point), so "a above b" gives the same bits from either side. Witnesses and ties use the
  // original indices (later, earlier).
  const int nt = double(K) * double(K) > double(1 << 18) ? num_threads() : 1;
  std::vector<Need> needs(nt, Need(m, n));
  std::vector<std::vector<WorstPair>> worsts(nt, std::vector<WorstPair>(m));
  std::vector<std::vector<idx_t>> counts(nt, std::vector<idx_t>(m, 0));
  std::vector<idx_t> pairs(nt, 0);
  ErrorBox err;
#pragma omp parallel num_threads(nt)
  {
    const int t = thread_id();
    std::vector<double> du(n), a(n), b(n), zq(n), yq(m);
    std::vector<idx_t> cnt(m, 0);
    std::vector<WorstPair> wst(m);
    Need nd(m, n);
    idx_t my_pairs = 0;
#pragma omp for schedule(dynamic, 1)
    for (idx_t ia = 0; ia < NC; ++ia) {
      err.run([&] {
        const idx_t A = cells[ia];
        for (idx_t ib = ia; ib < NC; ++ib) {
          const idx_t B = cells[ib];
          if (prune && !box_box(s, &G.zlo[A * n], &G.zhi[A * n], &G.ylo[A * m], &G.yhi[A * m],
                                &G.zlo[B * n], &G.zhi[B * n], &G.ylo[B * m], &G.yhi[B * m],
                                a.data(), b.data()))
            continue;
          for (idx_t ra = start[A]; ra < start[A + 1]; ++ra) {
            const idx_t rb0 = (A == B) ? ra + 1 : start[B];
            if (rb0 >= start[B + 1]) continue;
            load_point(sv, ra, n, m, zq.data(), yq.data());
            if (prune && A != B &&
                !point_box(s, zq.data(), yq.data(), &G.zlo[B * n], &G.zhi[B * n],
                           &G.ylo[B * m], &G.yhi[B * m], a.data(), b.data()))
              continue;
            const idx_t ka = order[ra];
            for (idx_t rb = rb0; rb < start[B + 1]; ++rb) {
              const idx_t kb = order[rb];
              const idx_t kq = std::max(ka, kb), kk = std::min(ka, kb);
              ++my_pairs;
              bool any = false;
              eval_pair(s, zq.data(), yq.data(), sv, rb, nd, du.data(), [&](int i, double g) {
                any = true;
                ++cnt[i];
                wst[i].offer(g, kq, kk);
              });
              if (any) {
#pragma omp atomic
                ++out.conflicts[kq];
#pragma omp atomic
                ++out.conflicts[kk];
              }
            }
          }
        }
      });
    }
    needs[t] = nd;
    worsts[t] = wst;
    counts[t] = cnt;
    pairs[t] = my_pairs;
  }
  err.rethrow();
  for (int t = 0; t < nt; ++t) {
    out.need.merge(needs[t]);
    for (int i = 0; i < m; ++i) {
      out.count[i] += counts[t][i];
      if (worsts[t][i].a >= 0) out.worst[i].offer(worsts[t][i].gap, worsts[t][i].a, worsts[t][i].b);
    }
    out.pairs += pairs[t];
  }
  return out;
}

}  // namespace mon
}  // namespace ddma
