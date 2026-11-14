// ddma_growth.hpp -- header-only C++ kernels for growth-cone models (bounded derivatives),
// shared by the Python (pybind11) and MATLAB (MEX) interfaces of ddma.
//
// A data point (z_k, y_k) of f + d, with df/dz_j in [gl_j, gh_j] and d in [d_lo, d_hi]
// (theta = d_hi - d_lo), gives the growth cone
//
//     y_k - theta + phi_lo(z - z_k)  <=  f(z) + d  <=  y_k + theta + phi_hi(z - z_k),
//     phi_hi(t) = sum_j max(gh_j t_j, gl_j t_j),  phi_lo(t) = sum_j min(gh_j t_j, gl_j t_j).
//
// growth_vertices: tightest bounds at all grid vertices (thesis Ch. 3, eqs. 3.20-3.21).
//   Each point is written to the 2^n vertices of its cell, then one forward and one
//   backward running min/max per axis spreads it over the grid (exact, O(K 2^n + |V| n)).
//
// growth_local: per-cell bounds from the cones of the data, max of the cone over the cell
//   (= over its vertices), min over the points in the cell ("cell") or in the cell's slabs
//   ("slab": points whose cell index equals the cell's on at least one axis).
//   Along axis j, the cone term of a point on a cell to its right (left) is linear in one
//   edge coordinate of that cell. So for each "pattern" (per axis: left of / in / right of
//   the point's cell) the min over points is a scatter at the point's own cell followed by
//   an exclusive running min along the axes where the pattern says left/right. Cost
//   O(K n 3^n + |Q| n 3^n), independent of the slab size.
//
// Both kernels use threads (OpenMP, optional; see ddma_parallel.hpp); results are
// bit-identical for any number of threads.
//
// Conventions identical to ddma/_bd_components.py: cells [e_j[c], e_j[c+1]], points
// outside the box belong to the nearest edge cell; arrays are C order.
#pragma once

#include "ddma_core.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ddma {

inline std::ptrdiff_t cell_of(const Axis& a, double z) {
  std::ptrdiff_t c = search(a, z, true) - 1;
  return c < 0 ? 0 : (c > a.k - 1 ? a.k - 1 : c);
}

inline std::vector<Axis> make_axes(const std::vector<const double*>& edges,
                                   const std::vector<std::ptrdiff_t>& n_edges) {
  std::vector<Axis> ax;
  for (std::size_t j = 0; j < edges.size(); ++j) ax.push_back(make_axis(edges[j], n_edges[j]));
  return ax;
}

// vlo/vup: preallocated prod(n_edges) values; overwritten with the bounds from the data.
inline void growth_vertices(const double* Z, const double* y, std::ptrdiff_t K, std::ptrdiff_t n,
                            const std::vector<const double*>& edges,
                            const std::vector<std::ptrdiff_t>& n_edges, const double* gl,
                            const double* gh, double theta, double* vlo, double* vup) {
  const auto ax = make_axes(edges, n_edges);
  std::vector<std::ptrdiff_t> vstride(n, 1);
  for (std::ptrdiff_t j = n - 2; j >= 0; --j) vstride[j] = vstride[j + 1] * n_edges[j + 1];
  std::ptrdiff_t NV = 1;
  for (auto s : n_edges) NV *= s;
  const double INF = std::numeric_limits<double>::infinity();
  parallel_fill(vup, NV, INF);
  parallel_fill(vlo, NV, -INF);

  // scatter: each point to the 2^n vertices of its cell (minmax_scatter, ddma_parallel.hpp)
  const std::ptrdiff_t NC = std::ptrdiff_t(1) << n;
  auto make_gen = [&]() {
    // everything the generator reads, captured by value (kept in registers)
    return [Z, y, n, NC, gl, gh, theta, axes = ax.data(), vstride = vstride.data(),
            buf = std::vector<double>(4 * n)](std::ptrdiff_t k, auto& emit_up,
                                              auto& emit_lo) mutable {
      double* ph = buf.data();  // cone terms at the two edges of the cell, per axis
      double* pl = ph + 2 * n;
      const double* z = Z + k * n;
      std::ptrdiff_t base = 0;
      for (std::ptrdiff_t j = 0; j < n; ++j) {
        const std::ptrdiff_t c = cell_of(axes[j], z[j]);
        base += c * vstride[j];
        for (int b = 0; b < 2; ++b) {
          const double t = axes[j].e[c + b] - z[j];
          ph[2 * j + b] = std::max(gh[j] * t, gl[j] * t);
          pl[2 * j + b] = std::min(gh[j] * t, gl[j] * t);
        }
      }
      const double yu = y[k] + theta, yl = y[k] - theta;
      for (std::ptrdiff_t m = 0; m < NC; ++m) {
        // corner bit of axis j: bit (n-1-j), so corners run in itertools.product order
        double su = 0.0, sl = 0.0;
        std::ptrdiff_t q = base;
        for (std::ptrdiff_t j = 0; j < n; ++j) {
          const int b = int((m >> (n - 1 - j)) & 1);
          su += ph[2 * j + b];
          sl += pl[2 * j + b];
          q += b * vstride[j];
        }
        emit_up(q, yu + su);
        emit_lo(q, yl + sl);
      }
    };
  };
  // span: the cell's lowest and highest vertex
  std::ptrdiff_t far = 0;
  for (std::ptrdiff_t j = 0; j < n; ++j) far += vstride[j];
  auto make_span = [&]() {
    return [&](std::ptrdiff_t k, std::ptrdiff_t& first, std::ptrdiff_t& last) {
      const double* z = Z + k * n;
      std::ptrdiff_t base = 0;
      for (std::ptrdiff_t j = 0; j < n; ++j) base += cell_of(ax[j], z[j]) * vstride[j];
      first = base;
      last = base + far;
    };
  };
  minmax_scatter<double>(K, NV, vup, vlo, NC, make_gen, make_span);

  // sweeps, one axis at a time (same order of operations as the NumPy version). For axis j
  // the array is (outer, S, inner); the lines (o, r) are independent, and threads share
  // out blocks of them. w[r] carries the running min/max of line r.
  std::ptrdiff_t inner = NV;
  for (std::ptrdiff_t j = 0; j < n; ++j) {
    const std::ptrdiff_t S = n_edges[j];
    inner /= S;
    const std::ptrdiff_t outer = NV / (S * inner);
    const double* e = ax[j].e;
    const double ghj = gh[j], glj = gl[j];
    const std::ptrdiff_t rch = 4096, nr = (inner + rch - 1) / rch;
    parallel_tasks(outer * nr, NV, [&](std::ptrdiff_t t) {
      const std::ptrdiff_t o = t / nr, r0 = (t % nr) * rch, r1 = std::min(inner, r0 + rch);
      std::vector<double> w(r1 - r0);
      double* U = vup + o * S * inner;
      double* L = vlo + o * S * inner;
      // upper, forward: moving up costs gh per unit
      std::fill(w.begin(), w.end(), INF);
      for (std::ptrdiff_t s = 0; s < S; ++s)
        for (std::ptrdiff_t r = r0; r < r1; ++r) {
          double& u = U[s * inner + r];
          double& x = w[r - r0];
          x = std::min(x, u - ghj * e[s]);
          u = std::min(u, x + ghj * e[s]);
        }
      // upper, backward: moving down costs -gl per unit
      std::fill(w.begin(), w.end(), INF);
      for (std::ptrdiff_t s = S - 1; s >= 0; --s)
        for (std::ptrdiff_t r = r0; r < r1; ++r) {
          double& u = U[s * inner + r];
          double& x = w[r - r0];
          x = std::min(x, u - glj * e[s]);
          u = std::min(u, x + glj * e[s]);
        }
      // lower, forward (+gl per unit) and backward (-gh per unit)
      std::fill(w.begin(), w.end(), -INF);
      for (std::ptrdiff_t s = 0; s < S; ++s)
        for (std::ptrdiff_t r = r0; r < r1; ++r) {
          double& l = L[s * inner + r];
          double& x = w[r - r0];
          x = std::max(x, l - glj * e[s]);
          l = std::max(l, x + glj * e[s]);
        }
      std::fill(w.begin(), w.end(), -INF);
      for (std::ptrdiff_t s = S - 1; s >= 0; --s)
        for (std::ptrdiff_t r = r0; r < r1; ++r) {
          double& l = L[s * inner + r];
          double& x = w[r - r0];
          x = std::max(x, l - ghj * e[s]);
          l = std::max(l, x + ghj * e[s]);
        }
    });
  }
}

// lo/up: preallocated prod(n_edges - 1) values; overwritten with the local bounds.
// slab = false: points of the cell only ("cell"); true: points of the cell's slabs.
inline void growth_local(const double* Z, const double* y, std::ptrdiff_t K, std::ptrdiff_t n,
                         const std::vector<const double*>& edges,
                         const std::vector<std::ptrdiff_t>& n_edges, const double* gl,
                         const double* gh, double theta, bool slab, double* lo, double* up) {
  const auto ax = make_axes(edges, n_edges);
  std::vector<std::ptrdiff_t> shape(n), stride(n, 1);
  for (std::ptrdiff_t j = 0; j < n; ++j) shape[j] = ax[j].k;
  for (std::ptrdiff_t j = n - 2; j >= 0; --j) stride[j] = stride[j + 1] * shape[j + 1];
  std::ptrdiff_t NQ = 1;
  for (auto s : shape) NQ *= s;
  const double INF = std::numeric_limits<double>::infinity();

  // patterns: digit_j in {0 = left of, 1 = in, 2 = right of} the point's cell on axis j;
  // "cell" uses only all-1, "slab" every pattern with at least one 1.
  std::ptrdiff_t NP = 1;
  for (std::ptrdiff_t j = 0; j < n; ++j) NP *= 3;
  std::vector<std::vector<int>> pats;
  for (std::ptrdiff_t p = 0; p < NP; ++p) {
    std::vector<int> d(n);
    std::ptrdiff_t r = p;
    bool any_in = false, all_in = true;
    for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
      d[j] = int(r % 3);
      r /= 3;
      any_in = any_in || d[j] == 1;
      all_in = all_in && d[j] == 1;
    }
    if (slab ? any_in : all_in) pats.push_back(d);
  }
  const std::ptrdiff_t P = std::ptrdiff_t(pats.size());
  std::vector<int> pat_flat;  // pats as one array, pattern p at [p * n, (p + 1) * n)
  for (const auto& d : pats) pat_flat.insert(pat_flat.end(), d.begin(), d.end());
  const int* pat = pat_flat.data();

  // Cone term of a point at t = e - z (e an edge of a cell left / right of the point's
  // cell) is linear in e: upper: right gh*(e - z), left gl*(e - z); lower: right
  // gl*(e - z), left gh*(e - z). Max (upper) / min (lower) over the cell's two edges:
  // the edge index offset (0 = lower edge, 1 = upper edge) depends only on the sign.
  std::vector<int> ou_R(n), ou_L(n), ol_R(n), ol_L(n);
  for (std::ptrdiff_t j = 0; j < n; ++j) {
    ou_R[j] = gh[j] >= 0 ? 1 : 0;
    ou_L[j] = gl[j] >= 0 ? 1 : 0;
    ol_R[j] = gl[j] >= 0 ? 0 : 1;
    ol_L[j] = gh[j] >= 0 ? 0 : 1;
  }

  // Per pattern: one array over cells, layout (cell, pattern); scatter, sweeps, then
  // min/max over the patterns into the output.
  const std::ptrdiff_t NA = NQ * P;
  std::vector<double> AU(NA), AL(NA);
  parallel_fill(AU.data(), NA, INF);
  parallel_fill(AL.data(), NA, -INF);
  auto make_gen = [&]() {
    // everything the generator reads, captured by value (kept in registers)
    return [Z, y, n, P, gl, gh, theta, pat, stride = stride.data(), ax = ax.data(),
            buf = std::vector<double>(2 * n)](std::ptrdiff_t k, auto& emit_up,
                                              auto& emit_lo) mutable {
      double* own_u = buf.data();  // cone term over the point's own cell, per axis
      double* own_l = own_u + n;
      const double* z = Z + k * n;
      std::ptrdiff_t q = 0;
      for (std::ptrdiff_t j = 0; j < n; ++j) {
        const std::ptrdiff_t c = cell_of(ax[j], z[j]);
        q += c * stride[j];
        const double t0 = ax[j].e[c] - z[j], t1 = ax[j].e[c + 1] - z[j];
        own_u[j] = std::max(std::max(gh[j] * t0, gl[j] * t0), std::max(gh[j] * t1, gl[j] * t1));
        own_l[j] = std::min(std::min(gh[j] * t0, gl[j] * t0), std::min(gh[j] * t1, gl[j] * t1));
      }
      const double yu = y[k] + theta, yl = y[k] - theta;
      for (std::ptrdiff_t p = 0; p < P; ++p) {
        const int* d = pat + p * n;
        double su = yu, sl = yl;
        for (std::ptrdiff_t j = 0; j < n; ++j) {
          if (d[j] == 1) { su += own_u[j]; sl += own_l[j]; }
          else if (d[j] == 2) { su -= gh[j] * z[j]; sl -= gl[j] * z[j]; }
          else { su -= gl[j] * z[j]; sl -= gh[j] * z[j]; }
        }
        emit_up(q * P + p, su);
        emit_lo(q * P + p, sl);
      }
    };
  };
  // span: the P entries of the point's own cell
  auto make_span = [&]() {
    return [&](std::ptrdiff_t k, std::ptrdiff_t& first, std::ptrdiff_t& last) {
      const double* z = Z + k * n;
      std::ptrdiff_t q = 0;
      for (std::ptrdiff_t j = 0; j < n; ++j) q += cell_of(ax[j], z[j]) * stride[j];
      first = q * P;
      last = q * P + P - 1;
    };
  };
  minmax_scatter<double>(K, NA, AU.data(), AL.data(), P, make_gen, make_span);

  // Exclusive running min/max along axis j for the patterns with digit 0 / 2 on axis j,
  // adding the linear term of the destination cell's edge. For axis j the array is
  // (outer, S, inner, P); each line (o, r, p) is independent. A task takes one o and a
  // range of r, and walks the S cells in order, so memory is read contiguously over (r, p).
  std::ptrdiff_t inner = NQ;
  for (std::ptrdiff_t j = 0; j < n; ++j) {
    const std::ptrdiff_t S = shape[j];
    inner /= S;
    const std::ptrdiff_t outer = NQ / (S * inner);
    const double* e = ax[j].e;
    std::vector<std::ptrdiff_t> right_p, left_p;  // patterns with digit 2 / 0 on axis j
    for (std::ptrdiff_t p = 0; p < P; ++p) {
      if (pats[p][j] == 2) right_p.push_back(p);
      if (pats[p][j] == 0) left_p.push_back(p);
    }
    const std::ptrdiff_t rch = std::max<std::ptrdiff_t>(1, 2048 / P);
    const std::ptrdiff_t nr = (inner + rch - 1) / rch;
    parallel_tasks(outer * nr, NA, [&](std::ptrdiff_t t) {
      const std::ptrdiff_t o = t / nr, r0 = (t % nr) * rch, r1 = std::min(inner, r0 + rch);
      std::vector<double> runu, runl;
      for (int dir = 0; dir < 2; ++dir) {
        const bool right = dir == 0;
        const auto& ps = right ? right_p : left_p;
        if (ps.empty()) continue;
        const std::ptrdiff_t np_ = std::ptrdiff_t(ps.size());
        const double cu = right ? gh[j] : gl[j], cl = right ? gl[j] : gh[j];
        const int eu = right ? ou_R[j] : ou_L[j], el = right ? ol_R[j] : ol_L[j];
        runu.assign((r1 - r0) * np_, INF);
        runl.assign((r1 - r0) * np_, -INF);
        for (std::ptrdiff_t i = 0; i < S; ++i) {
          const std::ptrdiff_t s = right ? i : S - 1 - i;
          const double eus = cu * e[s + eu], els = cl * e[s + el];
          for (std::ptrdiff_t r = r0; r < r1; ++r) {
            double* au_row = AU.data() + ((o * S + s) * inner + r) * P;
            double* al_row = AL.data() + ((o * S + s) * inner + r) * P;
            double* ru = runu.data() + (r - r0) * np_;
            double* rl = runl.data() + (r - r0) * np_;
            for (std::ptrdiff_t x = 0; x < np_; ++x) {
              const std::ptrdiff_t p = ps[x];
              const double au = au_row[p], al = al_row[p];
              au_row[p] = ru[x] == INF ? INF : ru[x] + eus;
              al_row[p] = rl[x] == -INF ? -INF : rl[x] + els;
              ru[x] = std::min(ru[x], au);
              rl[x] = std::max(rl[x], al);
            }
          }
        }
      }
    });
  }
  // min / max over the patterns, in pattern order
  parallel_tasks((NQ + 4095) / 4096, NA, [&](std::ptrdiff_t t) {
    const std::ptrdiff_t q0 = t * 4096, q1 = std::min(NQ, q0 + 4096);
    for (std::ptrdiff_t q = q0; q < q1; ++q) {
      double u = INF, l = -INF;
      for (std::ptrdiff_t p = 0; p < P; ++p) {
        u = std::min(u, AU[q * P + p]);
        l = std::max(l, AL[q * P + p]);
      }
      up[q] = u;
      lo[q] = l;
    }
  });
}

}  // namespace ddma
