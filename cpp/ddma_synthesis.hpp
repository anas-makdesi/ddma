// ddma_synthesis.hpp -- header-only C++ fixed points for controller synthesis on symbolic
// abstractions, shared by the Python (pybind11) and MATLAB (MEX) interfaces of ddma.
//
// The abstraction stores, for every state cell q and input cell p, whether p is enabled at
// q and the box of successor cells [first(q, p), last(q, p)] (one index per coordinate).
// "All successors of (q, p) lie in W" is checked in O(2^n) with an n-D summed-area table
// of the complement of W.
//
// safety(): greatest fixed point (maximal controlled invariant subset of the safe cells).
//   Each state keeps a pointer to its first input not yet ruled out. W only shrinks, so an
//   input that fails once fails for ever and the pointer only moves forward: over the whole
//   fixed point every (state, input) pair is examined at most once, plus one check of the
//   current pointer per state and round. Exact: same set as the textbook iteration.
// reach_avoid(): least fixed point (Jacobi rounds, as in the textbook iteration, so the
//   step counts are the same). A state that failed is examined again only when a cell that
//   entered the reach set in the last round lies in the hull of all its successor boxes.
//
// Layout: first/last element (q, p, j) is at q*sq + p*sp + j*sj; enabled (q, p) at
// q*eq + p*ep. Cell masks are flat in C order over `shape`; coordinate j of first/last is
// axis dim[j] of `shape`. With these strides and dim, the same code reads NumPy C-order
// arrays and MATLAB column-major arrays (dim reverses the axes) without copies.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ddma {

template <class T>
struct PairBoxes {
  const T* first;
  const T* last;
  const std::uint8_t* enabled;
  std::ptrdiff_t nq, np, n;
  std::ptrdiff_t sq, sp, sj;  // strides of first/last (elements)
  std::ptrdiff_t eq, ep;      // strides of enabled
  std::vector<std::ptrdiff_t> dim;  // coordinate j -> axis of `shape`
};

// Summed-area table of a 0/1 cell mask over `shape` (C order); size prod(shape_i + 1).
class SummedArea {
 public:
  explicit SummedArea(const std::vector<std::ptrdiff_t>& shape) : shape_(shape) {
    const std::ptrdiff_t n = std::ptrdiff_t(shape.size());
    if (n < 1 || n > 20) throw std::invalid_argument("state dimension must be 1..20");
    stride_.assign(n, 1);
    cstride_.assign(n, 1);
    for (std::ptrdiff_t d = n - 2; d >= 0; --d) {
      stride_[d] = stride_[d + 1] * (shape[d + 1] + 1);
      cstride_[d] = cstride_[d + 1] * shape[d + 1];
    }
    size_ = stride_[0] * (shape[0] + 1);
    ncells_ = cstride_[0] * shape[0];
    t_.assign(size_, 0);
  }

  std::ptrdiff_t n_cells() const { return ncells_; }

  // count(c) = 1 for the cells to count (e.g. cells outside W)
  template <class F>
  void build(F count) {
    const std::ptrdiff_t n = std::ptrdiff_t(shape_.size());
    std::fill(t_.begin(), t_.end(), 0);
    std::vector<std::ptrdiff_t> sub(n, 0);
    for (std::ptrdiff_t c = 0; c < ncells_; ++c) {
      if (count(c)) {
        std::ptrdiff_t k = 0;
        for (std::ptrdiff_t d = 0; d < n; ++d) k += (sub[d] + 1) * stride_[d];
        t_[k] = 1;
      }
      for (std::ptrdiff_t d = n - 1; d >= 0; --d) {  // next C-order subscript
        if (++sub[d] < shape_[d]) break;
        sub[d] = 0;
      }
    }
    for (std::ptrdiff_t d = 0; d < n; ++d) {  // prefix sums along each axis (blocked)
      const std::ptrdiff_t s = stride_[d], S = shape_[d] + 1, outer = size_ / (S * s);
      for (std::ptrdiff_t o = 0; o < outer; ++o) {
        std::int64_t* b = t_.data() + o * S * s;
        for (std::ptrdiff_t c = 1; c < S; ++c)
          for (std::ptrdiff_t r = 0; r < s; ++r) b[c * s + r] += b[(c - 1) * s + r];
      }
    }
  }

  // number of counted cells in the box lo <= c <= hi (per axis of shape)
  std::int64_t box(const std::ptrdiff_t* lo, const std::ptrdiff_t* hi) const {
    const std::ptrdiff_t n = std::ptrdiff_t(shape_.size());
    std::int64_t s = 0;
    for (std::ptrdiff_t c = 0; c < (std::ptrdiff_t(1) << n); ++c) {
      std::ptrdiff_t k = 0;
      int lows = 0;
      for (std::ptrdiff_t d = 0; d < n; ++d) {
        if ((c >> d) & 1) k += (hi[d] + 1) * stride_[d];
        else { k += lo[d] * stride_[d]; ++lows; }
      }
      s += (lows & 1) ? -t_[k] : t_[k];
    }
    return s;
  }

  std::ptrdiff_t flat(const std::ptrdiff_t* sub) const {
    std::ptrdiff_t k = 0;
    for (std::size_t d = 0; d < shape_.size(); ++d) k += sub[d] * cstride_[d];
    return k;
  }

 private:
  std::vector<std::ptrdiff_t> shape_, stride_, cstride_;
  std::ptrdiff_t size_ = 0, ncells_ = 0;
  std::vector<std::int64_t> t_;
};

template <class T>
inline void check_view(const PairBoxes<T>& a, const std::vector<std::ptrdiff_t>& shape) {
  if (std::ptrdiff_t(shape.size()) != a.n || std::ptrdiff_t(a.dim.size()) != a.n)
    throw std::invalid_argument("first/last must have one coordinate per state axis");
  std::ptrdiff_t N = 1;
  for (auto s : shape) N *= s;
  if (N != a.nq) throw std::invalid_argument("number of state cells does not match the shape");
  std::vector<char> seen(a.n, 0);
  for (auto d : a.dim) {
    if (d < 0 || d >= a.n || seen[d]) throw std::invalid_argument("dim must be a permutation");
    seen[d] = 1;
  }
}

// successor box of (q, p) in axis order of `shape`
template <class T>
inline void pair_box(const PairBoxes<T>& a, std::ptrdiff_t q, std::ptrdiff_t p,
                     std::ptrdiff_t* lo, std::ptrdiff_t* hi) {
  const std::ptrdiff_t base = q * a.sq + p * a.sp;
  for (std::ptrdiff_t j = 0; j < a.n; ++j) {
    lo[a.dim[j]] = std::ptrdiff_t(a.first[base + j * a.sj]);
    hi[a.dim[j]] = std::ptrdiff_t(a.last[base + j * a.sj]);
  }
}

// good(q, p) = enabled(q, p) and all successors in W. good uses the strides of enabled.
template <class T>
inline void box_all_in(const PairBoxes<T>& a, const std::vector<std::ptrdiff_t>& shape,
                       const std::uint8_t* W, std::uint8_t* good) {
  check_view(a, shape);
  SummedArea sat(shape);
  sat.build([&](std::ptrdiff_t c) { return !W[c]; });
  std::vector<std::ptrdiff_t> lo(a.n), hi(a.n);
  for (std::ptrdiff_t q = 0; q < a.nq; ++q)
    for (std::ptrdiff_t p = 0; p < a.np; ++p) {
      const std::ptrdiff_t e = q * a.eq + p * a.ep;
      std::uint8_t g = 0;
      if (a.enabled[e]) {
        pair_box(a, q, p, lo.data(), hi.data());
        g = sat.box(lo.data(), hi.data()) == 0;
      }
      good[e] = g;
    }
}

// Maximal controlled invariant subset of safe. W (in/out): safe on entry, winning set on
// exit. allowed (optional, strides of enabled): safe inputs at winning states.
// Returns the number of rounds.
template <class T>
inline std::ptrdiff_t safety(const PairBoxes<T>& a, const std::vector<std::ptrdiff_t>& shape,
                             std::uint8_t* W, std::uint8_t* allowed,
                             std::ptrdiff_t max_rounds = 100000) {
  check_view(a, shape);
  SummedArea sat(shape);
  std::vector<std::ptrdiff_t> ptr(a.nq, 0), lo(a.n), hi(a.n), dead;
  std::ptrdiff_t rounds = 0;
  for (;;) {
    if (++rounds > max_rounds) throw std::runtime_error("safety fixed point did not converge");
    sat.build([&](std::ptrdiff_t c) { return !W[c]; });
    dead.clear();
    for (std::ptrdiff_t q = 0; q < a.nq; ++q) {
      if (!W[q]) continue;
      std::ptrdiff_t& p = ptr[q];
      for (; p < a.np; ++p) {
        if (!a.enabled[q * a.eq + p * a.ep]) continue;
        pair_box(a, q, p, lo.data(), hi.data());
        if (sat.box(lo.data(), hi.data()) == 0) break;
      }
      if (p == a.np) dead.push_back(q);
    }
    if (dead.empty()) break;
    for (auto q : dead) W[q] = 0;
  }
  if (allowed) {
    // sat is the table of the final W; inputs before ptr[q] are known to fail
    for (std::ptrdiff_t q = 0; q < a.nq; ++q)
      for (std::ptrdiff_t p = 0; p < a.np; ++p) {
        const std::ptrdiff_t e = q * a.eq + p * a.ep;
        std::uint8_t g = 0;
        if (W[q] && p >= ptr[q] && a.enabled[e]) {
          pair_box(a, q, p, lo.data(), hi.data());
          g = sat.box(lo.data(), hi.data()) == 0;
        }
        allowed[e] = g;
      }
  }
  return rounds;
}

// States that can be driven to target while avoiding avoid. R (out): winning set;
// value (out): steps to the target (inf if unreachable); allowed (out, strides of enabled):
// at a state of value k > 0, the inputs whose successors all have value < k.
template <class T>
inline std::ptrdiff_t reach_avoid(const PairBoxes<T>& a, const std::vector<std::ptrdiff_t>& shape,
                                  const std::uint8_t* target, const std::uint8_t* avoid,
                                  std::uint8_t* R, double* value, std::uint8_t* allowed,
                                  std::ptrdiff_t max_rounds = 100000) {
  check_view(a, shape);
  const double INF = std::numeric_limits<double>::infinity();
  const std::ptrdiff_t n = a.n;
  // hull of the successor boxes of each undecided state (axis order of shape)
  std::vector<std::ptrdiff_t> hlo(a.nq * n), hhi(a.nq * n), lo(n), hi(n), todo, fresh;
  for (std::ptrdiff_t q = 0; q < a.nq; ++q) {
    R[q] = target[q] && !avoid[q];
    value[q] = R[q] ? 0.0 : INF;
    for (std::ptrdiff_t p = 0; p < a.np; ++p) allowed[q * a.eq + p * a.ep] = 0;
    if (R[q] || avoid[q]) continue;
    bool any = false;
    for (std::ptrdiff_t p = 0; p < a.np; ++p) {
      if (!a.enabled[q * a.eq + p * a.ep]) continue;
      pair_box(a, q, p, lo.data(), hi.data());
      for (std::ptrdiff_t d = 0; d < n; ++d) {
        if (!any || lo[d] < hlo[q * n + d]) hlo[q * n + d] = lo[d];
        if (!any || hi[d] > hhi[q * n + d]) hhi[q * n + d] = hi[d];
      }
      any = true;
    }
    if (any) todo.push_back(q);  // states without enabled inputs can never be added
  }
  SummedArea outside(shape), added(shape);
  std::vector<std::uint8_t> is_new(R, R + a.nq);  // cells that entered R last round
  std::ptrdiff_t k = 0;
  for (;;) {
    if (++k > max_rounds) throw std::runtime_error("reach-avoid fixed point did not converge");
    outside.build([&](std::ptrdiff_t c) { return !R[c]; });
    added.build([&](std::ptrdiff_t c) { return is_new[c] != 0; });
    fresh.clear();
    std::ptrdiff_t keep = 0;
    for (std::size_t i = 0; i < todo.size(); ++i) {
      const std::ptrdiff_t q = todo[i];
      bool won = false;
      if (added.box(&hlo[q * n], &hhi[q * n]) > 0) {
        for (std::ptrdiff_t p = 0; p < a.np; ++p) {
          const std::ptrdiff_t e = q * a.eq + p * a.ep;
          if (!a.enabled[e]) continue;
          pair_box(a, q, p, lo.data(), hi.data());
          if (outside.box(lo.data(), hi.data()) == 0) { allowed[e] = 1; won = true; }
        }
      }
      if (won) fresh.push_back(q);
      else todo[keep++] = q;
    }
    todo.resize(keep);
    if (fresh.empty()) break;
    std::fill(is_new.begin(), is_new.end(), 0);
    for (auto q : fresh) { R[q] = 1; value[q] = double(k); is_new[q] = 1; }
  }
  return k;
}

}  // namespace ddma
