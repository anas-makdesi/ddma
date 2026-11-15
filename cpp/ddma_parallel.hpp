// ddma_parallel.hpp -- threads and cache-friendly scattering for the learning kernels.
//
// Every learning kernel of ddma has the same two phases:
//   1. scatter: each data point writes a few values into a big grid array, keeping the
//      smallest (upper bounds) or the largest (lower bounds) value seen in each cell;
//   2. sweeps: running min / max along the axes of the grid.
// This header provides the scatter engine (minmax_scatter) and the thread settings.
//
// Why results are bit-identical for any number of threads: a cell only ever keeps
// min(old, new) or max(old, new). Min and max are exact (no rounding) and give the same
// answer in any order, so it does not matter which thread writes first. The sweeps
// process every grid line with exactly the same arithmetic in the same order; threads
// only split *which* lines they handle.
//
// OpenMP is optional: without it (e.g. Apple clang) the pragmas are ignored and every
// loop runs on one thread, with the same results.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ddma {

// ------------------------------------------------------------------ thread settings
// 0 = OpenMP's default (OMP_NUM_THREADS, else all cores).
inline int& threads_setting() {
  static int v = 0;
  return v;
}
inline void set_num_threads(int n) { threads_setting() = n < 0 ? 0 : n; }
inline int num_threads() {
#ifdef _OPENMP
  const int v = threads_setting();
  return v > 0 ? v : omp_get_max_threads();
#else
  return 1;
#endif
}
inline bool openmp_available() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}
inline int thread_id() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}
inline int team_size() {
#ifdef _OPENMP
  return omp_get_num_threads();
#else
  return 1;
#endif
}

// How minmax_scatter writes the points (an internal switch for benchmarks and tests;
// all modes give the same result).
enum ScatterMode : int {
  SCATTER_AUTO = 0,
  SCATTER_DIRECT = 1,    // one thread, each value written straight into the grid (v0.8)
  SCATTER_BUCKETED = 2,  // group values by block of the grid, then fill block by block
  SCATTER_PRIVATE = 3,   // each thread fills its own copy of the grid, copies merged
  SCATTER_OWNED = 4,     // grid cut into parts; points sorted by part; one thread per part
};
inline int& scatter_mode() {
  static int v = SCATTER_AUTO;
  return v;
}

// Tuning constants (bytes). A block of the grid (upper + lower parts) should sit in the
// per-core L2 cache while it is filled.
constexpr std::size_t kBlockBytes = std::size_t(256) << 10;   // 256 KB
constexpr std::size_t kPrivateBytes = std::size_t(4) << 20;   // small grid: copies <= 4 MB
constexpr std::ptrdiff_t kMaxBlocks = 4096;
// values per round (bucketed) / points per round (owned); internal knob for tests
inline std::ptrdiff_t& chunk_entries() {
  static std::ptrdiff_t v = std::ptrdiff_t(1) << 20;
  return v;
}
constexpr std::ptrdiff_t kSerialEntries = 1 << 14;  // fewer values: one thread, direct

// Split [0, n) into `parts` contiguous ranges; range number t.
inline void split_range(std::ptrdiff_t n, int parts, int t, std::ptrdiff_t& a,
                        std::ptrdiff_t& b) {
  a = n * t / parts;
  b = n * (t + 1) / parts;
}

// Fill a[0..n) with v, in parallel for big arrays.
template <class T> inline void parallel_fill(T* a, std::ptrdiff_t n, T v) {
  const int nt = n > (std::ptrdiff_t(1) << 16) ? num_threads() : 1;
#pragma omp parallel num_threads(nt)
  {
    std::ptrdiff_t lo, hi;
    split_range(n, team_size(), thread_id(), lo, hi);
    std::fill(a + lo, a + hi, v);
  }
}

// ------------------------------------------------------------------ scatter engine
// One thread, every value written straight into the grid. Kept in its own function (no
// OpenMP inside), so the compiler optimises the loop as well as in the v0.8 kernel. The
// "flat" version also inlines everything it calls (GCC's flatten): measured faster for
// fit_sigma inside the big Python module, slower for growth_local, so each kernel picks.
#if defined(__GNUC__) && !defined(__clang__)
#define DDMA_FLATTEN __attribute__((flatten))
#else
#define DDMA_FLATTEN
#endif
template <class T, class MakeGen>
void scatter_direct(std::ptrdiff_t K, T* up, T* lo, MakeGen& make_gen) {
  auto g = make_gen();
  auto eu = [up](std::ptrdiff_t i, T v) { up[i] = std::min(up[i], v); };
  auto el = [lo](std::ptrdiff_t i, T v) { lo[i] = std::max(lo[i], v); };
  for (std::ptrdiff_t k = 0; k < K; ++k) g(k, eu, el);
}
template <class T, class MakeGen>
DDMA_FLATTEN void scatter_direct_flat(std::ptrdiff_t K, T* up, T* lo, MakeGen& make_gen) {
  auto g = make_gen();
  auto eu = [up](std::ptrdiff_t i, T v) { up[i] = std::min(up[i], v); };
  auto el = [lo](std::ptrdiff_t i, T v) { lo[i] = std::max(lo[i], v); };
  for (std::ptrdiff_t k = 0; k < K; ++k) g(k, eu, el);
}

// up, lo: arrays of N values (already initialised). For each point k in [0, K), the
// generator emits values:  emit_up(i, v)  ->  up[i] = min(up[i], v)
//                          emit_lo(i, v)  ->  lo[i] = max(lo[i], v)
// `make_gen()` returns a fresh generator g(k, emit_up, emit_lo); each thread makes its own
// (so it can keep scratch space). `per_point` = the most values one point emits per array.
// `make_span()` (optional) returns span(k, first, last): the smallest range of indices
// [first, last] point k writes to, cheaper than running g. Kernels whose points write many
// values (growth cones) give one; on big grids they then use the "owned" mode. Kernels
// whose points write few values (fit_sigma) use "bucketed" on big grids, also on one
// thread. These choices come from measurements on a laptop (benchmarks/kernels.py);
// every mode gives the same result.
//
// A NaN value is never kept (min(x, NaN) = x as written below), so NaN data are ignored
// in every mode alike.
struct NoSpan {};
template <class T, bool Flat = false, class MakeGen, class MakeSpan = NoSpan>
void minmax_scatter(std::ptrdiff_t K, std::ptrdiff_t N, T* up, T* lo, std::ptrdiff_t per_point,
                    MakeGen make_gen, MakeSpan make_span = MakeSpan()) {
  constexpr bool has_span = !std::is_same<MakeSpan, NoSpan>::value;
  if (K <= 0) return;
  const int nt = num_threads();
  int mode = scatter_mode();
  const std::size_t grid_bytes = 2 * std::size_t(N) * sizeof(T);
  if (mode == SCATTER_AUTO) {
    if (K * per_point <= kSerialEntries)
      mode = SCATTER_DIRECT;  // small update: threads would cost more than they save
    else if (grid_bytes <= kBlockBytes)
      mode = nt > 1 ? SCATTER_PRIVATE : SCATTER_DIRECT;  // whole grid already in cache
    else if (!has_span)
      // few values per point (fit_sigma): sorting by block keeps the writes in cache;
      // measured fastest on a laptop for 1 to 4 threads (up to 2x on one thread)
      mode = (nt > 1 && grid_bytes <= kPrivateBytes) ? SCATTER_PRIVATE : SCATTER_BUCKETED;
    else if (nt == 1)
      mode = SCATTER_DIRECT;  // many values per point: too many to sort
    else if (grid_bytes <= kPrivateBytes)
      mode = SCATTER_PRIVATE;
    else
      mode = SCATTER_OWNED;  // many values per point: parts, not copies to merge
  }
  if (mode == SCATTER_OWNED && !has_span) mode = SCATTER_BUCKETED;

  // -- direct: one thread, straight into the grid (the v0.8 kernel)
  if (mode == SCATTER_DIRECT) {
    if constexpr (Flat) scatter_direct_flat<T>(K, up, lo, make_gen);
    else scatter_direct<T>(K, up, lo, make_gen);
    return;
  }

  // -- private: thread 0 writes into the grid, the others into their own copies,
  //    then the copies are merged into the grid
  if (mode == SCATTER_PRIVATE) {
    const T INF = std::numeric_limits<T>::infinity();
    std::vector<std::vector<T>> cu(nt), cl(nt);
#pragma omp parallel num_threads(nt)
    {
      const int t = thread_id(), T_ = team_size();
      T* pu = up;
      T* pl = lo;
      if (t > 0) {
        cu[t].assign(N, INF);
        cl[t].assign(N, -INF);
        pu = cu[t].data();
        pl = cl[t].data();
      }
      auto g = make_gen();
      auto eu = [pu](std::ptrdiff_t i, T v) { pu[i] = std::min(pu[i], v); };
      auto el = [pl](std::ptrdiff_t i, T v) { pl[i] = std::max(pl[i], v); };
      std::ptrdiff_t a, b;
      split_range(K, T_, t, a, b);
      for (std::ptrdiff_t k = a; k < b; ++k) g(k, eu, el);
#pragma omp barrier
      // merge: each thread owns a range of cells
      std::ptrdiff_t qa, qb;
      split_range(N, T_, t, qa, qb);
      for (int s = 1; s < T_; ++s) {
        const T* su = cu[s].data();
        const T* sl = cl[s].data();
        for (std::ptrdiff_t q = qa; q < qb; ++q) {
          if (su[q] < up[q]) up[q] = su[q];
          if (lo[q] < sl[q]) lo[q] = sl[q];
        }
      }
    }
    return;
  }

  // -- owned. The grid is cut into R parts (a few per thread). Points are handled in rounds;
  // in each round every thread finds, for its share of the points, which parts they write
  // to (span), and lists the point under those parts. Then the parts are shared out among
  // the threads: a thread runs the generator for the points listed under its part and
  // keeps only the values that fall inside the part. No two threads touch the same cell.
  if constexpr (has_span) {
    if (mode == SCATTER_OWNED) {
      const std::ptrdiff_t R = std::min<std::ptrdiff_t>(N, 4 * std::ptrdiff_t(nt));
      std::vector<std::ptrdiff_t> bound(R + 1);  // part r = [bound[r], bound[r + 1])
      for (std::ptrdiff_t r = 0; r <= R; ++r) bound[r] = N * r / R;
      auto part_of = [&](std::ptrdiff_t i) {
        return std::ptrdiff_t(std::upper_bound(bound.begin(), bound.end(), i) - bound.begin()) - 1;
      };
      const std::ptrdiff_t chunk = std::min(K, chunk_entries());  // points per round
      std::vector<std::vector<std::vector<std::uint32_t>>> lists(
          nt, std::vector<std::vector<std::uint32_t>>(R));
#pragma omp parallel num_threads(nt)
      {
        const int t = thread_id(), T_ = team_size();
        auto g = make_gen();
        auto sp = make_span();
        for (std::ptrdiff_t k0 = 0; k0 < K; k0 += chunk) {
          const std::ptrdiff_t k1 = std::min(K, k0 + chunk);
          std::ptrdiff_t a, b;
          split_range(k1 - k0, T_, t, a, b);
          for (auto& l : lists[t]) l.clear();
          for (std::ptrdiff_t k = k0 + a; k < k0 + b; ++k) {
            std::ptrdiff_t first, last;
            sp(k, first, last);
            const std::ptrdiff_t r1 = part_of(last);
            for (std::ptrdiff_t r = part_of(first); r <= r1; ++r)
              lists[t][r].push_back(std::uint32_t(k - k0));
          }
#pragma omp barrier
#pragma omp for schedule(dynamic, 1)
          for (std::ptrdiff_t r = 0; r < R; ++r) {
            const std::ptrdiff_t qa = bound[r], qb = bound[r + 1];
            auto eu = [up, qa, qb](std::ptrdiff_t i, T v) {
              if (i >= qa && i < qb) up[i] = std::min(up[i], v);
            };
            auto el = [lo, qa, qb](std::ptrdiff_t i, T v) {
              if (i >= qa && i < qb) lo[i] = std::max(lo[i], v);
            };
            for (int s = 0; s < T_; ++s)
              for (auto k : lists[s][r]) g(k0 + k, eu, el);
          }  // implicit barrier: lists can be reused
        }
      }
      return;
    }
  }

  // -- bucketed. The grid is cut into B blocks of 2^shift cells. Points are handled in
  // rounds of about kChunkEntries values. In each round:
  //   (a) every thread computes the values of its share of the points (in a local list);
  //   (b) the values are sorted by block (a counting sort, stable);
  //   (c) the blocks are shared out among the threads; a block's values are written while
  //       that block of the grid sits in the cache, and no two threads touch the same cell.
  int shift = 0;
  while ((std::size_t(2) << shift) * 2 * sizeof(T) <= kBlockBytes) ++shift;  // cells/block
  while (((N - 1) >> shift) + 1 > kMaxBlocks) ++shift;
  const std::ptrdiff_t B = ((N - 1) >> shift) + 1;
  const std::ptrdiff_t mask = (std::ptrdiff_t(1) << shift) - 1;
  const std::ptrdiff_t chunk = std::max<std::ptrdiff_t>(1, chunk_entries() / std::max<std::ptrdiff_t>(1, per_point));

  struct Item {
    std::ptrdiff_t i;
    T v;
  };
  std::vector<std::vector<Item>> loc_u(nt), loc_l(nt);
  std::vector<std::ptrdiff_t> cnt_u(std::size_t(nt) * B), cnt_l(std::size_t(nt) * B);
  std::vector<std::ptrdiff_t> start_u(B + 1), start_l(B + 1);
  std::vector<std::uint32_t> off_u, off_l;  // position inside the block
  std::vector<T> val_u, val_l;
  off_u.resize(std::min(K, chunk) * per_point);
  off_l.resize(off_u.size());
  val_u.resize(off_u.size());
  val_l.resize(off_u.size());

#pragma omp parallel num_threads(nt)
  {
    const int t = thread_id(), T_ = team_size();
    auto g = make_gen();
    auto& LU = loc_u[t];
    auto& LL = loc_l[t];
    LU.reserve(std::min(K, chunk) * per_point / T_ + per_point);
    LL.reserve(LU.capacity());
    std::ptrdiff_t* cu = cnt_u.data() + std::size_t(t) * B;
    std::ptrdiff_t* cl = cnt_l.data() + std::size_t(t) * B;
    // values are listed and counted per block as they are emitted
    auto eu = [&LU, cu, shift](std::ptrdiff_t i, T v) {
      LU.push_back({i, v});
      ++cu[i >> shift];
    };
    auto el = [&LL, cl, shift](std::ptrdiff_t i, T v) {
      LL.push_back({i, v});
      ++cl[i >> shift];
    };
    for (std::ptrdiff_t k0 = 0; k0 < K; k0 += chunk) {
      const std::ptrdiff_t k1 = std::min(K, k0 + chunk);
      // (a) values of this thread's points, counted per block
      std::ptrdiff_t a, b;
      split_range(k1 - k0, T_, t, a, b);
      LU.clear();
      LL.clear();
      std::fill(cu, cu + B, 0);
      std::fill(cl, cl + B, 0);
      for (std::ptrdiff_t k = k0 + a; k < k0 + b; ++k) g(k, eu, el);
#pragma omp barrier
#pragma omp single
      {
        // where each (block, thread) group starts: blocks in order, threads in order
        std::ptrdiff_t su = 0, sl = 0;
        for (std::ptrdiff_t blk = 0; blk < B; ++blk) {
          start_u[blk] = su;
          start_l[blk] = sl;
          for (int s = 0; s < T_; ++s) {
            std::ptrdiff_t& x = cnt_u[std::size_t(s) * B + blk];
            std::ptrdiff_t& y = cnt_l[std::size_t(s) * B + blk];
            const std::ptrdiff_t nx = x, ny = y;
            x = su;  // counts become write positions
            y = sl;
            su += nx;
            sl += ny;
          }
        }
        start_u[B] = su;
        start_l[B] = sl;
      }
      // (b) counting sort into the shared lists
      for (const auto& it : LU) {
        const std::ptrdiff_t p = cu[it.i >> shift]++;
        off_u[p] = std::uint32_t(it.i & mask);
        val_u[p] = it.v;
      }
      for (const auto& it : LL) {
        const std::ptrdiff_t p = cl[it.i >> shift]++;
        off_l[p] = std::uint32_t(it.i & mask);
        val_l[p] = it.v;
      }
#pragma omp barrier
      // (c) fill the grid block by block
#pragma omp for schedule(dynamic, 1)
      for (std::ptrdiff_t blk = 0; blk < B; ++blk) {
        T* bu = up + (blk << shift);
        T* bl = lo + (blk << shift);
        for (std::ptrdiff_t p = start_u[blk]; p < start_u[blk + 1]; ++p) {
          T& x = bu[off_u[p]];
          if (val_u[p] < x) x = val_u[p];
        }
        for (std::ptrdiff_t p = start_l[blk]; p < start_l[blk + 1]; ++p) {
          T& x = bl[off_l[p]];
          if (x < val_l[p]) x = val_l[p];
        }
      }  // implicit barrier: the shared lists can be reused in the next round
    }
  }
}

// Run body(task) for task in [0, n_tasks), shared among the threads (static order of
// tasks inside each thread; the result must not depend on which thread runs a task).
template <class F>
inline void parallel_tasks(std::ptrdiff_t n_tasks, std::ptrdiff_t work, F body) {
  const int nt = (work > (std::ptrdiff_t(1) << 15) && n_tasks > 1) ? num_threads() : 1;
#pragma omp parallel for num_threads(nt) schedule(static)
  for (std::ptrdiff_t t = 0; t < n_tasks; ++t) body(t);
}

}  // namespace ddma
