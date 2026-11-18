// ddma_monitor_mex.cpp -- pair checks of the assumption monitor and of ddma.check_data, MEX.
// The kernels are those of the Python module (cpp/ddma_monitor.hpp); same results.
//
// All indices passed in and out are 0-based (the MATLAB wrappers convert). Z, Y are the
// monitor's storage (cap x n, cap x m, column-major); only the first K rows are used.
// lo, hi: m x n slope bounds; theta: 1 x m; opts: ddma.internal.mex_opts() (threads).
//
// [any, w, nh, nho, nl, nlo, worst, wit] =
//     ddma_monitor_mex('list', lo, hi, theta, tol, Z, Y, K, z, y, idx, status, opts)
// [any, w, nh, nho, nl, nlo, worst, wit, inr, gaps, pairs] =
//     ddma_monitor_mex('full', lo, hi, theta, tol, Z, Y, K, status, first, last, grid, prune, opts)
//     grid: [] or the struct of 'grid'; worst, wit: P x m; inr: R x 3 [q k i]; gaps: R x 1
// [any, w, nh, nho, nl, nlo, count, worst, pair, conflicts, pairs] =
//     ddma_monitor_mex('all', lo, hi, theta, tol, Z, Y, prune, opts)
// G = ddma_monitor_mex('grid', Z, Y, K)
//     struct: cdim (n x 1), glo, ginv (n x 1), zlo, zhi (n x C), ylo, yhi (m x C), and
//     int64 cnt, head (C x 1), next (K x 1), cell (K x 1); lists end with -1.
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "ddma_mex_opts.hpp"
#include "ddma_monitor.hpp"
#include "mex.h"

namespace {

using ddma::mon::idx_t;

void need_double(const mxArray* a, const char* what) {
  if (!mxIsDouble(a) || mxIsComplex(a) || mxIsSparse(a))
    mexErrMsgIdAndTxt("ddma:monitor_mex", "%s must be a real double array", what);
}

double scalar(const mxArray* a, const char* what) {
  need_double(a, what);
  if (mxGetNumberOfElements(a) != 1) mexErrMsgIdAndTxt("ddma:monitor_mex", "%s must be a scalar", what);
  return mxGetScalar(a);
}

// Spec from MATLAB m x n slope bounds (column-major) -> row-major copies.
struct SpecHolder {
  std::vector<double> lo, hi, theta;
  ddma::mon::Spec s;
};

void make_spec(SpecHolder& h, const mxArray* lo, const mxArray* hi, const mxArray* th,
               const mxArray* tol) {
  need_double(lo, "lo");
  need_double(hi, "hi");
  need_double(th, "theta");
  const mwSize m = mxGetM(lo), n = mxGetN(lo);
  if (mxGetM(hi) != m || mxGetN(hi) != n) mexErrMsgIdAndTxt("ddma:monitor_mex", "lo and hi must be m x n");
  if (mxGetNumberOfElements(th) != m) mexErrMsgIdAndTxt("ddma:monitor_mex", "theta must have m entries");
  const double* L = mxGetPr(lo);
  const double* H = mxGetPr(hi);
  h.lo.resize(m * n);
  h.hi.resize(m * n);
  for (mwSize i = 0; i < m; ++i)
    for (mwSize j = 0; j < n; ++j) {
      h.lo[i * n + j] = L[i + j * m];
      h.hi[i * n + j] = H[i + j * m];
    }
  h.theta.assign(mxGetPr(th), mxGetPr(th) + m);
  h.s = {int(m), int(n), h.lo.data(), h.hi.data(), h.theta.data(), scalar(tol, "tol")};
}

ddma::mon::View make_view(const mxArray* Z, const mxArray* Y, idx_t K, int n, int m) {
  need_double(Z, "Z");
  need_double(Y, "Y");
  const idx_t cap = idx_t(mxGetM(Z));
  if (idx_t(mxGetN(Z)) != n || idx_t(mxGetN(Y)) != m || idx_t(mxGetM(Y)) != cap || K > cap || K < 0)
    mexErrMsgIdAndTxt("ddma:monitor_mex", "Z must be cap x n and Y cap x m, with K <= cap rows used");
  return {mxGetPr(Z), mxGetPr(Y), K, 1, cap, 1, cap};
}

const std::uint8_t* status_of(const mxArray* a, idx_t K) {
  if (!mxIsUint8(a) || idx_t(mxGetNumberOfElements(a)) < K)
    mexErrMsgIdAndTxt("ddma:monitor_mex", "status must be a uint8 vector with at least K entries");
  return static_cast<const std::uint8_t*>(mxGetData(a));
}

const idx_t* int64_of(const mxArray* s, const char* name, idx_t at_least) {
  const mxArray* a = mxGetField(s, 0, name);
  if (!a || !mxIsInt64(a) || idx_t(mxGetNumberOfElements(a)) < at_least)
    mexErrMsgIdAndTxt("ddma:monitor_mex", "grid.%s must be int64 with at least %lld entries", name,
                      (long long)at_least);
  return static_cast<const idx_t*>(mxGetData(a));
}

const double* double_of(const mxArray* s, const char* name, idx_t at_least) {
  const mxArray* a = mxGetField(s, 0, name);
  if (!a || !mxIsDouble(a) || idx_t(mxGetNumberOfElements(a)) < at_least)
    mexErrMsgIdAndTxt("ddma:monitor_mex", "grid.%s must be double with at least %lld entries", name,
                      (long long)at_least);
  return mxGetPr(a);
}

ddma::mon::Grid grid_of(const mxArray* s, int n, int m, idx_t K) {
  if (!mxIsStruct(s)) mexErrMsgIdAndTxt("ddma:monitor_mex", "grid must be [] or a struct");
  ddma::mon::Grid G;
  G.n = n;
  G.m = m;
  const double* cd = double_of(s, "cdim", n);
  std::int64_t C = 1;
  for (int j = 0; j < n; ++j) C *= std::int64_t(cd[j]);
  G.C = C;
  static thread_local std::vector<idx_t> cdim;
  cdim.assign(n, 0);
  for (int j = 0; j < n; ++j) cdim[j] = idx_t(cd[j]);
  G.cdim = cdim.data();
  G.glo = double_of(s, "glo", n);
  G.ginv = double_of(s, "ginv", n);
  G.zlo = double_of(s, "zlo", C * n);
  G.zhi = double_of(s, "zhi", C * n);
  G.ylo = double_of(s, "ylo", C * m);
  G.yhi = double_of(s, "yhi", C * m);
  G.cnt = int64_of(s, "cnt", C);
  G.head = int64_of(s, "head", C);
  G.next = int64_of(s, "next", K);
  for (idx_t c = 0; c < C; ++c)
    if (G.head[c] >= K) mexErrMsgIdAndTxt("ddma:monitor_mex", "grid lists point beyond the data");
  return G;
}

// need -> plhs[0..5]: any (1 x m logical), w (1 x m), nh, nho, nl, nlo (m x n).
void put_need(mxArray* plhs[], const ddma::mon::Need& N) {
  const int m = N.m, n = N.n;
  plhs[0] = mxCreateLogicalMatrix(1, m);
  plhs[1] = mxCreateDoubleMatrix(1, m, mxREAL);
  for (int i = 0; i < m; ++i) {
    mxGetLogicals(plhs[0])[i] = N.any[i] != 0;
    mxGetPr(plhs[1])[i] = N.width_r[i];
  }
  plhs[2] = mxCreateDoubleMatrix(m, n, mxREAL);
  plhs[3] = mxCreateLogicalMatrix(m, n);
  plhs[4] = mxCreateDoubleMatrix(m, n, mxREAL);
  plhs[5] = mxCreateLogicalMatrix(m, n);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      const std::size_t r = std::size_t(i) * n + j, c = i + std::size_t(j) * m;
      mxGetPr(plhs[2])[c] = N.hi[r];
      mxGetLogicals(plhs[3])[c] = N.hi_out[r] != 0;
      mxGetPr(plhs[4])[c] = N.lo[r];
      mxGetLogicals(plhs[5])[c] = N.lo_out[r] != 0;
    }
}

mxArray* int64_col(const std::vector<idx_t>& v) {
  mxArray* a = mxCreateNumericMatrix(v.size(), 1, mxINT64_CLASS, mxREAL);
  if (!v.empty()) std::memcpy(mxGetData(a), v.data(), v.size() * sizeof(idx_t));
  return a;
}

mxArray* double_mat(const std::vector<double>& v, mwSize r, mwSize c) {
  mxArray* a = mxCreateDoubleMatrix(r, c, mxREAL);
  if (!v.empty()) std::memcpy(mxGetPr(a), v.data(), v.size() * sizeof(double));
  return a;
}

}  // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  if (nrhs >= 1 && mxIsChar(prhs[0])) {
    char buf[16];
    mxGetString(prhs[0], buf, sizeof(buf));
    const std::string mode(buf);
    try {
      if (mode == "info") {
        ddma_info_call(nlhs, plhs, nrhs, prhs);
        return;
      }
      if (mode == "list") {
        if (nrhs < 12) mexErrMsgIdAndTxt("ddma:monitor_mex", "list: too few arguments");
        ddma_apply_opts(nrhs > 12 ? prhs[12] : nullptr);
        SpecHolder h;
        make_spec(h, prhs[1], prhs[2], prhs[3], prhs[4]);
        const auto& s = h.s;
        const idx_t K = idx_t(scalar(prhs[7], "K"));
        auto v = make_view(prhs[5], prhs[6], K, s.n, s.m);
        need_double(prhs[8], "z");
        need_double(prhs[9], "y");
        if (int(mxGetNumberOfElements(prhs[8])) != s.n || int(mxGetNumberOfElements(prhs[9])) != s.m)
          mexErrMsgIdAndTxt("ddma:monitor_mex", "z must have n and y m entries");
        need_double(prhs[10], "idx");
        const mwSize L = mxGetNumberOfElements(prhs[10]);
        std::vector<idx_t> idx(L);
        for (mwSize t = 0; t < L; ++t) idx[t] = idx_t(mxGetPr(prhs[10])[t]);
        const auto* st = status_of(prhs[11], K);
        ddma::mon::Need N(s.m, s.n);
        std::vector<ddma::mon::Worst> W(s.m);
        ddma::mon::check_list(s, v, mxGetPr(prhs[8]), mxGetPr(prhs[9]), idx.data(), idx_t(L), st, N, W.data());
        put_need(plhs, N);
        plhs[6] = mxCreateDoubleMatrix(1, s.m, mxREAL);
        plhs[7] = mxCreateDoubleMatrix(1, s.m, mxREAL);
        for (int i = 0; i < s.m; ++i) {
          mxGetPr(plhs[6])[i] = W[i].gap;
          mxGetPr(plhs[7])[i] = double(W[i].k);
        }
        return;
      }
      if (mode == "full") {
        if (nrhs < 13) mexErrMsgIdAndTxt("ddma:monitor_mex", "full: too few arguments");
        ddma_apply_opts(nrhs > 13 ? prhs[13] : nullptr);
        SpecHolder h;
        make_spec(h, prhs[1], prhs[2], prhs[3], prhs[4]);
        const auto& s = h.s;
        const idx_t K = idx_t(scalar(prhs[7], "K"));
        auto v = make_view(prhs[5], prhs[6], K, s.n, s.m);
        const auto* st = status_of(prhs[8], K);
        const idx_t first = idx_t(scalar(prhs[9], "first")), last = idx_t(scalar(prhs[10], "last"));
        if (last != K) mexErrMsgIdAndTxt("ddma:monitor_mex", "full: last must equal K");
        ddma::mon::Grid G;
        const ddma::mon::Grid* gp = nullptr;
        if (!mxIsEmpty(prhs[11])) {
          G = grid_of(prhs[11], s.n, s.m, K);
          gp = &G;
        }
        const bool prune = mxIsLogicalScalarTrue(prhs[12]) ||
                           (mxIsDouble(prhs[12]) && mxGetScalar(prhs[12]) != 0);
        auto o = ddma::mon::check_full(s, v, st, first, last, gp, prune);
        put_need(plhs, o.need);
        const idx_t P = last - first;
        plhs[6] = mxCreateDoubleMatrix(P, s.m, mxREAL);
        plhs[7] = mxCreateDoubleMatrix(P, s.m, mxREAL);
        for (idx_t p = 0; p < P; ++p)
          for (int i = 0; i < s.m; ++i) {
            mxGetPr(plhs[6])[p + i * P] = o.worst[p * s.m + i].gap;
            mxGetPr(plhs[7])[p + i * P] = double(o.worst[p * s.m + i].k);
          }
        const idx_t R = idx_t(o.inrange.size());
        plhs[8] = mxCreateDoubleMatrix(R, 3, mxREAL);
        plhs[9] = mxCreateDoubleMatrix(R, 1, mxREAL);
        for (idx_t r = 0; r < R; ++r) {
          mxGetPr(plhs[8])[r] = double(o.inrange[r].q);
          mxGetPr(plhs[8])[r + R] = double(o.inrange[r].k);
          mxGetPr(plhs[8])[r + 2 * R] = double(o.inrange[r].i);
          mxGetPr(plhs[9])[r] = o.inrange[r].gap;
        }
        plhs[10] = mxCreateDoubleScalar(double(o.pairs));
        return;
      }
      if (mode == "all") {
        if (nrhs < 8) mexErrMsgIdAndTxt("ddma:monitor_mex", "all: too few arguments");
        ddma_apply_opts(nrhs > 8 ? prhs[8] : nullptr);
        SpecHolder h;
        make_spec(h, prhs[1], prhs[2], prhs[3], prhs[4]);
        const auto& s = h.s;
        const idx_t K = idx_t(mxGetM(prhs[5]));
        auto v = make_view(prhs[5], prhs[6], K, s.n, s.m);
        const bool prune = mxIsLogicalScalarTrue(prhs[7]) ||
                           (mxIsDouble(prhs[7]) && mxGetScalar(prhs[7]) != 0);
        auto o = ddma::mon::check_all(s, v, prune);
        put_need(plhs, o.need);
        plhs[6] = mxCreateDoubleMatrix(1, s.m, mxREAL);
        plhs[7] = mxCreateDoubleMatrix(1, s.m, mxREAL);
        plhs[8] = mxCreateDoubleMatrix(s.m, 2, mxREAL);
        for (int i = 0; i < s.m; ++i) {
          mxGetPr(plhs[6])[i] = double(o.count[i]);
          mxGetPr(plhs[7])[i] = o.worst[i].gap;
          mxGetPr(plhs[8])[i] = double(o.worst[i].a);
          mxGetPr(plhs[8])[i + s.m] = double(o.worst[i].b);
        }
        plhs[9] = mxCreateDoubleMatrix(K, 1, mxREAL);
        for (idx_t k = 0; k < K; ++k) mxGetPr(plhs[9])[k] = double(o.conflicts[k]);
        plhs[10] = mxCreateDoubleScalar(double(o.pairs));
        return;
      }
      if (mode == "grid") {
        if (nrhs < 4) mexErrMsgIdAndTxt("ddma:monitor_mex", "grid: too few arguments");
        need_double(prhs[1], "Z");
        const int n = int(mxGetN(prhs[1])), m = int(mxGetN(prhs[2]));
        const idx_t K = idx_t(scalar(prhs[3], "K"));
        auto v = make_view(prhs[1], prhs[2], K, n, m);
        auto G = ddma::mon::build_grid(v, n, m);
        const char* names[] = {"cdim", "glo", "ginv", "zlo", "zhi", "ylo", "yhi", "cnt", "head", "next", "cell"};
        plhs[0] = mxCreateStructMatrix(1, 1, 11, names);
        std::vector<double> cd(G.cdim.begin(), G.cdim.end());
        mxSetField(plhs[0], 0, "cdim", double_mat(cd, n, 1));
        mxSetField(plhs[0], 0, "glo", double_mat(G.glo, n, 1));
        mxSetField(plhs[0], 0, "ginv", double_mat(G.ginv, n, 1));
        mxSetField(plhs[0], 0, "zlo", double_mat(G.zlo, n, G.C));  // n x C = row-major C x n
        mxSetField(plhs[0], 0, "zhi", double_mat(G.zhi, n, G.C));
        mxSetField(plhs[0], 0, "ylo", double_mat(G.ylo, m, G.C));
        mxSetField(plhs[0], 0, "yhi", double_mat(G.yhi, m, G.C));
        mxSetField(plhs[0], 0, "cnt", int64_col(G.cnt));
        mxSetField(plhs[0], 0, "head", int64_col(G.head));
        mxSetField(plhs[0], 0, "next", int64_col(G.next));
        mxSetField(plhs[0], 0, "cell", int64_col(G.cell));
        return;
      }
    } catch (const std::exception& e) {
      mexErrMsgIdAndTxt("ddma:monitor_mex", "%s", e.what());
    }
  }
  mexErrMsgIdAndTxt("ddma:monitor_mex", "usage: ddma_monitor_mex('list' | 'full' | 'all' | 'grid' | 'info', ...)");
}
