// ddma_abs_mex.cpp -- transitions of the symbolic abstraction (ddma.abstraction), MEX.
//
// [en, F, L] = ddma_abs_mex(edges, gathers, q, p, paired, g_lo, g_hi, assumption, opts)
//   edges      1 x nx cell of state edges;
//   gathers    struct array, one per bounds array read: offq (nQ x 1) and offp (nP x 1)
//              0-based offsets of the state and input cells into the map's grid
//              (flat cell = offq(q) + offp(p)); lo, hi (cells x m, double or single);
//              outs (0-based outputs of the m columns);
//   q, p       0-based state and input cells: all pairs (q(i), p(c)) (paired = false,
//              outputs numel(q) x numel(p) [x nx]) or the pairs (q(i), p(i)) (paired =
//              true, outputs numel(q) x 1 [x nx]);
//   g_lo, g_hi [] or bounds of g per pair (pairs x nx, pairs in the order of the outputs);
//   assumption [] or 4 x nx [a_lo; a_hi; c_lo; c_hi] (c 0-based);
//   opts       [threads, ...] (ddma.internal.mex_opts).
// en: logical; F, L: int32 0-based first/last successor cells. Same arithmetic as the
// MATLAB code in ddma.internal.pair_boxes and as the Python kernel (cpp/ddma_abstraction.hpp).
#include <cstdint>
#include <vector>

#include "ddma_abstraction.hpp"
#include "ddma_mex_opts.hpp"
#include "mex.h"

namespace {

std::vector<std::int64_t> to_index(const mxArray* a, const char* what) {
  if (!mxIsDouble(a) || mxIsComplex(a)) mexErrMsgIdAndTxt("ddma:abs", "%s must be double", what);
  const mwSize k = mxGetNumberOfElements(a);
  const double* v = mxGetPr(a);
  std::vector<std::int64_t> r(k);
  for (mwSize i = 0; i < k; ++i) {
    if (!(v[i] >= 0) || v[i] != double(std::int64_t(v[i])))
      mexErrMsgIdAndTxt("ddma:abs", "%s must hold non-negative integers", what);
    r[i] = std::int64_t(v[i]);
  }
  return r;
}

const mxArray* field(const mxArray* s, mwIndex i, const char* name) {
  const mxArray* f = mxGetField(s, i, name);
  if (!f) mexErrMsgIdAndTxt("ddma:abs", "gathers needs the field %s", name);
  return f;
}

}  // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  if (ddma_info_call(nlhs, plhs, nrhs, prhs)) return;
  if (nrhs < 8) mexErrMsgIdAndTxt("ddma:abs", "usage: ddma_abs_mex(edges, gathers, q, p, paired, g_lo, g_hi, assumption[, opts])");
  ddma_apply_opts(nrhs > 8 ? prhs[8] : nullptr);

  ddma::AbsSpec s;
  s.mode = 0;
  const mxArray* E = prhs[0];
  if (!mxIsCell(E)) mexErrMsgIdAndTxt("ddma:abs", "edges must be a cell array");
  s.n = int(mxGetNumberOfElements(E));
  for (int j = 0; j < s.n; ++j) {
    const mxArray* e = mxGetCell(E, j);
    if (!e || !mxIsDouble(e) || mxGetNumberOfElements(e) < 2)
      mexErrMsgIdAndTxt("ddma:abs", "edges{%d} must be a double vector with >= 2 entries", j + 1);
    s.edges.push_back(mxGetPr(e));
    s.ne.push_back(std::ptrdiff_t(mxGetNumberOfElements(e)));
  }

  const auto q = to_index(prhs[2], "q");
  const auto p = to_index(prhs[3], "p");
  const bool paired = mxIsLogicalScalarTrue(prhs[4]) || (mxIsDouble(prhs[4]) && mxGetScalar(prhs[4]) != 0);
  if (paired && q.size() != p.size()) mexErrMsgIdAndTxt("ddma:abs", "paired: q and p must have the same length");
  const std::ptrdiff_t r = std::ptrdiff_t(q.size()), k = paired ? 1 : std::ptrdiff_t(p.size());
  const std::ptrdiff_t pairs = r * k;

  const mxArray* G = prhs[1];
  if (!mxIsStruct(G)) mexErrMsgIdAndTxt("ddma:abs", "gathers must be a struct array");
  const mwSize ng = mxGetNumberOfElements(G);
  std::vector<std::vector<std::int64_t>> offs;
  offs.reserve(2 * ng);
  for (mwIndex i = 0; i < ng; ++i) {
    ddma::AbsGather g;
    offs.push_back(to_index(field(G, i, "offq"), "offq"));
    offs.push_back(to_index(field(G, i, "offp"), "offp"));
    const auto& oq = offs[offs.size() - 2];
    const auto& op = offs.back();
    for (auto v : q)
      if (v >= std::int64_t(oq.size())) mexErrMsgIdAndTxt("ddma:abs", "state index out of range");
    for (auto v : p)
      if (v >= std::int64_t(op.size())) mexErrMsgIdAndTxt("ddma:abs", "input index out of range");
    g.offq = oq.data();
    g.offp = op.data();
    const mxArray* lo = field(G, i, "lo");
    const mxArray* hi = field(G, i, "hi");
    if (mxGetClassID(lo) != mxGetClassID(hi) || (!mxIsDouble(lo) && !mxIsSingle(lo)) ||
        mxGetM(lo) != mxGetM(hi) || mxGetN(lo) != mxGetN(hi))
      mexErrMsgIdAndTxt("ddma:abs", "lo and hi must be double or single arrays of one size");
    g.f32 = mxIsSingle(lo);
    g.lo = mxGetData(lo);
    g.hi = mxGetData(hi);
    const auto outs = to_index(field(G, i, "outs"), "outs");
    g.m = int(outs.size());
    if (std::ptrdiff_t(mxGetN(lo)) != g.m) mexErrMsgIdAndTxt("ddma:abs", "lo must have one column per output");
    for (auto o : outs) {
      if (o >= s.n) mexErrMsgIdAndTxt("ddma:abs", "output index out of range");
      g.outs.push_back(int(o));
    }
    g.fstride = 1;
    g.cstride = std::ptrdiff_t(mxGetM(lo));
    // the largest flat cell read must lie inside lo
    std::int64_t mq = 0, mp = 0;
    for (auto v : q) mq = std::max(mq, oq[v]);
    for (auto v : p) mp = std::max(mp, op[v]);
    if (r > 0 && k > 0 && mq + mp >= std::int64_t(mxGetM(lo)))
      mexErrMsgIdAndTxt("ddma:abs", "offsets point outside the bounds array");
    s.gathers.push_back(std::move(g));
  }
  if (!mxIsEmpty(prhs[5])) {
    if (!mxIsDouble(prhs[5]) || !mxIsDouble(prhs[6]) ||
        std::ptrdiff_t(mxGetM(prhs[5])) != pairs || std::ptrdiff_t(mxGetN(prhs[5])) != s.n ||
        mxGetM(prhs[6]) != mxGetM(prhs[5]) || mxGetN(prhs[6]) != mxGetN(prhs[5]))
      mexErrMsgIdAndTxt("ddma:abs", "g_lo, g_hi must be double, one row per pair, nx columns");
    s.g_lo = mxGetPr(prhs[5]);
    s.g_hi = mxGetPr(prhs[6]);
  }
  s.pair_stride = 1;
  s.dim_stride = pairs;
  if (!mxIsEmpty(prhs[7])) {
    const mxArray* a = prhs[7];
    if (!mxIsDouble(a) || mxGetM(a) != 4 || std::ptrdiff_t(mxGetN(a)) != s.n)
      mexErrMsgIdAndTxt("ddma:abs", "assumption must be 4 x nx [a_lo; a_hi; c_lo; c_hi]");
    const double* v = mxGetPr(a);
    s.assume = true;
    for (int j = 0; j < s.n; ++j) {
      s.a_lo.push_back(v[4 * j]);
      s.a_hi.push_back(v[4 * j + 1]);
      s.c_lo.push_back(std::ptrdiff_t(v[4 * j + 2]));
      s.c_hi.push_back(std::ptrdiff_t(v[4 * j + 3]));
    }
  }
  s.prepare();

  const mwSize dims[3] = {mwSize(r), mwSize(k), mwSize(s.n)};
  plhs[0] = mxCreateLogicalMatrix(r, k);
  plhs[1] = mxCreateNumericArray(3, dims, mxINT32_CLASS, mxREAL);
  plhs[2] = mxCreateNumericArray(3, dims, mxINT32_CLASS, mxREAL);
  auto* en = reinterpret_cast<std::uint8_t*>(mxGetLogicals(plhs[0]));
  auto* F = static_cast<std::int32_t*>(mxGetData(plhs[1]));
  auto* L = static_cast<std::int32_t*>(mxGetData(plhs[2]));
  try {
    ddma::abstraction_colmajor<std::int32_t>(s, q.data(), r, p.data(), k, paired, en, F, L);
  } catch (const std::exception& e) {
    mexErrMsgIdAndTxt("ddma:abs", "%s", e.what());
  }
}
