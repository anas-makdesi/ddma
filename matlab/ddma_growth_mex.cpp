// MEX gateway to the growth-cone kernels of the shared C++ core (cpp/ddma_growth.hpp).
//   [lo, up] = ddma_growth_mex(mode, Z, y, edges, g_lo, g_hi, theta)
//   mode 0: bounds at the grid vertices (prod(numel(edges{j})) values);
//   mode 1: per-cell bounds from the cones of the points in the cell ("cell");
//   mode 2: same with the points in the cell's slabs ("slab") (prod(numel(edges{j})-1)).
//   Z: K x n, y: K x 1 (double); edges: 1 x n cell; g_lo, g_hi: 1 x n; theta: scalar.
//   Outputs are column vectors in MATLAB column-major order.
//   Optional 8th argument: opts = [threads, scatter_mode, chunk] (ddma_mex_opts.hpp).
// Build: see build_mex.m. The core works in C order; reversing the axis order makes its
// output column-major.
#include "mex.h"
#include "ddma_growth.hpp"
#include "ddma_mex_opts.hpp"
#include <vector>

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  if (nrhs != 7 && nrhs != 8)
    mexErrMsgIdAndTxt("ddma:mex", "usage: [lo, up] = ddma_growth_mex(mode, Z, y, edges, g_lo, g_hi, theta, opts)");
  ddma_apply_opts(nrhs == 8 ? prhs[7] : nullptr);
  for (int a : {0, 1, 2, 4, 5, 6})
    if (!mxIsDouble(prhs[a])) mexErrMsgIdAndTxt("ddma:mex", "numeric arguments must be double");
  if (!mxIsCell(prhs[3])) mexErrMsgIdAndTxt("ddma:mex", "edges must be a cell array");
  const int mode = int(mxGetScalar(prhs[0]));
  const mwSize K = mxGetM(prhs[1]), n = mxGetN(prhs[1]);
  if (mxGetNumberOfElements(prhs[2]) != K) mexErrMsgIdAndTxt("ddma:mex", "y needs one value per row of Z");
  if (mxGetNumberOfElements(prhs[3]) != n || mxGetNumberOfElements(prhs[4]) != n ||
      mxGetNumberOfElements(prhs[5]) != n)
    mexErrMsgIdAndTxt("ddma:mex", "one edge vector and one derivative bound per input");
  const double* Z = mxGetPr(prhs[1]);
  const double* y = mxGetPr(prhs[2]);
  const double* gl = mxGetPr(prhs[4]);
  const double* gh = mxGetPr(prhs[5]);
  const double theta = mxGetScalar(prhs[6]);

  std::vector<double> Zc(K * n), glc(n), ghc(n);
  for (mwSize k = 0; k < K; ++k)
    for (mwSize j = 0; j < n; ++j) Zc[k * n + j] = Z[k + K * (n - 1 - j)];
  std::vector<const double*> e(n);
  std::vector<std::ptrdiff_t> ne(n);
  std::ptrdiff_t NV = 1, NQ = 1;
  for (mwSize j = 0; j < n; ++j) {
    const mxArray* a = mxGetCell(prhs[3], n - 1 - j);
    if (!a || !mxIsDouble(a)) mexErrMsgIdAndTxt("ddma:mex", "edges must be double vectors");
    e[j] = mxGetPr(a);
    ne[j] = std::ptrdiff_t(mxGetNumberOfElements(a));
    NV *= ne[j];
    NQ *= ne[j] - 1;
    glc[j] = gl[n - 1 - j];
    ghc[j] = gh[n - 1 - j];
  }
  const std::ptrdiff_t N = mode == 0 ? NV : NQ;
  plhs[0] = mxCreateDoubleMatrix(N, 1, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(N, 1, mxREAL);
  try {
    if (mode == 0)
      ddma::growth_vertices(Zc.data(), y, K, n, e, ne, glc.data(), ghc.data(), theta,
                            mxGetPr(plhs[0]), mxGetPr(plhs[1]));
    else if (mode == 1 || mode == 2)
      ddma::growth_local(Zc.data(), y, K, n, e, ne, glc.data(), ghc.data(), theta, mode == 2,
                         mxGetPr(plhs[0]), mxGetPr(plhs[1]));
    else
      mexErrMsgIdAndTxt("ddma:mex", "mode must be 0 (vertices), 1 (cell) or 2 (slab)");
  } catch (const std::exception& ex) {
    mexErrMsgIdAndTxt("ddma:mex", ex.what());
  }
}
