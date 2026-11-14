// MEX gateway to the shared C++ core (cpp/ddma_core.hpp), for MATLAB and Octave.
//   [lower, upper] = ddma_fit_mex(Z, Y, edges, theta, signs)
//   Z: K x n, Y: K x m (double); edges: 1 x n cell of increasing vectors;
//   theta: 1 x m (d_hi - d_lo); signs: m x n int8 in {-1, 0, 1}.
//   lower, upper: N x m, cells in MATLAB column-major order.
// Build:  mex -O ddma_fit_mex.cpp -I../cpp        (MATLAB)
//         mkoctfile --mex -O ddma_fit_mex.cpp -I../cpp   (Octave)
// The core works in C order; reversing the axis order makes its output column-major.
#include "mex.h"
#include "ddma_core.hpp"
#include <vector>

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  if (nrhs != 5) mexErrMsgIdAndTxt("ddma:mex", "usage: [lo, hi] = ddma_fit_mex(Z, Y, edges, theta, signs)");
  if (!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) || !mxIsCell(prhs[2]) || !mxIsDouble(prhs[3]) ||
      !mxIsInt8(prhs[4]))
    mexErrMsgIdAndTxt("ddma:mex", "types: double, double, cell, double, int8");
  const mwSize K = mxGetM(prhs[0]), n = mxGetN(prhs[0]), m = mxGetN(prhs[1]);
  if (mxGetM(prhs[1]) != K) mexErrMsgIdAndTxt("ddma:mex", "Z and Y must have the same rows");
  if (mxGetNumberOfElements(prhs[2]) != n) mexErrMsgIdAndTxt("ddma:mex", "one edge vector per input");
  const double* Z = mxGetPr(prhs[0]);
  const double* Y = mxGetPr(prhs[1]);
  const double* theta = mxGetPr(prhs[3]);
  const signed char* S = static_cast<const signed char*>(mxGetData(prhs[4]));

  // reversed axis order
  std::vector<double> Zc(K * n), Yc(K * m);
  for (mwSize k = 0; k < K; ++k) {
    for (mwSize j = 0; j < n; ++j) Zc[k * n + j] = Z[k + K * (n - 1 - j)];
    for (mwSize i = 0; i < m; ++i) Yc[k * m + i] = Y[k + K * i];
  }
  std::vector<const double*> e(n);
  std::vector<std::ptrdiff_t> ne(n);
  std::ptrdiff_t N = 1;
  for (mwSize j = 0; j < n; ++j) {
    const mxArray* a = mxGetCell(prhs[2], n - 1 - j);
    if (!a || !mxIsDouble(a)) mexErrMsgIdAndTxt("ddma:mex", "edges must be double vectors");
    e[j] = mxGetPr(a);
    ne[j] = std::ptrdiff_t(mxGetNumberOfElements(a));
    N *= ne[j] - 1;
  }
  std::vector<std::int8_t> sc(m * n);
  for (mwSize i = 0; i < m; ++i)
    for (mwSize j = 0; j < n; ++j) sc[i * n + j] = S[i + m * (n - 1 - j)];
  std::vector<double> lo(N * m), hi(N * m);
  try {
    ddma::fit_sigma<double>(Zc.data(), Yc.data(), K, n, m, e, ne, theta, sc.data(), lo.data(), hi.data());
  } catch (const std::exception& ex) {
    mexErrMsgIdAndTxt("ddma:mex", ex.what());
  }
  plhs[0] = mxCreateDoubleMatrix(N, m, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(N, m, mxREAL);
  double* L = mxGetPr(plhs[0]);
  double* H = mxGetPr(plhs[1]);
  for (std::ptrdiff_t q = 0; q < N; ++q)
    for (mwSize i = 0; i < m; ++i) {
      L[q + N * i] = lo[q * m + i];
      H[q + N * i] = hi[q * m + i];
    }
}
