// ddma_mex_opts.hpp -- optional last argument of the learning MEX files (ddma_fit_mex,
// ddma_growth_mex): opts = [threads, scatter_mode, chunk] (doubles; missing entries = 0,
// meaning: OpenMP's default number of threads, automatic mode, default round size).
// MATLAB keeps the thread setting (ddma.set_num_threads) and passes it on every call,
// because each MEX file is a separate library with its own copy of the C++ setting.
#pragma once
#include "mex.h"
#include "ddma_parallel.hpp"

inline void ddma_apply_opts(const mxArray* a) {
  double v[3] = {0, 0, 0};
  if (a) {
    if (!mxIsDouble(a)) mexErrMsgIdAndTxt("ddma:mex", "opts must be a double vector");
    const mwSize k = mxGetNumberOfElements(a);
    const double* p = mxGetPr(a);
    for (mwSize i = 0; i < k && i < 3; ++i) v[i] = p[i];
  }
  ddma::set_num_threads(int(v[0]));
  ddma::scatter_mode() = int(v[1]);
  ddma::chunk_entries() = v[2] > 0 ? std::ptrdiff_t(v[2]) : (std::ptrdiff_t(1) << 20);
}

// [threads, openmp] = ddma_fit_mex('info', opts): threads the kernels would use.
inline bool ddma_info_call(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  if (nrhs < 1 || !mxIsChar(prhs[0])) return false;
  ddma_apply_opts(nrhs > 1 ? prhs[1] : nullptr);
  plhs[0] = mxCreateDoubleScalar(double(ddma::num_threads()));
  if (nlhs > 1) plhs[1] = mxCreateDoubleScalar(ddma::openmp_available() ? 1.0 : 0.0);
  return true;
}
