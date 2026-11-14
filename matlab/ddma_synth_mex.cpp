// MEX gateway to the synthesis fixed points of the shared C++ core (cpp/ddma_synthesis.hpp).
//   [W, allowed]         = ddma_synth_mex('safety', first, last, enabled, shapeX, safe)
//   [R, allowed, value]  = ddma_synth_mex('reach_avoid', first, last, enabled, shapeX, target, avoid)
//   good                 = ddma_synth_mex('box_all_in', first, last, enabled, shapeX, W)
//   first, last: nQ x nP x nx int32 (0-based cell ranges, as built by ddma.abstraction);
//   enabled: nQ x nP logical; masks: nQ x 1 logical, cells in column-major order.
// The arrays are read in place: column-major order over shapeX is C order over the
// reversed shape, so coordinate j of first/last is axis nx - 1 - j for the core.
// Build:  mex -O ddma_synth_mex.cpp -I../cpp        (MATLAB)
//         mkoctfile --mex -O ddma_synth_mex.cpp -I../cpp   (Octave)
#include "mex.h"
#include "ddma_synthesis.hpp"
#include <cstring>
#include <string>
#include <vector>

static const std::uint8_t* logical_in(const mxArray* a, std::ptrdiff_t n, const char* name) {
  if (!mxIsLogical(a) || std::ptrdiff_t(mxGetNumberOfElements(a)) != n)
    mexErrMsgIdAndTxt("ddma:mex", "%s must be a logical array with %d elements", name, int(n));
  static_assert(sizeof(mxLogical) == 1, "mxLogical must be one byte");
  return reinterpret_cast<const std::uint8_t*>(mxGetLogicals(a));
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  if (nrhs < 6 || !mxIsChar(prhs[0]))
    mexErrMsgIdAndTxt("ddma:mex", "usage: ddma_synth_mex(op, first, last, enabled, shapeX, ...)");
  char buf[32];
  mxGetString(prhs[0], buf, sizeof(buf));
  const std::string op(buf);
  if (!mxIsInt32(prhs[1]) || !mxIsInt32(prhs[2]))
    mexErrMsgIdAndTxt("ddma:mex", "first and last must be int32");
  const mxArray* en = prhs[3];
  if (!mxIsLogical(en)) mexErrMsgIdAndTxt("ddma:mex", "enabled must be logical");
  const std::ptrdiff_t nQ = std::ptrdiff_t(mxGetM(en)), nP = std::ptrdiff_t(mxGetN(en));
  const std::ptrdiff_t nx = std::ptrdiff_t(mxGetNumberOfElements(prhs[4]));
  if (!mxIsDouble(prhs[4]) || nx < 1) mexErrMsgIdAndTxt("ddma:mex", "shapeX must be a double vector");
  const double* shX = mxGetPr(prhs[4]);
  if (std::ptrdiff_t(mxGetNumberOfElements(prhs[1])) != nQ * nP * nx ||
      std::ptrdiff_t(mxGetNumberOfElements(prhs[2])) != nQ * nP * nx)
    mexErrMsgIdAndTxt("ddma:mex", "first/last must be nQ x nP x nx");

  std::vector<std::ptrdiff_t> shape(nx);
  ddma::PairBoxes<std::int32_t> a{static_cast<const std::int32_t*>(mxGetData(prhs[1])),
                                  static_cast<const std::int32_t*>(mxGetData(prhs[2])),
                                  reinterpret_cast<const std::uint8_t*>(mxGetLogicals(en)),
                                  nQ, nP, nx, 1, nQ, nQ * nP, 1, nQ, {}};
  for (std::ptrdiff_t j = 0; j < nx; ++j) {
    shape[nx - 1 - j] = std::ptrdiff_t(shX[j]);
    a.dim.push_back(nx - 1 - j);
  }
  try {
    if (op == "box_all_in") {
      const std::uint8_t* W = logical_in(prhs[5], nQ, "W");
      plhs[0] = mxCreateLogicalMatrix(nQ, nP);
      ddma::box_all_in(a, shape, W, reinterpret_cast<std::uint8_t*>(mxGetLogicals(plhs[0])));
    } else if (op == "safety") {
      const std::uint8_t* safe = logical_in(prhs[5], nQ, "safe");
      plhs[0] = mxCreateLogicalMatrix(nQ, 1);
      plhs[1] = mxCreateLogicalMatrix(nQ, nP);
      std::uint8_t* W = reinterpret_cast<std::uint8_t*>(mxGetLogicals(plhs[0]));
      std::memcpy(W, safe, nQ);
      ddma::safety(a, shape, W, reinterpret_cast<std::uint8_t*>(mxGetLogicals(plhs[1])));
    } else if (op == "reach_avoid") {
      if (nrhs < 7) mexErrMsgIdAndTxt("ddma:mex", "reach_avoid needs target and avoid");
      const std::uint8_t* target = logical_in(prhs[5], nQ, "target");
      const std::uint8_t* avoid = logical_in(prhs[6], nQ, "avoid");
      plhs[0] = mxCreateLogicalMatrix(nQ, 1);
      plhs[1] = mxCreateLogicalMatrix(nQ, nP);
      plhs[2] = mxCreateDoubleMatrix(nQ, 1, mxREAL);
      ddma::reach_avoid(a, shape, target, avoid,
                        reinterpret_cast<std::uint8_t*>(mxGetLogicals(plhs[0])), mxGetPr(plhs[2]),
                        reinterpret_cast<std::uint8_t*>(mxGetLogicals(plhs[1])));
    } else {
      mexErrMsgIdAndTxt("ddma:mex", "unknown operation '%s'", buf);
    }
  } catch (const std::exception& ex) {
    mexErrMsgIdAndTxt("ddma:mex", "%s", ex.what());
  }
  (void)nlhs;
}
