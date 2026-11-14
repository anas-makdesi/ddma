function e = msm_estimate(Z, Y, d_lo, d_hi, zq, signs)
%MSM_ESTIMATE Midpoint of the f-bounds of the minimal simulating map (set-membership
%   estimate); NaN where unbounded. Same as the Python MinimalSimulatingMap.estimate.
if nargin < 6, signs = []; end
[lo, hi] = ddma.msm_f_bounds(Z, Y, d_lo, d_hi, zq, signs);
e = 0.5 * (lo + hi);
e(~isfinite(e)) = NaN;
end
