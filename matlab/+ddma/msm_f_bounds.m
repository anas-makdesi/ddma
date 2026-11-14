function [lo, hi] = msm_f_bounds(Z, Y, d_lo, d_hi, zq, signs)
%MSM_F_BOUNDS Bounds on the unknown monotone f at query points zq:
%   [S_lo(z) - d_lo, S_hi(z) - d_hi] of the minimal simulating map (TAC Thm. 1).
%   Same as the Python MinimalSimulatingMap.f_bounds.
if nargin < 6, signs = []; end
[lo, hi] = ddma.msm_bounds(Z, Y, d_lo, d_hi, zq, signs);
m = size(Y, 2);
lo = lo - d_lo(:)' .* ones(1, m); hi = hi - d_hi(:)' .* ones(1, m);
end
