function ok = msm_is_consistent(Z, Y, d_lo, d_hi, signs, tol)
%MSM_IS_CONSISTENT Whether some monotone f + [d_lo, d_hi] can explain the data:
%   y_k <= y_l + (d_hi - d_lo) whenever z_k <= z_l (in the order of the sign pattern).
%   A violation shows that the monotonicity assumption, the sign pattern or the
%   disturbance bounds are wrong. (Merging two data sets = concatenating Z and Y.)
%   Same as the Python MinimalSimulatingMap.is_consistent.
if nargin < 5, signs = []; end
if nargin < 6, tol = 1e-12; end
[lo, hi] = ddma.msm_bounds(Z, Y, d_lo, d_hi, Z, signs);
ok = all(all(lo <= Y + tol & Y <= hi + tol));
end
