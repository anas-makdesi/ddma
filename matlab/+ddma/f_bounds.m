function [lo, hi] = f_bounds(S, zq)
%F_BOUNDS Bounds on the unknown function f itself at points zq (M x n):
%   [lower - d_lo, upper - d_hi] of the cells of the points. S from ddma.fit,
%   ddma.bd_fit or ddma.growth_map. Same as the Python f_bounds methods.
[lo, hi] = ddma.bounds(S, zq);
lo = lo - S.d_lo; hi = hi - S.d_hi;
end
