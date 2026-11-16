function M = monitor_monotone(signs, d_lo, d_hi, opts)
%MONITOR_MONOTONE Assumption monitor of a monotone model with sign pattern signs (m x n):
%   +1 -> slope in [0, inf], -1 -> [-inf, 0], 0 -> [0, 0]. See ddma.monitor.
if nargin < 4, opts = struct(); end
lo = zeros(size(signs)); hi = zeros(size(signs));
lo(signs < 0) = -inf; hi(signs > 0) = inf;
M = ddma.monitor(lo, hi, d_lo, d_hi, opts);
end
