function N = monitor_accumulate(N, lo, hi, th, i, du, dyu, gap)
%MONITOR_ACCUMULATE Violating pairs "a above b" of output i: du = z_a - z_b (P x n),
%   dyu = y_a(i) - y_b(i), gap (P x 1). Updates the largest gap (w), and for each slope
%   bound the value it would need (hi / lo) or that it is ruled out (hio / loo).
%   Same operations, same order as cpp/ddma_monitor.hpp (accumulate).
n = size(du, 2);
N.any(i) = true;
N.w(i) = max(N.w(i), max(gap));
s = ddma.internal.monitor_cone(du, lo(i, :), hi(i, :));
tot = zeros(size(du, 1), 1);
for j = 1:n, tot = tot + s(:, j); end
for j = 1:n
    rest = (dyu - th(i)) - (tot - s(:, j));
    ratio = rest ./ du(:, j);
    pos = du(:, j) > 0; neg = du(:, j) < 0;
    if any(pos), N.hi(i, j) = max(N.hi(i, j), max(ratio(pos))); end
    if ~all(pos), N.hio(i, j) = true; end
    if any(neg), N.lo(i, j) = min(N.lo(i, j), min(ratio(neg))); end
    if ~all(neg), N.loo(i, j) = true; end
end
end
