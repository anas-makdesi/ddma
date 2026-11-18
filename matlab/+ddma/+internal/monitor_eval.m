function [g, N] = monitor_eval(N, lo, hi, th, tol, zq, yq, Z, Y)
%MONITOR_EVAL Point (zq, yq) against the rows of Z, Y, all outputs. Updates N with the
%   violating pairs and returns g (L x m) = max(gap_new, gap_old) (only entries > tol
%   matter). Same arithmetic as pair_gaps in cpp/ddma_monitor.hpp:
%   d = zq - z_k; tn = sum_j s(d_j); to = sum_j s(-d_j) (left to right, from 0);
%   dy = yq - y_k; gap_new = (dy - theta) - tn; gap_old = (-dy - theta) - to.
[L, n] = size(Z); m = size(Y, 2);
g = zeros(L, m);
d = zq - Z;
for i = 1:m
    tn = zeros(L, 1); to = zeros(L, 1);
    sn = ddma.internal.monitor_cone(d, lo(i, :), hi(i, :));
    so = ddma.internal.monitor_cone(-d, lo(i, :), hi(i, :));
    for j = 1:n
        tn = tn + sn(:, j); to = to + so(:, j);
    end
    dy = yq(i) - Y(:, i);
    gn = (dy - th(i)) - tn;
    go = (-dy - th(i)) - to;
    vn = gn > tol; vo = go > tol;
    if any(vn), N = ddma.internal.monitor_accumulate(N, lo, hi, th, i, d(vn, :), dy(vn), gn(vn)); end
    if any(vo), N = ddma.internal.monitor_accumulate(N, lo, hi, th, i, -d(vo, :), -dy(vo), go(vo)); end
    g(:, i) = max(gn, go);
end
end
