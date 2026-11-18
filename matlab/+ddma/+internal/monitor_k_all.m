function [N, count, worst, pair, conflicts, pairs] = monitor_k_all(lo, hi, th, tol, Z, Y, prune, use_mex)
%MONITOR_K_ALL check_data: every pair once. count, worst (1 x m); pair (m x 2) 0-based
%   [later, earlier] (-1 if none); conflicts (K x 1). Same as monitor_all in C++.
[m, n] = size(lo); K = size(Z, 1);
if use_mex
    [a, w, nh, nho, nl, nlo, count, worst, pair, conflicts, pairs] = ddma_monitor_mex('all', ...
        lo, hi, th, tol, Z, Y, prune, ddma.internal.mex_opts());
    N = struct('any', a, 'w', w, 'hi', nh, 'hio', nho, 'lo', nl, 'loo', nlo);
    return
end
N = ddma.internal.monitor_newneed(m, n);
count = zeros(1, m); worst = -inf(1, m); pair = -ones(m, 2); conflicts = zeros(K, 1);
for q = 1:K - 1
    [g, N] = ddma.internal.monitor_eval(N, lo, hi, th, tol, Z(q + 1, :), Y(q + 1, :), ...
                                         Z(1:q, :), Y(1:q, :));
    v = g > tol;
    count = count + sum(v, 1);
    for i = 1:m
        if any(v(:, i))
            gi = g(:, i); gi(~v(:, i)) = -inf;
            b = max(gi);
            if b > worst(i)
                worst(i) = b; pair(i, :) = [q, find(gi == b, 1) - 1];
            end
        end
    end
    a = any(v, 2);
    conflicts(q + 1) = conflicts(q + 1) + sum(a);
    conflicts(1:q) = conflicts(1:q) + a;
end
pairs = K * (K - 1) / 2;
end
