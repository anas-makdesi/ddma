function [N, worst, wit] = monitor_k_list(lo, hi, th, tol, Z, Y, K, z, y, idx, st, use_mex)
%MONITOR_K_LIST Level 2: point (z, y) against the stored points idx (0-based, repeats
%   allowed; quarantined ones skipped). worst, wit (1 x m): largest gap > tol per output
%   and its 0-based stored index (ties: smaller index), -inf / -1 if none.
m = size(lo, 1);
if use_mex
    [a, w, nh, nho, nl, nlo, worst, wit] = ddma_monitor_mex('list', lo, hi, th, tol, Z, Y, K, ...
        z, y, idx, st, ddma.internal.mex_opts());
    N = struct('any', a, 'w', w, 'hi', nh, 'hio', nho, 'lo', nl, 'loo', nlo);
    return
end
N = ddma.internal.monitor_newneed(m, size(lo, 2));
idx = idx(:);
idx = idx(st(idx + 1) ~= 2);
worst = -inf(1, m); wit = -ones(1, m);
if isempty(idx), return; end
[g, N] = ddma.internal.monitor_eval(N, lo, hi, th, tol, z, y, Z(idx + 1, :), Y(idx + 1, :));
for i = 1:m
    v = g(:, i) > tol;
    if any(v)
        gi = g(:, i); gi(~v) = -inf;
        worst(i) = max(gi);
        wit(i) = min(idx(gi == worst(i)));
    end
end
end
