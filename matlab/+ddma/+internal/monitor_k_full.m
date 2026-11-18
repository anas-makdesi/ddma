function [N, worst, wit, inr, gaps] = monitor_k_full(lo, hi, th, tol, Z, Y, K, st, first, grid, prune, use_mex)
%MONITOR_K_FULL Level 3: each point q in [first, K) (0-based) against every earlier
%   point. worst, wit (P x m): largest gap > tol with a clean point settled before first
%   (0-based witness, -1 if none); inr (R x 3) [q k i] and gaps (R x 1): conflicts with
%   points of the same batch, sorted by (q, k, i) (0-based). Same as monitor_full in C++.
[m, n] = size(lo);
if use_mex
    [a, w, nh, nho, nl, nlo, worst, wit, inr, gaps] = ddma_monitor_mex('full', lo, hi, th, tol, ...
        Z, Y, K, st, first, K, grid, prune, ddma.internal.mex_opts());
    N = struct('any', a, 'w', w, 'hi', nh, 'hio', nho, 'lo', nl, 'loo', nlo);
    return
end
N = ddma.internal.monitor_newneed(m, n);
P = K - first;
worst = -inf(P, m); wit = -ones(P, m);
inr = zeros(0, 3); gaps = zeros(0, 1);
for q = first:K - 1
    if q == 0, continue; end
    [g, N] = ddma.internal.monitor_eval(N, lo, hi, th, tol, Z(q + 1, :), Y(q + 1, :), ...
                                         Z(1:q, :), Y(1:q, :));
    k = (0:q - 1)';
    clean = (k < first) & (st(1:q) == 0);
    for i = 1:m
        vc = (g(:, i) > tol) & clean;
        if any(vc)
            gi = g(:, i); gi(~vc) = -inf;
            worst(q - first + 1, i) = max(gi);
            wit(q - first + 1, i) = min(k(gi == max(gi)));
        end
    end
    late = (k >= first) & (g > tol);   % q x m
    [ii, kk] = find(late');            % on the transpose: ordered by k, then i
    if ~isempty(kk)
        inr = [inr; repmat(q, numel(kk), 1), kk(:) - 1, ii(:) - 1]; %#ok<AGROW>
        gaps = [gaps; g(sub2ind(size(g), kk(:), ii(:)))]; %#ok<AGROW>
    end
end
end
