function [M, out] = monitor_full_check(M)
%MONITOR_FULL_CHECK Level 3 for the pending points, each against all earlier points;
%   the blame is assigned in arrival order: a point is quarantined if it conflicts with an
%   earlier point that is clean at that moment. Returns the new violations (not recorded).
out = M.violations([]);
first = M.pending_from; K = M.K;
if first >= K, return; end
th = M.d_hi - M.d_lo;
[N, worst, wit, inr, gaps] = ddma.internal.monitor_k_full(M.slope_lo, M.slope_hi, th, M.tol, ...
    M.Z, M.Y, K, M.st, first, M.grid, M.prune, M.use_mex);
[M.need_width, M.need_lo, M.need_hi] = ddma.internal.monitor_merge(N, M.need_width, ...
    M.need_lo, M.need_hi, th);
found_at = M.n_checked;
r = 1; R = size(inr, 1);
for q = first:K - 1                         % 0-based
    rg = worst(q - first + 1, :); rk = wit(q - first + 1, :);
    while r <= R && inr(r, 1) == q          % partners from this same batch
        k = inr(r, 2); i = inr(r, 3) + 1; gp = gaps(r);
        if M.st(k + 1) == 0 && (gp > rg(i) || (gp == rg(i) && k < rk(i)))
            rg(i) = gp; rk(i) = k;
        end
        r = r + 1;
    end
    if M.st(q + 1) == 2, continue; end      % already quarantined: diagnosis only
    hit = find(rg > M.tol);
    if ~isempty(hit)
        M.st(q + 1) = 2;
        for i = hit
            out(end + 1) = struct('index', M.ids(q + 1), 'output', i, 'level', 'full', ...
                'gap', rg(i), 'witness', M.ids(rk(i) + 1), 'found_at', found_at); %#ok<AGROW>
        end
    end
end
M.pending_from = K;
end
