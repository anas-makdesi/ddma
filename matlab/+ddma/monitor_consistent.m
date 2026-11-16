function ok = monitor_consistent(M)
%MONITOR_CONSISTENT Whether all stored data are consistent with the current assumptions
%   (all pairs, O(K^2)). Same as AssumptionMonitor.consistent.
th = M.d_hi - M.d_lo;
ok = true;
for a = 1:size(M.Z, 1)
    dz = M.Z(a, :) - M.Z; dy = M.Y(a, :) - M.Y;
    for i = 1:M.m
        c = sum(ddma.internal.monitor_cone(dz, M.slope_lo(i, :), M.slope_hi(i, :)), 2);
        if any(dy(:, i) - th(i) - c > M.tol), ok = false; return; end
    end
end
end
