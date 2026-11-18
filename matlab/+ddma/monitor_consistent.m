function ok = monitor_consistent(M)
%MONITOR_CONSISTENT Whether all stored data (quarantined too) are consistent with the
%   current assumptions (all pairs, exact). Same as AssumptionMonitor.consistent.
ok = true;
if M.K < 2, return; end
[~, count] = ddma.internal.monitor_k_all(M.slope_lo, M.slope_hi, M.d_hi - M.d_lo, M.tol, ...
    M.Z(1:M.K, :), M.Y(1:M.K, :), M.prune, M.use_mex);
ok = ~any(count);
end
