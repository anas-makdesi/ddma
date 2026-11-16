function M = monitor_reset(M)
%MONITOR_RESET Clear the violation record and the fallback flag (assumptions unchanged).
M.violations = M.violations([]);
M.violating = zeros(0, 1);
M.fallback_active = false; M.fallback_since = [];
M.need_width = M.d_hi - M.d_lo; M.need_lo = M.slope_lo; M.need_hi = M.slope_hi;
end
