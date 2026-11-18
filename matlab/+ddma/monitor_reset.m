function M = monitor_reset(M)
%MONITOR_RESET Clear the violation record, the fallback flag and the needed changes
%   (assumptions and quarantine stay as they are). Same as AssumptionMonitor.reset.
M.violations = M.violations([]);
M.violating = zeros(0, 1);
M.fallback_active = false; M.fallback_since = []; M.fallback_reason = '';
M.need_width = M.d_hi - M.d_lo; M.need_lo = M.slope_lo; M.need_hi = M.slope_hi;
M.need_width_model = M.d_hi - M.d_lo;
end
