function need = monitor_needed(M)
%MONITOR_NEEDED Smallest single changes that make all stored data consistent: struct with
%   width (1 x m, disturbance width d_hi - d_lo needed), slope_lo, slope_hi (m x n), each
%   with all other assumptions unchanged. Equal to the current value: no change needed;
%   nan: this bound alone cannot explain the violations. Same as AssumptionMonitor.needed.
need = struct('width', M.need_width, 'slope_lo', M.need_lo, 'slope_hi', M.need_hi);
end
