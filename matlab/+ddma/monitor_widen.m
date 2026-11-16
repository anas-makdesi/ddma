function [M, changed] = monitor_widen(M, target, at)
%MONITOR_WIDEN Apply the smallest change of one assumption (times 1 + margin) that makes
%   all stored data consistent: target 'disturbance' (default; widened symmetrically) or
%   {'slope_lo', i, j} / {'slope_hi', i, j}. The model must then be refit from M.Z, M.Y
%   with the new assumptions (M.needs_refit); its guarantees hold for assumptions fitted
%   to the data seen so far. Same as AssumptionMonitor.widen.
if nargin < 2 || isempty(target), target = 'disturbance'; end
if nargin < 3, when = ''; else, when = sprintf(' at transition %d', at); end
[kind, i, j] = ddma.internal.monitor_target(target);
changed = false;
if strcmp(kind, 'disturbance')
    for i = 1:M.m
        th = M.d_hi(i) - M.d_lo(i);
        extra = M.need_width(i) - th;
        if extra > 0
            extra = extra * (1 + M.margin);
            M.d_lo(i) = M.d_lo(i) - extra / 2; M.d_hi(i) = M.d_hi(i) + extra / 2;
            M.changes{end + 1} = sprintf('disturbance width of %s: %.4g -> %.4g%s', ...
                ddma.internal.monitor_name(M, 'out', i), th, M.d_hi(i) - M.d_lo(i), when);
            changed = true;
        end
    end
else
    if strcmp(kind, 'slope_lo'), need = M.need_lo(i, j); old = M.slope_lo(i, j); side = 'lower';
    else, need = M.need_hi(i, j); old = M.slope_hi(i, j); side = 'upper'; end
    if isnan(need)
        error('ddma:monitor', '%s(%d, %d) alone cannot explain the violations', kind, i, j);
    end
    if need ~= old
        new = old + (1 + M.margin) * (need - old);
        if strcmp(kind, 'slope_lo'), M.slope_lo(i, j) = new; else, M.slope_hi(i, j) = new; end
        M.changes{end + 1} = sprintf('%s slope bound of %s w.r.t. %s: %.4g -> %.4g%s', side, ...
            ddma.internal.monitor_name(M, 'out', i), ddma.internal.monitor_name(M, 'in', j), ...
            old, new, when);
        changed = true;
    end
end
if changed
    M.needs_refit = true;
    M.model = [];   % built on the old assumptions
    M.need_width = M.d_hi - M.d_lo; M.need_lo = M.slope_lo; M.need_hi = M.slope_hi;
end
end
