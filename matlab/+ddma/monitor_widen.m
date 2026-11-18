function [M, changed] = monitor_widen(M, target, at)
%MONITOR_WIDEN Apply the smallest change of one assumption (times 1 + margin) that makes
%   all stored data consistent: target 'disturbance' (default; widened symmetrically) or
%   {'slope_lo', i, j} / {'slope_hi', i, j} (never beyond M.slope_limit). Pending full
%   checks are run first; quarantined points then rejoin the data. The model must be
%   refit from ddma.monitor_data(M) with the new assumptions (M.needs_refit).
%   Same as AssumptionMonitor.widen.
if nargin < 2 || isempty(target), target = 'disturbance'; end
if nargin < 3, when = ''; else, when = sprintf(' at transition %d', at); end
[kind, i, j] = ddma.internal.monitor_target(M, target);
[M, v] = ddma.internal.monitor_full_check(M);
if ~isempty(v), M = ddma.internal.monitor_record(M, v); end
changed = false;
if strcmp(kind, 'disturbance')
    need = max(M.need_width, M.need_width_model);
    for i = 1:M.m
        th = M.d_hi(i) - M.d_lo(i);
        extra = need(i) - th;
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
    name = sprintf('%s slope bound of %s w.r.t. %s', side, ddma.internal.monitor_name(M, 'out', i), ...
                   ddma.internal.monitor_name(M, 'in', j));
    if isnan(need)
        error('ddma:monitor', '%s alone cannot explain the data', name);
    end
    if need ~= old
        new = old + (1 + M.margin) * (need - old);
        lim = M.slope_limit;
        if strcmp(kind, 'slope_hi'), beyond = need > lim; else, beyond = need < lim; end
        if beyond
            M.notes{end + 1} = sprintf('%s not widened: it would need %.4g, beyond the limit %.4g%s', ...
                                       name, need, lim, when);
        else
            if strcmp(kind, 'slope_hi'), new = min(new, lim); M.slope_hi(i, j) = new;
            else, new = max(new, lim); M.slope_lo(i, j) = new; end
            M.changes{end + 1} = sprintf('%s: %.4g -> %.4g%s', name, old, new, when);
            changed = true;
        end
    end
end
if changed
    M.needs_refit = true;
    M.model = [];          % built on the old assumptions
    M.st(1:M.K) = 0;       % all data consistent now: the quarantine is lifted
    M.need_width = M.d_hi - M.d_lo; M.need_lo = M.slope_lo; M.need_hi = M.slope_hi;
    M.need_width_model = M.d_hi - M.d_lo;
end
end
