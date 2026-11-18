function M = monitor_respond(M, viols)
%MONITOR_RESPOND Apply M.on_violation after violations were found.
when = viols(1).found_at;
switch M.on_violation
    case 'alarm'
        return
    case 'fallback'
        M = fall(M, when, 'assumption violated');
        return
end
if M.fallback_active, return; end    % widen_then_fallback after falling back
[M, v] = ddma.internal.monitor_full_check(M);   % needed changes exact only after it
if ~isempty(v), M = ddma.internal.monitor_record(M, v); end
target = M.widen;
[kind, i, j] = ddma.internal.monitor_target(M, target);
if ~strcmp(kind, 'disturbance') && isnan(target_need(M, kind, i, j))
    if strcmp(M.on_violation, 'widen_then_fallback')
        M = fall(M, when, sprintf('%s alone cannot explain the data', target_name(M, kind, i, j)));
        return
    end
    target = 'disturbance'; kind = 'disturbance';
end
if strcmp(M.on_violation, 'widen_then_fallback')
    reason = beyond_limit(M, kind, i, j);
    if ~isempty(reason)
        M = fall(M, when, reason);
        return
    end
end
M = ddma.monitor_widen(M, target, when);
end

function M = fall(M, when, reason)
if ~M.fallback_active
    M.fallback_active = true; M.fallback_since = when; M.fallback_reason = reason;
end
end

function v = target_need(M, kind, i, j)
if strcmp(kind, 'slope_lo'), v = M.need_lo(i, j); else, v = M.need_hi(i, j); end
end

function s = target_name(M, kind, i, j)
if strcmp(kind, 'slope_lo'), side = 'lower'; else, side = 'upper'; end
s = sprintf('%s slope bound of %s w.r.t. %s', side, ddma.internal.monitor_name(M, 'out', i), ...
            ddma.internal.monitor_name(M, 'in', j));
end

function reason = beyond_limit(M, kind, i, j)
reason = '';
th = M.d_hi - M.d_lo;
if strcmp(kind, 'disturbance')
    need = max(M.need_width, M.need_width_model);
    lim = M.widen_limit .* M.theta0;
    for i = 1:M.m
        if need(i) > th(i) + M.tol && need(i) > lim(i) + M.tol
            reason = sprintf(['disturbance width of %s would need %.4g, beyond the limit ' ...
                '%.4g (%g x the original %.4g)'], ddma.internal.monitor_name(M, 'out', i), ...
                need(i), lim(i), M.widen_limit(i), M.theta0(i));
            return
        end
    end
    return
end
need = target_need(M, kind, i, j);
if (strcmp(kind, 'slope_hi') && need > M.slope_limit) || (strcmp(kind, 'slope_lo') && need < M.slope_limit)
    reason = sprintf('%s would need %.4g, beyond the limit %.4g', target_name(M, kind, i, j), ...
                     need, M.slope_limit);
end
end
