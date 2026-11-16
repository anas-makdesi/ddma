function [kind, i, j] = monitor_target(t)
%MONITOR_TARGET Parse a widening target: 'disturbance' or {'slope_lo'|'slope_hi', i, j}.
if ischar(t) && strcmp(t, 'disturbance')
    kind = t; i = []; j = []; return;
end
if iscell(t) && numel(t) == 3 && ischar(t{1}) && any(strcmp(t{1}, {'slope_lo', 'slope_hi'}))
    kind = t{1}; i = t{2}; j = t{3}; return;
end
error('ddma:monitor', 'widen target must be ''disturbance'', {''slope_lo'', i, j} or {''slope_hi'', i, j}');
end
