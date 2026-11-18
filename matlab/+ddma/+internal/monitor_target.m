function [kind, i, j] = monitor_target(M, t)
%MONITOR_TARGET Parse and check a widening target: 'disturbance' or {'slope_lo'|'slope_hi',
%   i, j}. A slope bound can be widened only if both of its ends are finite and with
%   M.slope_limit.
if ischar(t) && strcmp(t, 'disturbance')
    kind = t; i = []; j = []; return;
end
if ~(iscell(t) && numel(t) == 3 && ischar(t{1}) && any(strcmp(t{1}, {'slope_lo', 'slope_hi'})))
    error('ddma:monitor', 'widen target must be ''disturbance'', {''slope_lo'', i, j} or {''slope_hi'', i, j}');
end
kind = t{1}; i = t{2}; j = t{3};
if ~(isfinite(M.slope_lo(i, j)) && isfinite(M.slope_hi(i, j)))
    error('ddma:monitor', 'only a finite slope interval can be widened (a sign has no scale: use a bounded-derivative model)');
end
if isempty(M.slope_limit) || ~isfinite(M.slope_limit)
    error('ddma:monitor', 'widening a slope bound needs an explicit finite slope_limit');
end
end
