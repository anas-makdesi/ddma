function L = monitor_explain(i, need, th, slope_lo, slope_hi, tol, out, inp)
%MONITOR_EXPLAIN Plain-words lines for output i: the single changes that explain all data
%   and the bounds ruled out. out, inp: functions giving names. Same text as Python.
L = {}; cands = {}; ruled = {};
if need.width(i) > th(i) + tol
    cands{end + 1} = sprintf('disturbance width %.4g -> %.4g', th(i), need.width(i));
end
sides = {'lower', 'upper'};
for j = 1:size(slope_lo, 2)
    for t = 1:2
        if t == 1, v = need.slope_lo(i, j); c = slope_lo(i, j);
        else, v = need.slope_hi(i, j); c = slope_hi(i, j); end
        name = sprintf('%s slope bound w.r.t. %s', sides{t}, inp(j));
        if isnan(v)
            ruled{end + 1} = name; %#ok<AGROW>
        elseif v ~= c
            note = '';
            if t == 1 && c == 0 && v < 0
                if slope_hi(i, j) == 0, note = ' (contradicts ''does not depend on it'')';
                else, note = ' (contradicts ''non-decreasing'')'; end
            end
            if t == 2 && c == 0 && v > 0
                if slope_lo(i, j) == 0, note = ' (contradicts ''does not depend on it'')';
                else, note = ' (contradicts ''non-increasing'')'; end
            end
            cands{end + 1} = sprintf('%s %.4g -> %.4g%s', name, c, v, note); %#ok<AGROW>
        end
    end
end
if ~isempty(cands)
    L{end + 1} = '  Any one of these changes alone makes all data consistent:';
    for c = cands, L{end + 1} = ['   - ' c{1}]; end %#ok<AGROW>
end
if ~isempty(ruled)
    L{end + 1} = ['  Cannot explain the data on their own: ' strjoin(ruled, ', ') '.'];
end
end
