function txt = monitor_report(M)
%MONITOR_REPORT Plain-words summary: checks, violations, and the single changes of the
%   assumptions that would explain them. Same text as AssumptionMonitor.report (transition
%   numbers are 1-based here).
nv = numel(M.violating);
L = {sprintf('Assumption monitor: %d transitions checked, %d inconsistent with the assumptions', ...
             M.n_checked, nv)};
if nv, L{1} = sprintf('%s (first: #%d).', L{1}, M.violating(1)); else, L{1} = [L{1} '.']; end
th = M.d_hi - M.d_lo;
for i = 1:M.m
    if isempty(M.violations), break; end
    sel = [M.violations.output] == i;
    if ~any(sel), continue; end
    V = M.violations(sel);
    nmodel = sum(strcmp({V.kind}, 'model'));
    plural = 's'; if numel(V) == 1, plural = ''; end
    s = sprintf('%s: %d violation%s, worst %.4g beyond the assumptions', ...
                ddma.internal.monitor_name(M, 'out', i), numel(V), plural, max([V.gap]));
    if nmodel, s = sprintf('%s (%d found through the model bounds only)', s, nmodel); end
    L{end + 1} = [s '.']; %#ok<AGROW>
    cands = {}; ruled = {};
    if M.need_width(i) > th(i) + M.tol
        cands{end + 1} = sprintf('disturbance width %.4g -> %.4g', th(i), M.need_width(i)); %#ok<AGROW>
    end
    for j = 1:M.n
        sides = {'lower', 'upper'};
        for t = 1:2
            if t == 1, v = M.need_lo(i, j); c = M.slope_lo(i, j);
            else, v = M.need_hi(i, j); c = M.slope_hi(i, j); end
            name = sprintf('%s slope bound w.r.t. %s', sides{t}, ddma.internal.monitor_name(M, 'in', j));
            if isnan(v)
                ruled{end + 1} = name; %#ok<AGROW>
            elseif v ~= c
                note = '';
                if t == 1 && c == 0 && v < 0
                    if M.slope_hi(i, j) == 0, note = ' (contradicts ''does not depend on it'')';
                    else, note = ' (contradicts ''non-decreasing'')'; end
                end
                if t == 2 && c == 0 && v > 0
                    if M.slope_lo(i, j) == 0, note = ' (contradicts ''does not depend on it'')';
                    else, note = ' (contradicts ''non-increasing'')'; end
                end
                cands{end + 1} = sprintf('%s %.4g -> %.4g%s', name, c, v, note); %#ok<AGROW>
            end
        end
    end
    if ~isempty(cands)
        L{end + 1} = '  Any one of these changes alone makes all stored data consistent:'; %#ok<AGROW>
        for c = cands, L{end + 1} = ['   - ' c{1}]; end %#ok<AGROW>
    elseif nmodel == numel(V)
        L{end + 1} = ['  No witness among the stored data (model bounds only): ' ...
                      'at least the disturbance width must grow by the gap.']; %#ok<AGROW>
    end
    if ~isempty(ruled)
        L{end + 1} = ['  Cannot explain the data on their own: ' strjoin(ruled, ', ') '.']; %#ok<AGROW>
    end
end
if M.fallback_active
    L{end + 1} = sprintf('Fallback active since #%d.', M.fallback_since);
end
if ~isempty(M.changes)
    L{end + 1} = ['Assumptions widened (the model must be refit; its guarantees now rest ' ...
                  'on bounds fitted to the data):'];
    for c = M.changes, L{end + 1} = ['   - ' c{1}]; end %#ok<AGROW>
end
txt = strjoin(L, sprintf('\n'));
end
