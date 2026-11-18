function txt = monitor_report(M)
%MONITOR_REPORT Plain-words summary: checks, violations, the levels that found them, and
%   the single changes of the assumptions that would explain them. Same text as
%   AssumptionMonitor.report (transition numbers are 1-based here).
nv = numel(M.violating);
L = {sprintf('Assumption monitor: %d transitions checked, %d inconsistent with the assumptions', ...
             M.n_checked, nv)};
if nv, L{1} = sprintf('%s (first: #%d).', L{1}, M.violating(1)); else, L{1} = [L{1} '.']; end
st = ddma.monitor_status(M);
if nv
    lv = {M.violations.level};
    L{end + 1} = sprintf('Found by: cells %d, sample %d, full check %d. Quarantined points: %d.', ...
        sum(strcmp(lv, 'cells')), sum(strcmp(lv, 'sample')), sum(strcmp(lv, 'full')), st.quarantined);
end
if st.pending
    L{end + 1} = sprintf(['%d full checks pending (ddma.monitor_flush runs them); the needed ' ...
        'changes below cover the pairs checked so far.'], st.pending);
end
th = M.d_hi - M.d_lo;
need = ddma.monitor_needed(M);
out = @(i) ddma.internal.monitor_name(M, 'out', i);
inp = @(j) ddma.internal.monitor_name(M, 'in', j);
for i = 1:M.m
    if isempty(M.violations), break; end
    V = M.violations([M.violations.output] == i);
    if isempty(V), continue; end
    plural = 's'; if numel(V) == 1, plural = ''; end
    L{end + 1} = sprintf('%s: %d violation%s, worst %.4g beyond the assumptions.', out(i), ...
                         numel(V), plural, max([V.gap])); %#ok<AGROW>
    ex = ddma.internal.monitor_explain(i, need, th, M.slope_lo, M.slope_hi, M.tol, out, inp);
    if isempty(ex) && all(strcmp({V.level}, 'cells'))
        ex = {['  Found through the model''s cell bounds only: at least the disturbance ' ...
               'width must grow by the gap.']};
    end
    L = [L, ex]; %#ok<AGROW>
end
if M.fallback_active
    if isempty(M.fallback_reason)
        L{end + 1} = sprintf('Fallback active since #%d.', M.fallback_since);
    else
        L{end + 1} = sprintf('Fallback active since #%d: %s.', M.fallback_since, M.fallback_reason);
    end
end
if ~isempty(M.changes)
    L{end + 1} = ['Assumptions widened (the model must be refit; its guarantees now rest ' ...
                  'on bounds fitted to the data):'];
    for c = M.changes, L{end + 1} = ['   - ' c{1}]; end %#ok<AGROW>
end
for c = M.notes, L{end + 1} = ['Note: ' c{1} '.']; end %#ok<AGROW>
txt = strjoin(L, sprintf('\n'));
end
