function M = monitor_record(M, viols)
%MONITOR_RECORD Add violations to the record and call the callback.
M.violations = [M.violations, viols];
idx = unique([viols.index]);
for t = idx
    if ~any(M.violating == t), M.violating(end + 1, 1) = t; end
end
if ~isempty(M.callback), M.callback(viols); end
end
