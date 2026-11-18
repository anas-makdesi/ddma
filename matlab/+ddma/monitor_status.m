function s = monitor_status(M)
%MONITOR_STATUS Counts: stored points, pending full checks, quarantined points, cells.
cells = 0; if ~isempty(M.grid), cells = numel(M.grid.cnt); end
s = struct('points', M.K, 'pending', M.K - M.pending_from, ...
           'quarantined', sum(M.st(1:M.K) == 2), 'full', M.full, 'cells', cells);
end
