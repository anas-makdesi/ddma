function M = monitor_add(M, Z, Y)
%MONITOR_ADD Store data without checking them (e.g. the data the model was learned from;
%   check those with ddma.check_data). Pending full checks are run first. Room is reserved
%   for at least half as many new points, so a run does not trigger a copy.
if size(Z, 2) ~= M.n || size(Y, 2) ~= M.m || size(Z, 1) ~= size(Y, 1)
    error('ddma:monitor', 'expected Z (K x %d) and Y (K x %d)', M.n, M.m);
end
if ~all(isfinite([Z(:); Y(:)])), error('ddma:monitor', 'data must be finite'); end
if M.K > M.pending_from, M = ddma.monitor_flush(M); end
K = size(Z, 1);
cap = M.K + K; cap = cap + max(4096, floor(cap / 2));
if cap > size(M.Z, 1)
    M.Z(cap, M.n) = 0; M.Y(cap, M.m) = 0; M.ids(cap, 1) = 0; M.st(cap, 1) = 0;
    if ~isempty(M.grid), M.grid.next(cap, 1) = int64(-1); M.grid.next(M.K + 1:cap) = int64(-1); end
end
M = ddma.internal.monitor_store(M, Z, Y, -(M.n_seed + (1:K)), 0);
M.n_seed = M.n_seed + K;
M.pending_from = M.K;
M = ddma.monitor_rebuild_index(M);
end
