function M = monitor_add(M, Z, Y)
%MONITOR_ADD Store data without checking them (e.g. the data the model was learned from;
%   check those with ddma.check_data). Pending full checks are run first.
if size(Z, 2) ~= M.n || size(Y, 2) ~= M.m || size(Z, 1) ~= size(Y, 1)
    error('ddma:monitor', 'expected Z (K x %d) and Y (K x %d)', M.n, M.m);
end
if ~all(isfinite([Z(:); Y(:)])), error('ddma:monitor', 'data must be finite'); end
if M.K > M.pending_from, M = ddma.monitor_flush(M); end
K = size(Z, 1);
M = ddma.internal.monitor_store(M, Z, Y, -(M.n_seed + (1:K)), 0);
M.n_seed = M.n_seed + K;
M.pending_from = M.K;
M = ddma.monitor_rebuild_index(M);
end
