function M = monitor_add(M, Z, Y)
%MONITOR_ADD Store data without checking them (e.g. the data the model was learned from).
if size(Z, 2) ~= M.n || size(Y, 2) ~= M.m || size(Z, 1) ~= size(Y, 1)
    error('ddma:monitor', 'expected Z (K x %d) and Y (K x %d)', M.n, M.m);
end
K = size(Z, 1);
for k = 1:K
    M = ddma.internal.monitor_store(M, Z(k, :), Y(k, :), -(M.n_seed + k));
end
M.n_seed = M.n_seed + K;
end
