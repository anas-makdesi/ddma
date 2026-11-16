function M = monitor_store(M, z, y, id)
%MONITOR_STORE Add one data point to a monitor (reservoir sampling beyond max_points).
M.n_seen = M.n_seen + 1;
if size(M.Z, 1) < M.max_points
    M.Z(end + 1, :) = z; M.Y(end + 1, :) = y; M.ids(end + 1, 1) = id;
else
    k = floor(rand() * M.n_seen) + 1;
    if k <= M.max_points
        M.Z(k, :) = z; M.Y(k, :) = y; M.ids(k) = id;
    end
end
end
