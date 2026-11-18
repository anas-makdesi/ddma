function M = monitor_store(M, Z, Y, ids, status)
%MONITOR_STORE Append points (rows of Z, Y) with ids and status (0 clean, 2 quarantined);
%   storage grows by doubling; each point goes into its grid cell (if there is a grid).
k = size(Z, 1);
cap = size(M.Z, 1);
if M.K + k > cap
    cap = max(2 * cap, M.K + k);
    M.Z(cap, M.n) = 0; M.Y(cap, M.m) = 0; M.ids(cap, 1) = 0; M.st(cap, 1) = 0;
    if ~isempty(M.grid)
        M.grid.next(cap, 1) = int64(-1);
        M.grid.next(M.K + 1:cap) = int64(-1);
    end
end
r = M.K + (1:k);
M.Z(r, :) = Z; M.Y(r, :) = Y; M.ids(r) = ids(:); M.st(r) = status;
M.K = M.K + k;
if ~isempty(M.grid)
    for t = r, M = grid_insert(M, t); end
end
end

function M = grid_insert(M, t)
G = M.grid; z = M.Z(t, :); y = M.Y(t, :);
c = 0;
for j = 1:M.n
    u = 0;
    if G.cdim(j) > 1
        u = floor((z(j) - G.glo(j)) * G.ginv(j));
        u = min(max(u, 0), G.cdim(j) - 1);
    end
    c = c * G.cdim(j) + u;
end
c1 = c + 1;
G.zlo(:, c1) = min(G.zlo(:, c1), z'); G.zhi(:, c1) = max(G.zhi(:, c1), z');
G.ylo(:, c1) = min(G.ylo(:, c1), y'); G.yhi(:, c1) = max(G.yhi(:, c1), y');
G.cnt(c1) = G.cnt(c1) + 1;
G.next(t) = G.head(c1);
G.head(c1) = int64(t - 1);
M.grid = G;
end
