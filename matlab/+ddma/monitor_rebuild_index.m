function M = monitor_rebuild_index(M)
%MONITOR_REBUILD_INDEX Rebuild the cell grid used for pruning from all stored points (done
%   by ddma.monitor_add, and when the data have doubled since the last build). Only with
%   the MEX kernel; the MATLAB code checks all pairs (same results).
if ~(M.prune && M.use_mex) || M.K == 0
    M.grid = []; return;
end
G = ddma_monitor_mex('grid', M.Z, M.Y, M.K);
nxt = -ones(size(M.Z, 1), 1, 'int64');
nxt(1:M.K) = G.next;
G.next = nxt;
G = rmfield(G, 'cell');
M.grid = G;
M.grid_K = M.K;
end
