function cnt = sat_count(sat, shX, F, L)
%SAT_COUNT Number of cells outside W in the boxes [F(k,:), L(k,:)] (0-based, k x nx),
%   from the summed-area table of ddma.internal.sat_build. O(2^nx) per box.
nx = numel(shX);
stride = [1, cumprod(shX(1:end-1) + 1)];
F = double(F); L = double(L) + 1;
cnt = zeros(size(F, 1), 1);
for c = 0:2^nx - 1
    lin = ones(size(F, 1), 1);
    nUp = 0;
    for j = 1:nx
        if bitget(c, j), v = L(:, j); nUp = nUp + 1; else, v = F(:, j); end
        lin = lin + v * stride(j);
    end
    cnt = cnt + (-1) ^ (nx - nUp) * sat(lin);
end
end
