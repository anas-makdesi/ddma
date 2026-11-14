function [vlo, vup] = bd_growth_vertices(Z, y, edges, gl, gh, theta, useMex)
%BD_GROWTH_VERTICES Exact bounds at all grid vertices (thesis eqs. 3.20-3.21): scatter to
%   the 2^n vertices of each point's cell, then forward/backward running min/max per axis.
%   Column vectors over the vertices, column-major. useMex: the C++ kernel (same values).
if useMex
    [vlo, vup] = ddma_growth_mex(0, Z, y, edges, gl, gh, theta, ddma.internal.mex_opts());
    return
end
[K, n] = size(Z);
ne = cellfun(@numel, edges(:)');
NV = prod(ne);
c = ddma.internal.bd_cells_of(edges, Z);
vup = inf(NV, 1); vlo = -inf(NV, 1);
for m = 0:2^n-1
    b = bitand(m, 2 .^ (n-1:-1:0)) > 0;
    vi = c + b;
    t = zeros(K, n);
    for j = 1:n, t(:, j) = edges{j}(vi(:, j) + 1) - Z(:, j); end
    phi = sum(max(gh .* t, gl .* t), 2);
    plo = sum(min(gh .* t, gl .* t), 2);
    f = ddma.flat_index(ne, vi);
    vup = min(vup, ddma.internal.bd_scatter(f, y + theta + phi, NV, @min, Inf));
    vlo = max(vlo, ddma.internal.bd_scatter(f, y - theta + plo, NV, @max, -Inf));
end
shp = ne; if n == 1, shp = [ne 1]; end
vup = reshape(vup, shp); vlo = reshape(vlo, shp);
for j = 1:n
    E = reshape(edges{j}, [ones(1, j - 1), ne(j), 1]);
    w = cummin(vup - gh(j) * E, j);                         vup = min(vup, w + gh(j) * E);
    w = flip(cummin(flip(vup - gl(j) * E, j), j), j);       vup = min(vup, w + gl(j) * E);
    w = cummax(vlo - gl(j) * E, j);                         vlo = max(vlo, w + gl(j) * E);
    w = flip(cummax(flip(vlo - gh(j) * E, j), j), j);       vlo = max(vlo, w + gh(j) * E);
end
vup = vup(:); vlo = vlo(:);
end
