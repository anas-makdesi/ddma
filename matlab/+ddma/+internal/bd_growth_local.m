function [lo, up] = bd_growth_local(Z, y, edges, gl, gh, theta, slab, useMex)
%BD_GROWTH_LOCAL Per-cell cone bounds: min over the points in the cell ('cell') or in its
%   slabs ('slab') of the max of the cone over the cell. For each pattern (per axis: left
%   of / in / right of the point's cell) one scatter, then exclusive running min/max along
%   the axes where the pattern says left/right (the cone term there is linear in one edge
%   of the destination cell). Same algorithm as cpp/ddma_growth.hpp (useMex: that kernel).
if useMex
    [lo, up] = ddma_growth_mex(1 + slab, Z, y, edges, gl, gh, theta);
    return
end
n = numel(edges);
ne = cellfun(@numel, edges(:)');
shape = ne - 1; N = prod(shape);
shp = shape; if n == 1, shp = [shape 1]; end
c = ddma.internal.bd_cells_of(edges, Z);
f = ddma.flat_index(shape, c);
ownU = zeros(size(Z)); ownL = ownU;
for j = 1:n
    e = edges{j};
    t0 = e(c(:, j) + 1) - Z(:, j); t1 = e(c(:, j) + 2) - Z(:, j);
    t0 = t0(:); t1 = t1(:);
    ownU(:, j) = max(max(gh(j) * t0, gl(j) * t0), max(gh(j) * t1, gl(j) * t1));
    ownL(:, j) = min(min(gh(j) * t0, gl(j) * t0), min(gh(j) * t1, gl(j) * t1));
end
up = inf(N, 1); lo = -inf(N, 1);
for p = 0:3^n-1
    d = mod(floor(p ./ 3 .^ (n-1:-1:0)), 3);   % digit per axis: 0 left, 1 in, 2 right
    if slab, if ~any(d == 1), continue; end
    elseif ~all(d == 1), continue; end
    su = y + theta; sl = y - theta;
    for j = 1:n
        if d(j) == 1, su = su + ownU(:, j); sl = sl + ownL(:, j);
        elseif d(j) == 2, su = su - gh(j) * Z(:, j); sl = sl - gl(j) * Z(:, j);
        else, su = su - gl(j) * Z(:, j); sl = sl - gh(j) * Z(:, j); end
    end
    AU = reshape(ddma.internal.bd_scatter(f, su, N, @min, Inf), shp);
    AL = reshape(ddma.internal.bd_scatter(f, sl, N, @max, -Inf), shp);
    for j = 1:n
        if d(j) == 1, continue; end
        right = d(j) == 2;
        e = edges{j}(:)';
        if right
            cu = gh(j); eu = gh(j) >= 0; cl = gl(j); el = gl(j) < 0;
        else
            cu = gl(j); eu = gl(j) >= 0; cl = gh(j); el = gh(j) < 0;
        end
        EU = reshape(e((1:shape(j)) + eu), [ones(1, j - 1), shape(j), 1]);
        EL = reshape(e((1:shape(j)) + el), [ones(1, j - 1), shape(j), 1]);
        AU = excl_cum(AU, j, right, @cummin, Inf) + cu * EU;
        AL = excl_cum(AL, j, right, @cummax, -Inf) + cl * EL;
    end
    up = min(up, AU(:)); lo = max(lo, AL(:));
end
end

function B = excl_cum(A, j, forward, op, fill)
%EXCL_CUM Exclusive running min/max along dimension j (forward or backward).
if ~forward, A = flip(A, j); end
C = op(A, j);
idx = repmat({':'}, 1, max(ndims(A), j));
idx{j} = 1:size(A, j) - 1;
first = size(A); first(j) = 1;
B = cat(j, repmat(fill, first), C(idx{:}));
if ~forward, B = flip(B, j); end
end
