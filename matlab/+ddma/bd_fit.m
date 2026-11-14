function S = bd_fit(Z, Y, edges, g_lo, g_hi, d_lo, d_hi, method, local, backend)
%BD_FIT Model of a map with bounded partial derivatives on a rectangular partition.
%   S = ddma.bd_fit(Z, Y, edges, g_lo, g_hi, d_lo, d_hi, method) learns, for every cell
%   and output i, an interval containing f_i(z) + d_i for all z in the cell, from data
%   Y(k, i) = f_i(Z(k, :)) + d_i with d_i in [d_lo(i), d_hi(i)], for an unknown f whose
%   partial derivatives satisfy  g_lo(i, j) <= df_i/dz_j <= g_hi(i, j).
%   Thesis, Chapter 3 (approaches of Section 3.4); same models as the Python
%   BoundedDerivativeModel(method=...):
%     'two_matrix' (approach 1, default)  two monotone transforms, interval final step
%     'all_matrix' (2)                    all 2^n slope choices per output
%     'lin_mono'   (3)                    as 'all_matrix', linear-programming final step
%     'growth'     (4)                    growth cones at the grid vertices
%     'lin_growth' (5)                    as 'growth', linear-programming final step
%   Z (K x n), Y (K x m); edges: cell array of edge vectors (one per input);
%   g_lo, g_hi: m x n (or 1 x n for all outputs); d_lo, d_hi: scalars or 1 x m.
%   Optional: local = '' (default) | 'cell' | 'slab' (growth methods: also intersect with
%   the cones of the data in each cell / in its slabs); backend = 'auto' (MEX if compiled)
%   | 'mex' | 'matlab'. The 'lin_*' methods need linprog (Optimization Toolbox) or
%   Octave's glpk; their bounds are certified by weak duality, so they stay sound
%   whatever the solver tolerance.
%   S has the fields of ddma.fit (edges, shape, lower, upper: N x m, cells in
%   column-major order, d_lo, d_hi), so ddma.bounds and ddma.abstraction take it as is.
%   As with ddma.fit, subtract a known part g from the data first and pass its bounds to
%   ddma.abstraction.
[K, n] = size(Z); m = size(Y, 2);
if size(Y, 1) ~= K, error('ddma:bd_fit', 'Z and Y must have the same number of rows'); end
if numel(edges) ~= n, error('ddma:bd_fit', 'one edge vector per input'); end
if nargin < 8 || isempty(method), method = 'two_matrix'; end
if nargin < 9, local = ''; end
if nargin < 10 || isempty(backend), backend = 'auto'; end
methods = {'two_matrix', 'all_matrix', 'lin_mono', 'growth', 'lin_growth'};
if ~any(strcmp(method, methods)), error('ddma:bd_fit', 'unknown method %s', method); end
if ~isempty(local) && ~any(strcmp(local, {'cell', 'slab'}))
    error('ddma:bd_fit', 'local must be '''', ''cell'' or ''slab''');
end
isGrowth = any(strcmp(method, {'growth', 'lin_growth'}));
if ~isempty(local) && ~isGrowth
    error('ddma:bd_fit', 'local is available for the growth methods');
end
if ~any(strcmp(backend, {'auto', 'mex', 'matlab'}))
    error('ddma:bd_fit', 'backend must be ''auto'', ''mex'' or ''matlab''');
end
g_lo = g_lo .* ones(m, n); g_hi = g_hi .* ones(m, n);
if any(g_lo(:) > g_hi(:)) || ~all(isfinite([g_lo(:); g_hi(:)]))
    error('ddma:bd_fit', 'derivative bounds must be finite with g_lo <= g_hi');
end
d_lo = d_lo(:)' .* ones(1, m); d_hi = d_hi(:)' .* ones(1, m);
if any(d_lo > d_hi), error('ddma:bd_fit', 'need d_lo <= d_hi'); end
useMex = strcmp(backend, 'mex') || (strcmp(backend, 'auto') && exist('ddma_growth_mex', 'file') == 3);
if strcmp(backend, 'mex') && exist('ddma_growth_mex', 'file') ~= 3
    error('ddma:bd_fit', 'ddma_growth_mex is not compiled (run build_mex)');
end
lp = strncmp(method, 'lin_', 4);

shape = ddma.grid_shape(edges);
N = prod(shape);
[clo, chi] = ddma.cell_bounds(edges, ddma.all_subs(shape));  % N x n
lower = zeros(N, m); upper = zeros(N, m);
for i = 1:m
    gl = g_lo(i, :); gh = g_hi(i, :);
    if isGrowth
        [lo, up] = growth_cells(Z, Y(:, i), edges, gl, gh, d_hi(i) - d_lo(i), lp, local, useMex, clo, chi);
    else
        if strcmp(method, 'two_matrix'), which = 'two'; else, which = 'all'; end
        [a, s] = sign_patterns(gl, gh, which);
        L = size(a, 1);
        fb = backend; if strcmp(fb, 'mex'), fb = 'auto'; end
        M = ddma.fit(Z, Y(:, i) - Z * a', edges, d_lo(i) * ones(1, L), d_hi(i) * ones(1, L), s, fb);
        ap = max(a, 0); am = min(a, 0);
        az_lo = clo * ap' + chi * am';   % interval of a . z over each cell, N x L
        az_hi = chi * ap' + clo * am';
        up = min(az_hi + M.upper, [], 2);
        lo = max(az_lo + M.lower, [], 2);
        if lp
            up = ddma.max_min_affine(M.upper, a, clo, chi, up);
            lo = -ddma.max_min_affine(-M.lower, -a, clo, chi, -lo);
        end
    end
    lower(:, i) = lo; upper(:, i) = up;
end
S = struct('edges', {edges}, 'shape', shape, 'lower', lower, 'upper', upper, ...
           'd_lo', d_lo, 'd_hi', d_hi, 'method', method, 'local', local, ...
           'g_lo', g_lo, 'g_hi', g_hi);
end

function [a, s] = sign_patterns(gl, gh, which)
%SIGN_PATTERNS Slope vectors a (L x n) and signs s of the monotone transforms f - a.z.
n = numel(gl);
lo = min(gl, 0); hi = max(gh, 0);
zero = gl == 0 & gh == 0;
if strcmp(which, 'two')
    a = [lo; hi]; s = [ones(1, n); -ones(1, n)];
    a(:, zero) = 0; s(:, zero) = 0;
    if isequal(a(1, :), a(2, :)), a = a(1, :); s = s(1, :); end
    return
end
free = find(~zero);
L = 2 ^ numel(free);
a = zeros(L, n); s = zeros(L, n);
for r = 0:L-1  % same order as itertools.product: first free axis varies slowest
    for t = 1:numel(free)
        j = free(t);
        if bitand(r, 2 ^ (numel(free) - t)), a(r + 1, j) = hi(j); s(r + 1, j) = -1;
        else, a(r + 1, j) = lo(j); s(r + 1, j) = 1; end
    end
end
end

function [lo, up] = growth_cells(Z, y, edges, gl, gh, theta, lp, local, useMex, clo, chi)
%GROWTH_CELLS Cell bounds of one output from the growth cones (approaches 4 and 5).
n = numel(edges);
ne = cellfun(@numel, edges(:)');
shape = ne - 1;
if useMex
    [vlo, vup] = ddma_growth_mex(0, Z, y, edges, gl, gh, theta);
else
    [vlo, vup] = growth_vertices(Z, y, edges, gl, gh, theta);
end
N = size(clo, 1);
h = chi - clo;
sub = ddma.all_subs(shape);
up = inf(N, 1); lo = -inf(N, 1);
nc = 2 ^ n;
if lp, offu = zeros(N, nc); offl = offu; SU = zeros(nc, n); SL = SU; end
for c = 0:nc-1
    b = bitand(c, 2 .^ (n-1:-1:0)) > 0;   % corner, itertools.product order
    v = ddma.flat_index(ne, sub + b);
    ru = sum(max(0, where(b, -gl, gh) .* h), 2);
    rl = sum(min(0, where(b, -gh, gl) .* h), 2);
    up = min(up, vup(v) + ru);
    lo = max(lo, vlo(v) + rl);
    if lp  % cone from the vertex, inside the cell
        su = where(b, gl, gh); sl = where(b, gh, gl);
        vc = clo + b .* h;
        offu(:, c + 1) = vup(v) - vc * su';
        offl(:, c + 1) = vlo(v) - vc * sl';
        SU(c + 1, :) = su; SL(c + 1, :) = sl;
    end
end
if lp
    up = ddma.max_min_affine(offu, SU, clo, chi, up);
    lo = -ddma.max_min_affine(-offl, -SL, clo, chi, -lo);
end
if ~isempty(local)
    if useMex
        [llo, lup] = ddma_growth_mex(1 + strcmp(local, 'slab'), Z, y, edges, gl, gh, theta);
    else
        [llo, lup] = growth_local(Z, y, edges, gl, gh, theta, strcmp(local, 'slab'));
    end
    up = min(up, lup); lo = max(lo, llo);
end
end

function r = where(b, x, y)
r = y; r(b) = x(b);
end

function c = cells_of(edges, Z)
%CELLS_OF 0-based cell of each point; points outside the box: nearest edge cell.
n = numel(edges);
c = zeros(size(Z, 1), n);
for j = 1:n
    e = edges{j};
    c(:, j) = min(max(ddma.searchsorted(e, Z(:, j), 'right') - 1, 0), numel(e) - 2);
end
end

function v = scatter(f, vals, N, fun, empty)
v = repmat(empty, N, 1);
if isempty(f), return; end
r = accumarray(f, vals, [N 1], fun);
hit = accumarray(f, 1, [N 1]) > 0;
v(hit) = r(hit);
end

function [vlo, vup] = growth_vertices(Z, y, edges, gl, gh, theta)
%GROWTH_VERTICES Exact bounds at all grid vertices (thesis eqs. 3.20-3.21): scatter to
%   the 2^n vertices of each point's cell, then forward/backward running min/max per axis.
[K, n] = size(Z);
ne = cellfun(@numel, edges(:)');
NV = prod(ne);
c = cells_of(edges, Z);
vup = inf(NV, 1); vlo = -inf(NV, 1);
for m = 0:2^n-1
    b = bitand(m, 2 .^ (n-1:-1:0)) > 0;
    vi = c + b;
    t = zeros(K, n);
    for j = 1:n, t(:, j) = edges{j}(vi(:, j) + 1) - Z(:, j); end
    phi = sum(max(gh .* t, gl .* t), 2);
    plo = sum(min(gh .* t, gl .* t), 2);
    f = ddma.flat_index(ne, vi);
    vup = min(vup, scatter(f, y + theta + phi, NV, @min, Inf));
    vlo = max(vlo, scatter(f, y - theta + plo, NV, @max, -Inf));
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

function [lo, up] = growth_local(Z, y, edges, gl, gh, theta, slab)
%GROWTH_LOCAL Per-cell cone bounds: min over the points in the cell ('cell') or in its
%   slabs ('slab') of the max of the cone over the cell. For each pattern (per axis: left
%   of / in / right of the point's cell) one scatter, then exclusive running min/max along
%   the axes where the pattern says left/right (the cone term there is linear in one edge
%   of the destination cell). Same algorithm as cpp/ddma_growth.hpp.
n = numel(edges);
ne = cellfun(@numel, edges(:)');
shape = ne - 1; N = prod(shape);
shp = shape; if n == 1, shp = [shape 1]; end
c = cells_of(edges, Z);
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
    AU = reshape(scatter(f, su, N, @min, Inf), shp);
    AL = reshape(scatter(f, sl, N, @max, -Inf), shp);
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
