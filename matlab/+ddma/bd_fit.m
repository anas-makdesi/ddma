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
%   Online use: ddma.bd_update adds data exactly (same model as a refit on all data);
%   ddma.bd_update_local (growth methods) updates a window of cells around each point;
%   then ddma.abstraction_refresh and ddma.safety(..., known_winning).
n = size(Z, 2); m = size(Y, 2);
if size(Y, 1) ~= size(Z, 1), error('ddma:bd_fit', 'Z and Y must have the same number of rows'); end
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
if strcmp(backend, 'mex') && exist('ddma_growth_mex', 'file') ~= 3
    error('ddma:bd_fit', 'ddma_growth_mex is not compiled (run build_mex)');
end
g_lo = g_lo .* ones(m, n); g_hi = g_hi .* ones(m, n);
if any(g_lo(:) > g_hi(:)) || ~all(isfinite([g_lo(:); g_hi(:)]))
    error('ddma:bd_fit', 'derivative bounds must be finite with g_lo <= g_hi');
end
d_lo = d_lo(:)' .* ones(1, m); d_hi = d_hi(:)' .* ones(1, m);
if any(d_lo > d_hi), error('ddma:bd_fit', 'need d_lo <= d_hi'); end

shape = ddma.grid_shape(edges);
N = prod(shape); NV = prod(shape + 1);
raw = cell(1, m);
for i = 1:m
    R = struct('gl', g_lo(i, :), 'gh', g_hi(i, :), 'd_lo', d_lo(i), 'd_hi', d_hi(i));
    if isGrowth
        R.kind = 'growth';
        R.vlo = -inf(NV, 1); R.vup = inf(NV, 1);
        if ~isempty(local), R.llo = -inf(N, 1); R.lup = inf(N, 1); end
    else
        R.kind = 'sign';
        if strcmp(method, 'two_matrix'), which = 'two'; else, which = 'all'; end
        [R.a, R.s] = ddma.internal.bd_sign_patterns(R.gl, R.gh, which);
        L = size(R.a, 1);
        R.slo = -inf(N, L); R.shi = inf(N, L);
    end
    raw{i} = R;
end
S = struct('edges', {edges}, 'shape', shape, 'lower', -inf(N, m), 'upper', inf(N, m), ...
           'd_lo', d_lo, 'd_hi', d_hi, 'method', method, 'local', local, ...
           'g_lo', g_lo, 'g_hi', g_hi, 'backend', backend, 'n_data', 0);
S.raw = raw;
S.changed = zeros(0, 2 * n);   % boxes of cells changed by bd_update_local: [first, last]
S = ddma.bd_update(S, Z, Y);
end
