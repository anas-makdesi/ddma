function S = fit(Z, Y, edges, d_lo, d_hi, signs, backend)
%FIT Tightest interval-valued simulating map on a fixed rectangular partition.
%   S = ddma.fit(Z, Y, edges, d_lo, d_hi) computes sigma (TAC paper, Theorem 2) with the
%   O(|K| log|Q| + |Q|) sweep (Proposition 5).
%     Z (K x n), Y (K x m) data; edges: cell array of edge vectors (one per input);
%     d_lo, d_hi: disturbance bounds (scalars or 1 x m).
%   Optional: signs (m x n, -1/0/+1), backend 'auto' (MEX if compiled) | 'mex' | 'matlab'.
%   S.lower, S.upper: N x m, cells in column-major order; S.shape, S.edges, S.d_lo, ...
[K, n] = size(Z); m = size(Y, 2);
if nargin < 6 || isempty(signs), signs = ones(m, n); end
if nargin < 7, backend = 'auto'; end
d_lo = d_lo(:)' .* ones(1, m); d_hi = d_hi(:)' .* ones(1, m);
if any(d_lo > d_hi), error('ddma:fit', 'need d_lo <= d_hi'); end
theta = d_hi - d_lo;
shape = ddma.grid_shape(edges);
useMex = strcmp(backend, 'mex') || (strcmp(backend, 'auto') && exist('ddma_fit_mex', 'file') == 3);
if useMex
    [lower, upper] = ddma_fit_mex(Z, Y, edges, theta, int8(signs));
else
    [lower, upper] = fit_matlab(Z, Y, edges, theta, signs, shape);
end
S = struct('edges', {edges}, 'shape', shape, 'lower', lower, 'upper', upper, ...
           'd_lo', d_lo, 'd_hi', d_hi, 'signs', signs);
end

function [lower, upper] = fit_matlab(Z, Y, edges, theta, signs, shape)
[K, n] = size(Z); m = size(Y, 2);
N = prod(shape);
last = shape - 1;
ub = zeros(K, n); lb = zeros(K, n);
for j = 1:n
    ub(:, j) = ddma.searchsorted(edges{j}, Z(:, j), 'right') - 2;  % last cell with upper edge <= z
    lb(:, j) = ddma.searchsorted(edges{j}, Z(:, j), 'left');       % first cell with lower edge >= z
end
ubOk = ub >= 0; lbOk = lb <= last;
ub = min(ub, last); lb = max(lb, 0);
lower = zeros(N, m); upper = zeros(N, m);
shp = shape; if numel(shp) == 1, shp = [shp 1]; end
for i = 1:m
    s = signs(i, :);
    [qu, vu] = corner(s, true, ub, lb, ubOk, lbOk, last);
    [ql, vl] = corner(s, false, ub, lb, ubOk, lbOk, last);
    up = scatter(ddma.flat_index(shape, qu(vu, :)), Y(vu, i) + theta(i), N, @min, Inf);
    lw = scatter(ddma.flat_index(shape, ql(vl, :)), Y(vl, i) - theta(i), N, @max, -Inf);
    up = reshape(up, shp); lw = reshape(lw, shp);
    for j = 1:n
        if s(j) >= 0, up = flip(cummin(flip(up, j), j), j); else, up = cummin(up, j); end
        if s(j) <= 0, lw = flip(cummax(flip(lw, j), j), j); else, lw = cummax(lw, j); end
    end
    upper(:, i) = up(:); lower(:, i) = lw(:);
end
end

function [q, ok] = corner(s, isUpper, ub, lb, ubOk, lbOk, last)
%CORNER Cell each data point writes to, for the upper or lower bound of one output.
[K, n] = size(ub);
q = zeros(K, n); ok = true(K, 1);
for j = 1:n
    if s(j) == 0
        q(:, j) = last(j);
    elseif (s(j) > 0) == isUpper
        q(:, j) = ub(:, j); ok = ok & ubOk(:, j);
    else
        q(:, j) = lb(:, j); ok = ok & lbOk(:, j);
    end
end
end

function v = scatter(f, vals, N, fun, empty)
%SCATTER Reduce vals into N bins with fun; bins without data get the value empty.
%   (Empty bins are filled explicitly: Octave's accumarray ignores the fill value for
%   @min/@max.)
v = repmat(empty, N, 1);
if isempty(f), return; end
r = accumarray(f, vals, [N 1], fun);
hit = accumarray(f, 1, [N 1]) > 0;
v(hit) = r(hit);
end
