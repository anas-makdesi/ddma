function A = abstraction(Xedges, Uedges, S, g_bounds, features, assumption)
%ABSTRACTION Symbolic abstraction of x+ in f(x,u) + g(x,u) + W from a fitted map S.
%   A = ddma.abstraction(Xedges, Uedges, S, g_bounds, features, assumption)
%     Xedges, Uedges: state and input partitions (cell arrays of edges);
%     S: map of f + W from ddma.fit, on the (x, u) coordinates listed in features
%        (default all), whose partition must be the projection of Xedges x Uedges;
%     g_bounds: [] (g = 0) or handle [glo, ghi] = g_bounds(xlo, xhi, ulo, uhi);
%     assumption: [] or 2 x nx [lo; hi] state box for assume-guarantee contracts
%        (use -Inf/Inf for free coordinates; finite bounds on cell edges).
%   A.first, A.last: nQ x nP x nx (0-based successor cell ranges), A.enabled: nQ x nP.
%   Cells are numbered in column-major order.
nx = numel(Xedges); nu = numel(Uedges);
if nargin < 4, g_bounds = []; end
if nargin < 5 || isempty(features), features = 1:(nx + nu); end
if nargin < 6, assumption = []; end
joint = [Xedges(:); Uedges(:)];
for k = 1:numel(features)
    if ~isequal(joint{features(k)}(:), S.edges{k}(:))
        error('ddma:abstraction', 'S.edges must equal the state x input partition on features');
    end
end
shX = ddma.grid_shape(Xedges); shU = ddma.grid_shape(Uedges);
nQ = prod(shX); nP = prod(shU);
qs = ddma.all_subs(shX); ps = ddma.all_subs(shU);
[xlo, xhi] = ddma.cell_bounds(Xedges, qs);
[ulo, uhi] = ddma.cell_bounds(Uedges, ps);
Xlo = cellfun(@(e) e(1), Xedges(:)'); Xhi = cellfun(@(e) e(end), Xedges(:)');
if ~isempty(assumption)
    cLo = zeros(1, nx); cHi = shX - 1;
    for j = 1:nx
        if isfinite(assumption(1, j)), cLo(j) = ddma.searchsorted(Xedges{j}, assumption(1, j), 'left'); end
        if isfinite(assumption(2, j)), cHi(j) = ddma.searchsorted(Xedges{j}, assumption(2, j), 'right') - 2; end
    end
end
first = zeros(nQ, nP, nx, 'int32'); last = first;
enabled = false(nQ, nP);
for p = 1:nP
    sub = [qs, repmat(ps(p, :), nQ, 1)];
    f = ddma.flat_index(S.shape, sub(:, features));
    lo = S.lower(f, :); hi = S.upper(f, :);
    if ~isempty(g_bounds)
        [glo, ghi] = g_bounds(xlo, xhi, repmat(ulo(p, :), nQ, 1), repmat(uhi(p, :), nQ, 1));
        lo = lo + glo; hi = hi + ghi;
    end
    if ~isempty(assumption)
        lo = min(max(lo, assumption(1, :)), assumption(2, :));
        hi = min(max(hi, assumption(1, :)), assumption(2, :));
    end
    ok = all(lo <= hi, 2) & all(isfinite(lo) & isfinite(hi), 2) & all(lo >= Xlo & hi <= Xhi, 2);
    F = zeros(nQ, nx); L = F;
    for j = 1:nx
        e = Xedges{j}; k = numel(e) - 2;
        F(ok, j) = min(max(ddma.searchsorted(e, lo(ok, j), 'right') - 1, 0), k);
        L(ok, j) = min(max(ddma.searchsorted(e, hi(ok, j), 'right') - 1, 0), k);
        if ~isempty(assumption)
            F(ok, j) = min(max(F(ok, j), cLo(j)), cHi(j));
            L(ok, j) = min(max(L(ok, j), cLo(j)), cHi(j));
        end
    end
    first(:, p, :) = reshape(int32(F), nQ, 1, nx);
    last(:, p, :) = reshape(int32(L), nQ, 1, nx);
    enabled(:, p) = ok;
end
A = struct('Xedges', {Xedges}, 'Uedges', {Uedges}, 'shapeX', shX, 'shapeU', shU, ...
           'first', first, 'last', last, 'enabled', enabled);
end
