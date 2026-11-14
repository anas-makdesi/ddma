function A = abstraction(Xedges, Uedges, S, g_bounds, features, assumption)
%ABSTRACTION Symbolic abstraction of x+ in f(x,u) + g(x,u) + W from a fitted map S.
%   A = ddma.abstraction(Xedges, Uedges, S, g_bounds, features, assumption)
%     Xedges, Uedges: state and input partitions (cell arrays of edges);
%     S: map of f + W from ddma.fit or ddma.bd_fit, on the (x, u) coordinates listed in
%        features (default all), whose partition must be the projection of Xedges x Uedges;
%     g_bounds: [] (g = 0) or handle [glo, ghi] = g_bounds(xlo, xhi, ulo, uhi);
%     assumption: [] or 2 x nx [lo; hi] state box for assume-guarantee contracts
%        (use -Inf/Inf for free coordinates; finite bounds on cell edges).
%   A.first, A.last: nQ x nP x nx (0-based successor cell ranges), A.enabled: nQ x nP.
%   Cells are numbered in column-major order. A.opts keeps g_bounds, features and
%   assumption for ddma.abstraction_refresh.
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
A = struct('Xedges', {Xedges}, 'Uedges', {Uedges}, 'shapeX', shX, 'shapeU', shU, ...
           'first', zeros(nQ, nP, nx, 'int32'), 'last', zeros(nQ, nP, nx, 'int32'), ...
           'enabled', false(nQ, nP));
A.opts = struct('g_bounds', {g_bounds}, 'features', features, 'assumption', assumption);
A = ddma.internal.abstraction_rows(A, S, (1:nQ)');
end
