function A = abstraction(Xedges, Uedges, S, g_bounds, features, assumption, lazy, backend)
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
%   lazy = true: no transitions are stored (A.first, A.last, A.enabled are empty; A.S
%   keeps the map). ddma.safety then evaluates a transition only when its scan reaches
%   it: states outside the safe set are never expanded and each state stops at its first
%   good input. Memory O(nQ) instead of O(nQ nP); same winning set. ddma.controller_input
%   computes the allowed inputs on demand. reach_avoid needs an eager abstraction.
%   Same as the Python SymbolicAbstraction(..., lazy=True).
%   backend: 'auto' (default: the MEX kernel ddma_abs_mex if built, on all cores, see
%   ddma.set_num_threads), 'mex' or 'matlab'. All give the same abstraction.
nx = numel(Xedges); nu = numel(Uedges);
if nargin < 4, g_bounds = []; end
if nargin < 5 || isempty(features), features = 1:(nx + nu); end
if nargin < 6, assumption = []; end
if nargin < 7 || isempty(lazy), lazy = false; end
if nargin < 8 || isempty(backend), backend = 'auto'; end
if ~any(strcmp(backend, {'auto', 'mex', 'matlab'}))
    error('ddma:abstraction', 'backend must be ''auto'', ''mex'' or ''matlab''');
end
if strcmp(backend, 'mex') && exist('ddma_abs_mex', 'file') ~= 3
    error('ddma:abstraction', 'ddma_abs_mex is not built (run build_mex)');
end
joint = [Xedges(:); Uedges(:)];
for k = 1:numel(features)
    if ~isequal(joint{features(k)}(:), S.edges{k}(:))
        error('ddma:abstraction', 'S.edges must equal the state x input partition on features');
    end
end
shX = ddma.grid_shape(Xedges); shU = ddma.grid_shape(Uedges);
nQ = prod(shX); nP = prod(shU);
A = struct('Xedges', {Xedges}, 'Uedges', {Uedges}, 'shapeX', shX, 'shapeU', shU, ...
           'first', zeros(0, 0, nx, 'int32'), 'last', zeros(0, 0, nx, 'int32'), ...
           'enabled', false(0, 0), 'lazy', logical(lazy));
A.opts = struct('g_bounds', {g_bounds}, 'features', features, 'assumption', assumption, ...
                'backend', backend);
if A.lazy
    A.S = S;
    return
end
A.first = zeros(nQ, nP, nx, 'int32'); A.last = A.first; A.enabled = false(nQ, nP);
A = ddma.internal.abstraction_rows(A, S, (1:nQ)');
end
