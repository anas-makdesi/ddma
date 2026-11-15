function [ok, F, L] = abs_mex(A, S, q, p, paired, glo, ghi)
%ABS_MEX Transitions through the MEX kernel ddma_abs_mex (same result as the MATLAB code).
%   paired = true: the pairs (q(k), p(k)); ok k x 1, F/L k x nx.
%   paired = false: all pairs (q(i), p(c)); ok numel(q) x numel(p), F/L numel(q) x
%   numel(p) x nx (int32). q, p: 1-based state and input cells. glo, ghi: [] or bounds of
%   g per pair (rows in the order of the outputs, column-major).
%   Only the offsets of the cells asked for are computed, so lazy abstractions (many small
%   calls) do not pay for the whole grid.
nx = numel(A.Xedges);
q = q(:); p = p(:);
qs = ddma.all_subs_of(A.shapeX, q); ps = ddma.all_subs_of(A.shapeU, p);
features = A.opts.features;
if isfield(S, 'features') && ~isempty(S.features)
    G = repmat(struct('offq', [], 'offp', [], 'lo', [], 'hi', [], 'outs', []), 1, numel(S.raw));
    for i = 1:numel(S.raw)
        R = S.raw{i};
        [oq, op] = offsets(R.shape, features(R.feat), qs, ps, nx);
        G(i) = struct('offq', oq, 'offp', op, 'lo', R.lo(:), 'hi', R.up(:), 'outs', i - 1);
    end
else
    [oq, op] = offsets(S.shape, features, qs, ps, nx);
    G = struct('offq', oq, 'offp', op, 'lo', S.lower, 'hi', S.upper, 'outs', 0:size(S.lower, 2) - 1);
end
assumption = [];
a = A.opts.assumption;
if ~isempty(a)
    assumption = zeros(4, nx);
    for j = 1:nx
        e = A.Xedges{j};
        cLo = 0; cHi = A.shapeX(j) - 1;
        if isfinite(a(1, j)), cLo = ddma.searchsorted(e, a(1, j), 'left'); end
        if isfinite(a(2, j)), cHi = ddma.searchsorted(e, a(2, j), 'right') - 2; end
        assumption(:, j) = [a(1, j); a(2, j); cLo; cHi];
    end
end
if nargin < 6, glo = []; ghi = []; end
[ok, F, L] = ddma_abs_mex(A.Xedges(:)', G, (0:numel(q) - 1)', (0:numel(p) - 1)', paired, ...
                          glo, ghi, assumption, ddma.internal.mex_opts());
if paired
    F = reshape(double(F), numel(q), nx); L = reshape(double(L), numel(q), nx);
end
end

function [oq, op] = offsets(shape, coords, qs, ps, nx)
% 0-based offsets into a grid of the given shape (column-major) whose axes are the joint
% (x, u) coordinates coords: flat cell = oq(i) + op(c).
stride = [1, cumprod(shape(1:end - 1))];
oq = zeros(size(qs, 1), 1); op = zeros(size(ps, 1), 1);
for k = 1:numel(coords)
    c = coords(k);
    if c <= nx
        oq = oq + qs(:, c) * stride(k);
    else
        op = op + ps(:, c - nx) * stride(k);
    end
end
end
