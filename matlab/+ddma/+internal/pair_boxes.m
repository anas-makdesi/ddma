function [ok, F, L] = pair_boxes(A, S, q, p)
%PAIR_BOXES Successor boxes of state-input pairs (q(k), p(k)) (1-based column vectors):
%   ok(k) whether the pair is enabled, F/L (k x nx, 0-based) the first/last successor
%   cells. Shared by the eager abstraction (all pairs) and the lazy one (on demand).
Xedges = A.Xedges; Uedges = A.Uedges;
nx = numel(Xedges);
g_bounds = A.opts.g_bounds; features = A.opts.features; assumption = A.opts.assumption;
k = numel(q);
qs = ddma.all_subs_of(A.shapeX, q); ps = ddma.all_subs_of(A.shapeU, p);
[xlo, xhi] = ddma.cell_bounds(Xedges, qs);
[ulo, uhi] = ddma.cell_bounds(Uedges, ps);
Xlo = cellfun(@(e) e(1), Xedges(:)'); Xhi = cellfun(@(e) e(end), Xedges(:)');
sub = [qs, ps];
[lo, hi] = ddma.internal.map_bounds_at(S, sub(:, features));
if ~isempty(g_bounds)
    [glo, ghi] = g_bounds(xlo, xhi, ulo, uhi);
    lo = lo + glo; hi = hi + ghi;
end
if ~isempty(assumption)
    lo = min(max(lo, assumption(1, :)), assumption(2, :));
    hi = min(max(hi, assumption(1, :)), assumption(2, :));
end
ok = all(lo <= hi, 2) & all(isfinite(lo) & isfinite(hi), 2) & all(lo >= Xlo & hi <= Xhi, 2);
F = zeros(k, nx); L = F;
for j = 1:nx
    e = Xedges{j}; kk = numel(e) - 2;
    F(ok, j) = min(max(ddma.searchsorted(e, lo(ok, j), 'right') - 1, 0), kk);
    L(ok, j) = min(max(ddma.searchsorted(e, hi(ok, j), 'right') - 1, 0), kk);
    if ~isempty(assumption)
        cLo = 0; cHi = A.shapeX(j) - 1;
        if isfinite(assumption(1, j)), cLo = ddma.searchsorted(e, assumption(1, j), 'left'); end
        if isfinite(assumption(2, j)), cHi = ddma.searchsorted(e, assumption(2, j), 'right') - 2; end
        F(ok, j) = min(max(F(ok, j), cLo), cHi);
        L(ok, j) = min(max(L(ok, j), cLo), cHi);
    end
end
end
