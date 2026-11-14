function A = abstraction_rows(A, S, qidx)
%ABSTRACTION_ROWS Recompute the transitions of the state cells qidx (1-based, column)
%   for all inputs, from the map S and the options stored in A.opts.
Xedges = A.Xedges; Uedges = A.Uedges;
shX = A.shapeX; shU = A.shapeU;
nx = numel(Xedges); nP = prod(shU);
g_bounds = A.opts.g_bounds; features = A.opts.features; assumption = A.opts.assumption;
nQ = numel(qidx);
if nQ == 0, return; end
qs = ddma.all_subs_of(shX, qidx); ps = ddma.all_subs(shU);
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
    A.first(qidx, p, :) = reshape(int32(F), nQ, 1, nx);
    A.last(qidx, p, :) = reshape(int32(L), nQ, 1, nx);
    A.enabled(qidx, p) = ok;
end
end
