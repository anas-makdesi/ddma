function A = abstraction_rows(A, S, qidx)
%ABSTRACTION_ROWS Recompute the transitions of the state cells qidx (1-based, column)
%   for all inputs, from the map S and the options stored in A.opts.
nQ = numel(qidx);
if nQ == 0, return; end
nx = numel(A.Xedges); nP = prod(A.shapeU);
qidx = qidx(:);
if ddma.internal.abs_use_mex(A)
    g_bounds = A.opts.g_bounds;
    if isempty(g_bounds)
        [ok, F, L] = ddma.internal.abs_mex(A, S, qidx, (1:nP)', false);
        A.enabled(qidx, :) = ok; A.first(qidx, :, :) = F; A.last(qidx, :, :) = L;
        return
    end
    % g evaluated here, for blocks of inputs (about a million pairs per call)
    [xlo, xhi] = ddma.cell_bounds(A.Xedges, ddma.all_subs_of(A.shapeX, qidx));
    block = max(1, floor(1e6 / nQ));
    for p0 = 1:block:nP
        ps = (p0:min(nP, p0 + block - 1))';
        [ulo, uhi] = ddma.cell_bounds(A.Uedges, ddma.all_subs_of(A.shapeU, ps));
        r = repmat((1:nQ)', numel(ps), 1); c = kron((1:numel(ps))', ones(nQ, 1));
        [glo, ghi] = g_bounds(xlo(r, :), xhi(r, :), ulo(c, :), uhi(c, :));
        [ok, F, L] = ddma.internal.abs_mex(A, S, qidx, ps, false, glo, ghi);
        A.enabled(qidx, ps) = ok; A.first(qidx, ps, :) = F; A.last(qidx, ps, :) = L;
    end
    return
end
for p = 1:nP
    [ok, F, L] = ddma.internal.pair_boxes(A, S, qidx, p * ones(nQ, 1));
    A.first(qidx, p, :) = reshape(int32(F), nQ, 1, nx);
    A.last(qidx, p, :) = reshape(int32(L), nQ, 1, nx);
    A.enabled(qidx, p) = ok;
end
end
