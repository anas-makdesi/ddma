function A = abstraction_rows(A, S, qidx)
%ABSTRACTION_ROWS Recompute the transitions of the state cells qidx (1-based, column)
%   for all inputs, from the map S and the options stored in A.opts.
nQ = numel(qidx);
if nQ == 0, return; end
nx = numel(A.Xedges); nP = prod(A.shapeU);
qidx = qidx(:);
for p = 1:nP
    [ok, F, L] = ddma.internal.pair_boxes(A, S, qidx, p * ones(nQ, 1));
    A.first(qidx, p, :) = reshape(int32(F), nQ, 1, nx);
    A.last(qidx, p, :) = reshape(int32(L), nQ, 1, nx);
    A.enabled(qidx, p) = ok;
end
end
