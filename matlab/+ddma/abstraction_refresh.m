function [A, S, mask] = abstraction_refresh(A, S, states)
%ABSTRACTION_REFRESH Recompute the transitions of some states after the model was refined.
%   [A, S, mask] = ddma.abstraction_refresh(A, S) recomputes, for all inputs, the
%   transitions of the state cells whose coordinates lie in a box changed by
%   ddma.bd_update_local (S.changed, which is then cleared). Same result as rebuilding
%   the abstraction with ddma.abstraction (options from A.opts), at the cost of the
%   changed states only. mask (nQ x 1 logical): the recomputed states.
%   ddma.abstraction_refresh(A, S, states): recompute the given states instead (logical
%   mask or 1-based indices), e.g. all of them after ddma.bd_update.
%   After a refinement the successor boxes only shrink, so the previous winning set stays
%   controlled invariant: pass it to ddma.safety as known_winning.
%   Same as the Python SymbolicAbstraction.refresh.
nQ = size(A.enabled, 1);
nx = numel(A.Xedges);
if nargin < 3 || isempty(states)
    mask = false(nQ, 1);
    if isfield(S, 'changed') && ~isempty(S.changed)
        qs = ddma.all_subs(A.shapeX);
        feats = A.opts.features;
        nf = numel(feats);
        for b = 1:size(S.changed, 1)
            first = S.changed(b, 1:nf); last = S.changed(b, nf + 1:end);
            ok = true(nQ, 1);
            for k = 1:nf
                j = feats(k);
                if j <= nx, ok = ok & qs(:, j) >= first(k) & qs(:, j) <= last(k); end
            end
            mask = mask | ok;
        end
        S.changed = zeros(0, size(S.changed, 2));
    end
elseif islogical(states)
    mask = states(:);
else
    mask = false(nQ, 1); mask(states) = true;
end
A = ddma.internal.abstraction_rows(A, S, find(mask));
end
