function [domain, allowed] = safety(A, safe, backend)
%SAFETY Maximal controlled invariant subset of the safe cells (greatest fixed point).
%   [domain, allowed] = ddma.safety(A, safe): safe is a logical nQ x 1 mask;
%   domain (nQ x 1) is the winning set, allowed(q, p) the safe inputs at q.
%   backend: 'auto' (MEX if compiled, see build_mex) | 'mex' | 'matlab'. Same result.
%   The MEX kernel keeps, for each state, a pointer to its first input not yet ruled
%   out; the set only shrinks, so each (state, input) pair is examined at most once.
if nargin < 3, backend = 'auto'; end
safe = logical(safe(:));
if use_mex(backend)
    [domain, allowed] = ddma_synth_mex('safety', A.first, A.last, A.enabled, A.shapeX, safe);
    return
end
W = safe;
while true
    good = ddma.box_all_in(A, W, 'matlab');
    Wn = W & any(good, 2);
    if isequal(Wn, W), break; end
    W = Wn;
end
domain = W;
allowed = good & W;
end

function tf = use_mex(backend)
tf = strcmp(backend, 'mex') || (strcmp(backend, 'auto') && exist('ddma_synth_mex', 'file') == 3);
end
