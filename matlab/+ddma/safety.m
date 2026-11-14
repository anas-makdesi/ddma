function [domain, allowed] = safety(A, safe, backend, known_winning, check)
%SAFETY Maximal controlled invariant subset of the safe cells (greatest fixed point).
%   [domain, allowed] = ddma.safety(A, safe): safe is a logical nQ x 1 mask;
%   domain (nQ x 1) is the winning set, allowed(q, p) the safe inputs at q.
%   backend: 'auto' (MEX if compiled, see build_mex) | 'mex' | 'matlab'. Same result.
%   The MEX kernel keeps, for each state, a pointer to its first input not yet ruled
%   out; the set only shrinks, so each (state, input) pair is examined at most once.
%
%   ddma.safety(A, safe, backend, known_winning): incremental update. known_winning is a
%   set of cells known to be controlled invariant inside safe, typically the previous
%   winning set after the model was refined (ddma.bd_update / bd_update_local, then
%   ddma.abstraction_refresh): refined successor boxes only shrink, so the old set stays
%   controlled invariant and the new maximal set contains it. These cells are kept and
%   not tested; the result is the same maximal set, and it only grows during online
%   refinement. With check = true (default) the assumption is verified first and an
%   error is raised if it does not hold. Same as the Python safety(known_winning=).
if nargin < 3 || isempty(backend), backend = 'auto'; end
if nargin < 4, known_winning = []; end
if nargin < 5 || isempty(check), check = true; end
safe = logical(safe(:));
nQ = size(A.enabled, 1);
if numel(safe) ~= nQ, error('ddma:safety', 'safe must have one entry per state cell'); end
fixed = [];
if ~isempty(known_winning)
    fixed = logical(known_winning(:));
    if numel(fixed) ~= nQ
        error('ddma:safety', 'known_winning must have one entry per state cell');
    end
    if any(fixed & ~safe), error('ddma:safety', 'known_winning must be a subset of safe'); end
    if check && any(fixed)
        inner = ddma.safety(A, fixed, backend);
        if ~isequal(inner, fixed)
            error('ddma:safety', 'known_winning is not controlled invariant for this abstraction');
        end
    end
    if ~any(fixed), fixed = []; end
end
if use_mex(backend)
    if isempty(fixed)
        [domain, allowed] = ddma_synth_mex('safety', A.first, A.last, A.enabled, A.shapeX, safe);
    else
        [domain, allowed] = ddma_synth_mex('safety', A.first, A.last, A.enabled, A.shapeX, safe, fixed);
    end
    return
end
W = safe;
while true
    good = ddma.box_all_in(A, W, 'matlab');
    Wn = W & any(good, 2);
    if ~isempty(fixed), Wn = Wn | fixed; end
    if isequal(Wn, W), break; end
    W = Wn;
end
domain = W;
allowed = good & W;
end

function tf = use_mex(backend)
tf = strcmp(backend, 'mex') || (strcmp(backend, 'auto') && exist('ddma_synth_mex', 'file') == 3);
end
