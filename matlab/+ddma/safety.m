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
%
%   With a lazy abstraction (ddma.abstraction(..., lazy=true)) transitions are evaluated
%   on demand by the pointer scan, and allowed is returned as a struct (lazy, domain,
%   sat, n_evaluated = number of transitions evaluated) from which ddma.controller_input computes the allowed inputs of a state.
if nargin < 3 || isempty(backend), backend = 'auto'; end
if nargin < 4, known_winning = []; end
if nargin < 5 || isempty(check), check = true; end
safe = logical(safe(:));
nQ = prod(A.shapeX);
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
if isfield(A, 'lazy') && A.lazy
    [domain, allowed] = safety_lazy(A, safe, fixed);
    return
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

function [W, allowed] = safety_lazy(A, safe, fixed)
% Pointer scan on a lazy abstraction (same algorithm as the Python _safety_lazy): each
% state keeps its current candidate input and that input's successor box; a failed
% candidate moves the pointer forward, evaluating 1, 2, 4, ... transitions at a time.
shX = A.shapeX; nq = prod(shX); np = prod(A.shapeU); n = numel(shX);
W = safe;
if isempty(fixed), test = true(nq, 1); else, test = ~fixed; end
ptr = zeros(nq, 1);                 % 0-based candidate input
wf = zeros(nq, n); wl = zeros(nq, n);
known = false(nq, 1);
nev = 0;
for it = 1:100000
    sat = ddma.internal.sat_build(shX, W);
    rows = find(W & known & test);
    bad = ddma.internal.sat_count(sat, shX, wf(rows, :), wl(rows, :)) ~= 0;
    failed = rows(bad);
    ptr(failed) = ptr(failed) + 1; known(failed) = false;
    todo = find(W & ~known & test & ptr < np);
    block = 1;
    while ~isempty(todo)
        P = ptr(todo) + (0:block - 1);
        valid = P < np;
        Rr = repmat(todo, 1, block);
        [ok, f, la] = ddma.internal.pair_boxes(A, A.S, Rr(valid), P(valid) + 1);
        nev = nev + nnz(valid);
        gv = ok;
        gv(ok) = ddma.internal.sat_count(sat, shX, f(ok, :), la(ok, :)) == 0;
        g = false(size(P)); g(valid) = gv;
        found = any(g, 2);
        [~, j] = max(g(found, :), [], 2);
        hit = todo(found);
        ptr(hit) = ptr(hit) + j - 1;
        where = zeros(size(P)); where(valid) = 1:nnz(valid);
        fr = find(found);
        sel = where(sub2ind(size(P), fr(:), j(:)));
        wf(hit, :) = f(sel, :); wl(hit, :) = la(sel, :);
        known(hit) = true;
        miss = todo(~found);
        ptr(miss) = min(ptr(miss) + block, np);
        todo = miss(ptr(miss) < np);
        block = min(2 * block, 64);
    end
    dead = find(W & test & ptr >= np);
    if isempty(dead), break; end
    W(dead) = false;
end
allowed = struct('lazy', true, 'domain', W, 'sat', ddma.internal.sat_build(shX, W), ...
                 'n_evaluated', nev);
end

function tf = use_mex(backend)
tf = strcmp(backend, 'mex') || (strcmp(backend, 'auto') && exist('ddma_synth_mex', 'file') == 3);
end
