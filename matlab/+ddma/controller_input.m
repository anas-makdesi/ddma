function u = controller_input(A, allowed, x, pick)
%CONTROLLER_INPUT A concrete input at state x: point of an allowed input cell.
%   pick: 'center' (default, first allowed cell), 'random', or a desired input vector
%   (safety filter: nearest admissible point). Returns [] outside the domain.
%   allowed: the matrix from ddma.safety / ddma.reach_avoid, or the struct that
%   ddma.safety returns for a lazy abstraction.
if nargin < 4, pick = 'center'; end
Xlo = cellfun(@(e) e(1), A.Xedges(:)'); Xhi = cellfun(@(e) e(end), A.Xedges(:)');
u = [];
if any(x(:)' < Xlo | x(:)' > Xhi), return; end
q = ddma.flat_index(A.shapeX, ddma.locate(A.Xedges, x(:)'));
if isstruct(allowed)   % lazy abstraction: allowed inputs of q computed on demand
    nP = prod(A.shapeU); ps = [];
    if allowed.domain(q)
        [ok, F, L] = ddma.internal.pair_boxes(A, A.S, q * ones(nP, 1), (1:nP)');
        row = ok;
        row(ok) = ddma.internal.sat_count(allowed.sat, A.shapeX, F(ok, :), L(ok, :)) == 0;
        ps = find(row);
    end
else
    ps = find(allowed(q, :));
end
if isempty(ps), return; end
[lo, hi] = ddma.cell_bounds(A.Uedges, ddma.all_subs_of(A.shapeU, ps));
if ischar(pick) && strcmp(pick, 'random')
    k = randi(numel(ps)); u = lo(k, :) + rand(1, size(lo, 2)) .* (hi(k, :) - lo(k, :));
elseif ischar(pick)
    u = 0.5 * (lo(1, :) + hi(1, :));
else
    cand = min(max(pick(:)', lo), hi - 1e-9 * (hi - lo));
    [~, k] = min(sum((cand - pick(:)').^2, 2));
    u = cand(k, :);
end
end
