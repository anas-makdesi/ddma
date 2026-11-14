function [M, keep] = maximal_points(P)
%MAXIMAL_POINTS Rows of P that no other row dominates (componentwise >=), duplicates once.
%   [M, keep] = ddma.maximal_points(P) returns M = P(keep, :). Sort-filter skyline: rows
%   sorted by decreasing sum can only be dominated by earlier rows. Same as the Python
%   ddma.invariance.maximal_points.
[R, d] = size(P);
if R == 0, M = P; keep = zeros(0, 1); return; end
[~, order] = sortrows([-sum(P, 2), -P]);
S = P(order, :);
front = zeros(0, d); keep = zeros(0, 1);
block = 256;
for s = 1:block:R
    e = min(R, s + block - 1);
    C = S(s:e, :); idx = (s:e)';
    if ~isempty(front)
        dom = false(size(C, 1), 1);
        for f0 = 1:4096:size(front, 1)
            F = front(f0:min(size(front, 1), f0 + 4095), :);
            ge = all(bsxfun(@ge, permute(F, [3 1 2]), permute(C, [1 3 2])), 3);
            dom = dom | any(ge, 2);
        end
        C = C(~dom, :); idx = idx(~dom);
    end
    if isempty(C), continue; end
    ge = all(bsxfun(@ge, permute(C, [3 1 2]), permute(C, [1 3 2])), 3);  % ge(i,j): C_j >= C_i
    ge = tril(ge, -1);
    ok = ~any(ge, 2);
    front = [front; C(ok, :)]; %#ok<AGROW>
    keep = [keep; idx(ok)]; %#ok<AGROW>
end
keep = order(keep);
M = P(keep, :);
end
