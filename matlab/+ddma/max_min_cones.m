function val = max_min_cones(p, q, alpha, beta)
%MAX_MIN_CONES max over s in [0,1]^d of min(p + alpha*s, q + beta*(1 - s)), exactly.
%   val = ddma.max_min_cones(p, q, alpha, beta): p, q are R x 1, alpha, beta are R x d
%   (all >= 0); val is R x 1. By LP duality this is the minimum over lam in [0, 1] of a
%   convex piecewise-linear function; one sort of the breakpoints beta./(alpha+beta) finds
%   it. Same as the Python ddma.invariance.max_min_cones.
tot = alpha + beta;
lam = 2 * ones(size(tot));
pos = tot > 0;
lam(pos) = beta(pos) ./ tot(pos);
[lam_s, ord] = sort(lam, 2);
R = size(tot, 1);
rows = repmat((1:R)', 1, size(tot, 2));
tot_s = tot(sub2ind(size(tot), rows, ord));
s0 = p - q - sum(beta, 2);
slope = bsxfun(@plus, s0, cumsum(tot_s, 2));
nonneg = slope >= 0;
has = any(nonneg, 2);
[~, first] = max(nonneg, [], 2);
lam_star = lam_s(sub2ind(size(lam_s), (1:R)', first));
lam_star(~has) = 1;
lam_star(s0 >= 0) = 0;
lam_star = min(max(lam_star, 0), 1);
val = lam_star .* p + (1 - lam_star) .* (q + sum(beta, 2)) + ...
      sum(max(0, bsxfun(@times, lam_star, tot) - beta), 2);
end
