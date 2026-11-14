function [a_lo, a_hi, w] = estimate_derivative_bounds(Z, Y, n_lp_pairs, n_w_pairs, w_weight, pairs)
%ESTIMATE_DERIVATIVE_BOUNDS Derivative and disturbance bounds from data (TAC Sec. IV).
%   [a_lo, a_hi, w] = ddma.estimate_derivative_bounds(Z, Y) with data Y(k, :) in
%   f(Z(k, :)) + [-w, w]. Random pairs of samples are drawn; for each output i the linear
%   program (32) on n_lp_pairs pairs (default 50000) gives the derivative bounds
%   a_lo(i, :), a_hi(i, :), then (33) on n_w_pairs fresh pairs (default 1e6) gives w(i).
%   w_weight (default 1) multiplies the disturbance term of the LP cost (the original
%   experiments used 50); any value keeps the guarantee of Proposition 9, which concerns
%   only the second step. pairs (optional) fixes the pairs: struct with 1-based index
%   vectors a1, b1 (first step) and a2, b2 (second step).
%   a_lo, a_hi: m x n; w: 1 x m (disturbance interval [-w, w]). See
%   ddma.scenario_confidence for the probabilistic guarantee. Needs linprog or glpk.
%   Same as the Python estimate_derivative_bounds.
[K, n] = size(Z); m = size(Y, 2);
if nargin < 3 || isempty(n_lp_pairs), n_lp_pairs = 50000; end
if nargin < 4 || isempty(n_w_pairs), n_w_pairs = 1000000; end
if nargin < 5 || isempty(w_weight), w_weight = 1; end
if nargin < 6 || isempty(pairs)
    pairs = struct('a1', randi(K, n_lp_pairs, 1), 'b1', randi(K, n_lp_pairs, 1), ...
                   'a2', randi(K, n_w_pairs, 1), 'b2', randi(K, n_w_pairs, 1));
end
dz1 = Z(pairs.a1, :) - Z(pairs.b1, :); P1 = max(dz1, 0); N1 = min(dz1, 0);
dz2 = Z(pairs.a2, :) - Z(pairs.b2, :); P2 = max(dz2, 0); N2 = min(dz2, 0);
dY1 = Y(pairs.a1, :) - Y(pairs.b1, :); dY2 = Y(pairs.a2, :) - Y(pairs.b2, :);
n1 = size(dz1, 1);
% variables [a_lo (n), a_hi (n), w]: min sum_k (N_k a_lo + P_k a_hi) + 2 n1 w_weight w
% s.t. N_k a_lo + P_k a_hi + 2 w >= dY_k,  a_lo <= a_hi,  w >= 0
c = [sum(N1, 1), sum(P1, 1), 2 * n1 * w_weight]';
A = [-[N1, P1, 2 * ones(n1, 1)]; [eye(n), -eye(n), zeros(n, 1)]];
lb = [-inf(2 * n, 1); 0]; ub = inf(2 * n + 1, 1);
a_lo = zeros(m, n); a_hi = zeros(m, n); w = zeros(1, m);
for i = 1:m
    [x, ok] = ddma.internal.lp_solve(c, A, [-dY1(:, i); zeros(n, 1)], lb, ub);
    if ~ok, error('ddma:estimate_derivative_bounds', 'LP for output %d failed', i); end
    a_hi(i, :) = x(n + 1:2 * n)';
    a_lo(i, :) = min(x(1:n)', a_hi(i, :));   % a_lo <= a_hi up to solver tolerance
    slack = dY2(:, i) - P2 * a_hi(i, :)' - N2 * a_lo(i, :)';
    w(i) = max(0.5 * max(slack), 0);
end
end
