function [lo, hi] = msm_bounds(Z, Y, d_lo, d_hi, zq, signs)
%MSM_BOUNDS Minimal simulating map (TAC paper, Theorem 1) at query points zq.
%   [lo, hi] = ddma.msm_bounds(Z, Y, d_lo, d_hi, zq) with data Z (K x n), Y (K x m),
%   disturbance bounds d_lo, d_hi (1 x m or scalars), queries zq (M x n).
%   Optional signs (m x n, entries -1/0/+1): monotonicity pattern of each output.
[K, n] = size(Z); m = size(Y, 2);
if nargin < 6 || isempty(signs), signs = ones(m, n); end
theta = (d_hi(:) - d_lo(:))' .* ones(1, m);
M = size(zq, 1);
lo = -inf(M, m); hi = inf(M, m);
for a = 1:M
    D = Z - zq(a, :);                % z_k - z
    for i = 1:m
        sd = D .* signs(i, :);
        above = all(sd >= 0, 2);
        below = all(sd <= 0, 2);
        if any(above), hi(a, i) = min(Y(above, i)) + theta(i); end
        if any(below), lo(a, i) = max(Y(below, i)) - theta(i); end
    end
end
end
