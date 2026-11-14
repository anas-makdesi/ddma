% ENVELOPE_NONLINEAR  Safe envelopes of non-monotone nonlinear systems from data and
% Jacobian bounds (MATLAB version of examples/envelope_nonlinear.py).
% Run from the matlab/ folder:  run examples/envelope_nonlinear
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));   % the +ddma package
LO = 0.6; HI = 1.1; UMAX = 0.02; W = 0.005;
f = @(x, u) [x(:, 1) .* (2 - x(:, 1) - 0.2 * x(:, 2)) + u(:, 1), ...
             x(:, 2) .* (2 - x(:, 2) - 0.2 * x(:, 1)) + u(:, 2)];
rand('seed', 2); %#ok<RAND>
N = 400;
X = LO + (HI - LO) * rand(N, 2); U = UMAX * (2 * rand(N, 2) - 1);
Xn = f(X, U) + W * (2 * rand(N, 2) - 1);
jl = [2 - 2.2 * HI, -0.2 * HI; -0.2 * HI, 2 - 2.2 * HI];
jh = [2 - 2.2 * LO, -0.2 * LO; -0.2 * LO, 2 - 2.2 * LO];
tic;
B = ddma.embedding_envelope(X, Xn, U, jl, jh, eye(2), eye(2), ...
    struct('w_lo', -W, 'w_hi', W, 'x_lo', [LO LO], 'x_hi', [HI HI]));
t = toc;
S = LO + (HI - LO) * rand(100000, 2);
inside = ddma.envelope_contains(B, S);
fprintf('competition: %d pairs in %.2f s -> %d boxes, %.1f%% of the domain\n', ...
    B.n_pairs, t, size(B.lo, 1), 100 * mean(inside));
x = S(inside, :); x = x(1:min(200, end), :);
for step = 1:40
    u = zeros(size(x));
    for k = 1:size(x, 1), u(k, :) = ddma.envelope_input(B, x(k, :), [-UMAX -UMAX]); end
    x = f(x, u) + W * (2 * rand(size(x)) - 1);
    assert(all(ddma.envelope_contains(B, x)));
end
fprintf('closed loop, %d starts x 40 steps: 0 violations\n', size(x, 1));
% ten coupled units, autonomous
n = 10; N = 1000;
g = @(x) 0.5 * x + 0.1 * sin(x(:, [n 1:n-1])) + 0.1 * cos(x(:, [2:n 1]));
jl = zeros(n); jh = zeros(n);
for i = 1:n
    jl(i, i) = 0.5; jh(i, i) = 0.5;
    for j = [mod(i - 2, n) + 1, mod(i, n) + 1], jl(i, j) = -0.1; jh(i, j) = 0.1; end
end
X = 4 * rand(N, n) - 2;
tic; C = ddma.embedding_envelope(X, g(X), [], jl, jh, [], []); t = toc;
fprintf('10-D chain: %d pairs in %.2f s -> %d invariant boxes\n', C.n_pairs, t, size(C.lo, 1));
for k = 1:size(C.lo, 1)
    x = bsxfun(@plus, C.lo(k, :), bsxfun(@times, rand(500, n), C.hi(k, :) - C.lo(k, :)));
    for step = 1:30
        x = g(x);
        assert(all(all(bsxfun(@ge, x, C.lo(k, :)) & bsxfun(@le, x, C.hi(k, :)))));
    end
end
