% ENVELOPE_ROOM_COOLING  Safe operating envelope from data, without a grid (MATLAB version
% of examples/envelope_room_cooling.py). Three rooms, one cooling input u in [0, 1],
% unknown heat gains in [0, 0.15]; the plant is monotone (increasing in the temperatures,
% decreasing in u). Specification: every room at or below 26 C.
% Run from the matlab/ folder:  run examples/envelope_room_cooling
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));   % the +ddma package
if ~exist('N', 'var'), N = 1000; end
plant = @(T, u) T + 0.25 * (0.05 * (34 - T) + 0.3 * [T(:, 2) - T(:, 1), ...
    T(:, 1) + T(:, 3) - 2 * T(:, 2), T(:, 2) - T(:, 3)] - 0.6 * bsxfun(@times, u, T - 14));
rand('seed', 1); %#ok<RAND>
X = 16 + 16 * rand(N, 3); U = rand(N, 1);
Xn = plant(X, U) + 0.15 * rand(N, 3);
tic;
E = ddma.monotone_envelope(X, Xn, U, struct('w_lo', 0, 'w_hi', 0.15, 'input_signs', -1, ...
    'x_hi', [26 26 26], 'u_lo', 0, 'u_hi', 1));
t = toc;
S = 16 + 10 * rand(100000, 3);
inside = ddma.envelope_contains(E, S);
fprintf('envelope: %.2f s, %d transitions, %d corners, certifies %.1f%% of the safe box\n', ...
    t, N, size(E.corners, 1), 100 * mean(inside));
% closed loop: the controller wants no cooling; the envelope gives the least safe cooling
x = S(inside, :); x = x(1:min(300, end), :);
for step = 1:100
    u = zeros(size(x, 1), 1);
    for k = 1:size(x, 1), u(k) = ddma.envelope_input(E, x(k, :), 0); end
    x = plant(x, u) + 0.15 * (rand(size(x)) > 0.5);
    assert(all(ddma.envelope_contains(E, x)) && all(x(:) <= 26 + 1e-9));
end
fprintf('closed loop, %d starts x 100 steps: 0 violations\n', size(x, 1));
