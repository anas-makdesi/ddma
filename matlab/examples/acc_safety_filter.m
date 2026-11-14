%ACC_SAFETY_FILTER Safe adaptive cruise control learned from data (MATLAB/Octave).
%   Same problem as examples/adaptive_cruise_control.py: state (gap h, speed v), input u.
%   The speed dynamics are unknown (monotone in v and u); the leader speed vl in [13, 17]
%   acts as a disturbance. Learn the model from data, compute the maximal controlled
%   invariant set of h in [10, 50], v in [2, 28], and use it as a safety filter around a
%   naive cruise controller. Run from the matlab/ folder:  run examples/acc_safety_filter
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));   % the +ddma package
T = 0.5; M = 1370; f0 = 51; f1 = 1.2567; f2 = 0.4342;
VL = [13 17]; W = 0.05;
Xlo = [5 0]; Xhi = [60 30]; Ulo = -3; Uhi = 3;
SAFE_LO = [10 2]; SAFE_HI = [50 28];
speed = @(v, u) v + T * (u - (f0 + f1 * v + f2 * v.^2) / M);   % unknown (only used for data)
plant = @(x, u, vl) [x(1) - T * x(2) + T * vl, speed(x(2), u) + W * (2 * rand - 1)];
A_G = [1 -T; 0 0];                                                % known part g(x) = A_G x

% 1. data
ddma.seed(0);
K = 200000;
X = Xlo + rand(K, 2) .* (Xhi - Xlo);
U = Ulo + rand(K, 1) * (Uhi - Ulo);
Xn = [X(:, 1) - T * X(:, 2) + T * (VL(1) + rand(K, 1) * diff(VL)), ...
      speed(X(:, 2), U) + W * (2 * rand(K, 1) - 1)];
Y = Xn - X * A_G';                                                % y = x' - g(x, u)

% 2. model on the (v, u) grid, 3. abstraction, 4. safety
Xedges = ddma.uniform_partition(Xlo, Xhi, [110 60]);
Uedges = ddma.uniform_partition(Ulo, Uhi, 24);
features = [2 3];
Zall = [X U];
tic; S = ddma.fit(Zall(:, features), Y, {Xedges{2}, Uedges{1}}, [T * VL(1), -W], [T * VL(2), W]);
t_fit = toc;
tic; A = ddma.abstraction(Xedges, Uedges, S, ddma.affine_bounds(A_G), features); t_abs = toc;
safe = ddma.cells_inside(Xedges, SAFE_LO, SAFE_HI);
tic; [domain, allowed] = ddma.safety(A, safe); t_syn = toc;
fprintf('times: fit %.3f s, abstraction %.2f s, synthesis %.2f s\n', t_fit, t_abs, t_syn);
fprintf('certified safe region: %.1f%% of the specification cells\n', 100 * nnz(domain) / nnz(safe));

% 5. closed loop: naive cruise controller (hold 22 m/s) with and without the filter
N = 400; vl = 15 + 2 * sign(sin((0:N-1) * 2 * pi / 80)) .* (1 - 0.2 * rand(1, N));
nominal = @(x) min(max(0.3 * (22 - x(2)), Ulo), Uhi);
x = [30 15]; xr = x; crash = NaN; worst = Inf;
for k = 1:N
    u = ddma.controller_input(A, allowed, x, nominal(x));     % safety filter
    if isempty(u), error('left the controller domain'); end
    x = plant(x, u, vl(k)); worst = min(worst, x(1));
    if any(x < SAFE_LO | x > SAFE_HI), error('specification violated'); end
    if isnan(crash)
        xr = plant(xr, nominal(xr), vl(k));
        if xr(1) <= 0, crash = k * T; end
    end
end
fprintf('nominal controller alone: collision at t = %.1f s\n', crash);
fprintf('with safety filter: %d steps, minimum gap %.1f m, no violation\n', N, worst);
