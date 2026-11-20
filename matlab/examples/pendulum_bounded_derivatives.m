%PENDULUM_BOUNDED_DERIVATIVES Safety for a non-monotone system learned from data (MATLAB/Octave).
%   Same problem as examples/pendulum_bounded_derivatives.py, on a coarser grid:
%     theta+ = theta + T omega                                (known)
%     omega+ = omega + T (a sin(theta) - b omega + c u) + d    (unknown, learned)
%   On theta in [-2, 2] the unknown part is not monotone in theta (cos theta changes
%   sign), but its derivatives are bounded:
%     d/dtheta in [T a cos 2, T a],  d/domega = -T b,  d/du = T c,
%   which is all ddma.bd_fit needs. Each model gives a maximal controlled invariant set
%   inside [-2, 2] x [-4, 4] (|u| <= 6); each controller then runs on the true pendulum.
%   Run from the matlab/ folder:  run examples/pendulum_bounded_derivatives
%   (set `methods` before running to choose the models, e.g. add 'lin_growth'; the LP
%   methods solve one small LP per cell and are slow in MATLAB/Octave).
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));   % the +ddma package
T = 0.05; a = 9.81; b = 0.5; c = 1.0; D = 0.01;
Xlo = [-2 -4]; Xhi = [2 4]; Umax = 6;
f = @(X, U) T * (a * sin(X(:, 1)) - b * X(:, 2) + c * U(:, 1));   % unknown (data only)
A_G = [1 T; 0 1];                                                    % known part g(x) = A_G x
if ~exist('methods', 'var'), methods = {'two_matrix', 'all_matrix', 'growth'}; end

% 1. data
ddma.seed(0);
K = 300000;
X = Xlo + rand(K, 2) .* (Xhi - Xlo);
U = Umax * (2 * rand(K, 1) - 1);
Xn = X * A_G' + [zeros(K, 1), f(X, U) + D * (2 * rand(K, 1) - 1)];
Y = Xn - X * A_G';                                                   % y = x' - g(x)

% 2. partition of (theta, omega, u) and derivative bounds of f = (0, f_omega)
Xedges = ddma.uniform_partition(Xlo, Xhi, [61 61]);
Uedges = ddma.uniform_partition(-Umax, Umax, 13);
edges = [Xedges(:); Uedges(:)]';
g_lo = [0 0 0; T * a * cos(2), -T * b, T * c];
g_hi = [0 0 0; T * a,           -T * b, T * c];
safe = true(prod(ddma.grid_shape(Xedges)), 1);
fprintf('%d transitions, %d state cells x %d inputs\n', K, numel(safe), numel(Uedges{1}) - 1);
fprintf('%-12s %9s %10s %10s %12s  %s\n', 'method', 'model [s]', 'abstr. [s]', 'safety [s]', ...
        'invariant', 'closed loop');
for k = 1:numel(methods)
    m = methods{k};
    loc = ''; if ~isempty(strfind(m, 'growth')), loc = 'slab'; end
    tic; S = ddma.bd_fit([X U], Y, edges, g_lo, g_hi, [0 -D], [0 D], m, loc); t1 = toc;
    tic; A = ddma.abstraction(Xedges, Uedges, S, ddma.affine_bounds(A_G)); t2 = toc;
    tic; [domain, allowed] = ddma.safety(A, safe); t3 = toc;
    % closed loop: 50 runs of 200 steps from random states of the invariant set,
    % random allowed inputs, on the true pendulum
    viol = 0; dom = find(domain);
    for r = 1:50
        [lo, hi] = ddma.cell_bounds(Xedges, ddma.all_subs_of(A.shapeX, dom(ddma.internal.rand_int(1, numel(dom)))));
        x = lo + rand(1, 2) .* (hi - lo);
        for s = 1:200
            u = ddma.controller_input(A, allowed, x, 'random');
            if isempty(u), viol = viol + 1; break; end
            x = x * A_G' + [0, f(x, u) + D * (2 * rand - 1)];
            if any(x < Xlo | x > Xhi), viol = viol + 1; break; end
        end
    end
    fprintf('%-12s %9.2f %10.2f %10.2f %11.1f%%  %d violations / 50 runs\n', m, t1, t2, t3, ...
            100 * mean(domain), viol);
end
