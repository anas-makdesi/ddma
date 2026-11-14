%ONLINE_REFINEMENT Learn while running: the safe region grows as data come in (MATLAB/Octave).
%   A damped pendulum near the bottom (theta+ = theta + T omega known; omega+ unknown,
%   with loose derivative bounds). Start from data collected near the bottom only, run
%   the certified controller with random allowed inputs (it explores the edge of the
%   safe region), and refine the model with every measurement:
%     ddma.bd_update_local   each step (cells around the new point; cheap)
%     ddma.bd_update         every 200 steps with the batch since the last one (exact)
%     ddma.abstraction_refresh + ddma.safety(..., known_winning)   after each refinement
%   The safe set only grows (about 50 % -> 80 % of the specification in 1000 steps), and
%   the closed loop never leaves the specification. Same functions as the Python
%   BoundedDerivativeModel.update / update_local, SymbolicAbstraction.refresh and
%   safety(known_winning=). Settings: nc (cells per state axis), K, r0 (set them first).
%   Run from the matlab/ folder:  run examples/online_refinement
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));   % the +ddma package
T = 0.1; D = 0.005;
f = @(X, U) T * (-sin(X(:, 1)) - 0.2 * X(:, 2) + U(:, 1));     % unknown (data only)
A_G = [1 T; 0 1];                                              % known part g(x) = A_G x
step = @(x, u) x * A_G' + [0, f(x, u) + D * (2 * rand - 1)];
if ~exist('nc', 'var'), nc = 60; end
Xedges = ddma.uniform_partition([-1 -1.5], [1 1.5], [nc nc]);
Uedges = ddma.uniform_partition(-2, 2, 7);
edges = [Xedges(:); Uedges(:)]';
% loose derivative bounds of f = (0, f_omega) on (theta, omega, u)
g_lo = [0 0 0; -0.3, -0.1, 0.05];
g_hi = [0 0 0;  0.1,  0.05, 0.15];
safe = ddma.cells_inside(Xedges, [-0.8 -1.2], [0.8 1.2]);

% initial model from exploration data near the bottom only (|theta| <= r0, |omega| <= 1.5 r0)
ddma.seed(1);
if ~exist('K', 'var'), K = 2000; end
if ~exist('r0', 'var'), r0 = 0.15; end
X = [r0 * (2 * rand(K, 1) - 1), 1.5 * r0 * (2 * rand(K, 1) - 1)]; U = 4 * rand(K, 1) - 2;
Y = [zeros(K, 1), f(X, U) + D * (2 * rand(K, 1) - 1)];
S = ddma.bd_fit([X U], Y, edges, g_lo, g_hi, [0 -D], [0 D], 'growth', 'slab');
A = ddma.abstraction(Xedges, Uedges, S, ddma.affine_bounds(A_G));
[domain, allowed] = ddma.safety(A, safe);
fprintf('initial model (%d samples): safe region %.1f %% of the specification\n', K, ...
        100 * nnz(domain) / nnz(safe));
if ~any(domain), error('empty initial safe set: use more initial data (K, r0)'); end

% run, learn, refine
dom = find(domain);
[lo, hi] = ddma.cell_bounds(Xedges, ddma.all_subs_of(A.shapeX, dom(randi(numel(dom)))));
x = lo + rand(1, 2) .* (hi - lo);
batch = zeros(0, 5); viol = 0; t_ref = 0;
for k = 1:1000
    u = ddma.controller_input(A, allowed, x, 'random');
    if isempty(u), viol = viol + 1; break; end
    xn = step(x, u);
    y = [0, xn(2) - x * A_G(2, :)'];                          % measurement of f + d
    S = ddma.bd_update_local(S, [x u], y, 1);
    batch(end + 1, :) = [x u y];
    if any(xn < [-0.8 -1.2] | xn > [0.8 1.2]), viol = viol + 1; end
    x = xn;
    if mod(k, 200) == 0
        tic;
        S = ddma.bd_update(S, batch(:, 1:3), batch(:, 4:5)); batch = zeros(0, 5);
        [A, S] = ddma.abstraction_refresh(A, S, true(numel(safe), 1));
        [domain, allowed] = ddma.safety(A, safe, [], domain);
        t_ref = t_ref + toc;
        fprintf('step %4d: safe region %.1f %% of the specification\n', k, 100 * nnz(domain) / nnz(safe));
    elseif mod(k, 20) == 0
        tic;
        [A, S] = ddma.abstraction_refresh(A, S);            % cells changed by update_local
        [domain, allowed] = ddma.safety(A, safe, [], domain);
        t_ref = t_ref + toc;
    end
end
fprintf('%d steps, %d violations; refinement time %.2f s in total\n', k, viol, t_ref);
