%LAZY_SYNTHESIS Lazy abstraction: synthesis that evaluates only the transitions it needs.
%   Same problem as examples/lazy_synthesis.py. Three rooms in a row, a hot outside
%   (34 C) and one shared cooling input u in [0, 1]; unknown heat gains in [0, 0.15].
%   The dynamics are learned from data as a monotone map (increasing in the
%   temperatures, decreasing in u). Specification: every room stays at or below 26 C.
%   An eager abstraction stores a successor box for every (state cell, input cell) pair;
%   a lazy one (ddma.abstraction(..., true)) evaluates a transition only when the
%   safety scan reaches it: a few percent of the pairs, the same winning set, O(|Q|)
%   memory. Run from the matlab/ folder:  run examples/lazy_synthesis   (set K first
%   for another number of cells per axis; default 30)
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));   % the +ddma package
if ~exist('K', 'var'), K = 30; end
T_LO = 16; T_HI = 32; T_MAX = 26; T_OUT = 34; T_SUP = 14; TS = 0.25; W_HI = 0.15;
plant = @(T, u) T + TS * (0.05 * (T_OUT - T) + 0.3 * [T(:, 2) - T(:, 1), ...
        T(:, 1) + T(:, 3) - 2 * T(:, 2), T(:, 2) - T(:, 3)] - 0.6 * u .* (T - T_SUP));
ddma.seed(0);
N = 200000;
T = T_LO + (T_HI - T_LO) * rand(N, 3); U = rand(N, 1);
Tn = plant(T, U) + W_HI * rand(N, 3);
Xedges = ddma.uniform_partition([T_LO T_LO T_LO], [T_HI T_HI T_HI], K);
Uedges = ddma.uniform_partition(0, 1, 21);
signs = repmat([1 1 1 -1], 3, 1);                 % increasing in T, decreasing in u
S = ddma.fit([T U], Tn, [Xedges(:); Uedges(:)]', 0, 0, signs);
[~, chi] = ddma.cell_bounds(Xedges, ddma.all_subs(ddma.grid_shape(Xedges)));
safe = all(chi <= T_MAX, 2);
nPairs = numel(safe) * 21;
tic; A = ddma.abstraction(Xedges, Uedges, S); [d, al] = ddma.safety(A, safe); te = toc;
fprintf('eager: %5.2f s, transitions evaluated %9d of %d, winning %.1f %% (safe set %.1f %%)\n', ...
        te, nPairs, nPairs, 100 * mean(d), 100 * mean(safe));
tic; Al = ddma.abstraction(Xedges, Uedges, S, [], [], [], true); [dl, all_] = ddma.safety(Al, safe); tl = toc;
fprintf('lazy : %5.2f s, transitions evaluated %9d of %d, winning %.1f %%\n', ...
        tl, all_.n_evaluated, nPairs, 100 * mean(dl));
if ~isequal(d, dl), error('lazy and eager winning sets differ'); end
fprintf('same winning set\n');
x = [24 25 24.5];
u = ddma.controller_input(Al, all_, x);   % allowed inputs computed on demand
fprintf('at T = [%g %g %g]: cooling u = %.3f\n', x, u);
