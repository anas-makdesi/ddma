%BOUNDED_DERIVATIVE_MAP Models of a scalar map with bounded derivative (not monotone).
%   Same as examples/bounded_derivative_map.py: F(z) = sin(z) + 0.3 z + [-0.1, 0.1] on
%   [0, 2 pi], dF/dz in [-0.7, 1.3], 25 samples on [1, 5.5], 12 cells: the two-matrix
%   model (approach 1) and the growth-cone model with LP final step (approach 5).
%   Run from the matlab/ folder:  run examples/bounded_derivative_map   (plots in MATLAB)
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));   % the +ddma package
ddma.seed(1);
D = 0.1;
f = @(z) sin(z) + 0.3 * z;                       % unknown (data and plot only)
Z = 1 + 4.5 * rand(25, 1); Y = f(Z) + D * (2 * rand(25, 1) - 1);
edges = ddma.uniform_partition(0, 2 * pi, 12);
zs = linspace(0, 2 * pi, 1201)';
methods = {'two_matrix', 'lin_growth'};
for k = 1:2
    S = ddma.bd_fit(Z, Y, edges, -0.7, 1.3, -D, D, methods{k});
    [lo, hi] = ddma.bounds(S, zs);
    w = hi - lo;
    fprintf('%-11s bounded on %.0f %% of [0, 2 pi], mean width where bounded %.3f\n', ...
            methods{k}, 100 * mean(isfinite(w)), mean(w(isfinite(w))));
    B{k} = [lo, hi]; %#ok<SAGROW>
end
if ~exist('OCTAVE_VERSION', 'builtin')
    figure; hold on
    c = {[0.95 0.76 0.31], [0.16 0.62 0.56]};
    for k = 1:2
        stairs(zs, min(max(B{k}, -2.6), 3), 'Color', c{k});
    end
    plot(zs, f(zs) - D, 'k--', zs, f(zs) + D, 'k--'); plot(Z, Y, 'o');
    xlabel('z'); ylabel('y'); ylim([-2.6 3]);
    title('A map that is not monotone, from 25 samples and bounds on its derivative');
end
