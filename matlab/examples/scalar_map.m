%SCALAR_MAP Minimal simulating map of a scalar monotone map, exact and on a partition.
%   Same as examples/scalar_map.py: F(z) = z^2 + [-0.5, 0.5] on [0, 3], 25 samples.
%   Run from the matlab/ folder:  run examples/scalar_map   (plots in MATLAB)
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));   % the +ddma package
ddma.seed(0);
f = @(z) z .^ 2;                                 % unknown (data and plot only)
Z = 3 * rand(25, 1); Y = f(Z) + (rand(25, 1) - 0.5);
zs = linspace(0, 3, 1201)';
[lo, hi] = ddma.msm_bounds(Z, Y, -0.5, 0.5, zs);            % exact minimal map
edges = ddma.uniform_partition(0, 3, 8);
S = ddma.fit(Z, Y, edges, -0.5, 0.5);                       % on 8 cells
[clo, chi] = ddma.bounds(S, zs);
mid = zs > 0.5 & zs < 2.5;
w = ddma.widths(S);
fprintf('25 samples; mean width of the exact map on [0.5, 2.5]: %.3f\n', mean(hi(mid) - lo(mid)));
fprintf('mean width on the 8-cell partition (bounded cells): %.3f\n', mean(w(isfinite(w))));
if ~exist('OCTAVE_VERSION', 'builtin')
    figure; hold on
    stairs(zs, min(max(chi, -1.5), 10.5), 'Color', [0.95 0.76 0.31]);
    stairs(zs, min(max(clo, -1.5), 10.5), 'Color', [0.95 0.76 0.31]);
    stairs(zs, min(max(hi, -1.5), 10.5), 'Color', [0.16 0.62 0.56]);
    stairs(zs, min(max(lo, -1.5), 10.5), 'Color', [0.16 0.62 0.56]);
    plot(zs, f(zs) - 0.5, 'k--', zs, f(zs) + 0.5, 'k--'); plot(Z, Y, 'o');
    xlabel('z'); ylabel('y'); ylim([-1.5 10.5]);
    title('Tightest model consistent with 25 noisy samples of an unknown monotone map');
end
