%PAPER_FIGURES Paper-style figures with the MATLAB version of ddma (saved as PNG files).
%   Run from the matlab/ folder:  run examples/paper_figures
%   Fig. 8 (TAC) is computed here from data; the 3-D sets (TAC Figs. 11-12, ADHS Fig. 6) are
%   loaded from figure_data.mat (computed by the Python reproductions; the full-scale Lorenz
%   set takes about 6 minutes and 4 GB, too long for a MATLAB Online basic session).
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..'));
outdir = fullfile(here, 'figures'); if ~exist(outdir, 'dir'), mkdir(outdir); end

% TAC Fig. 8: F(z) = 2 (sin z1 + sin z2 + z1 + z2) + w, |w| <= 0.1, 1e6 samples
F = @(Z) 2 * (sin(Z(:, 1)) + sin(Z(:, 2)) + Z(:, 1) + Z(:, 2));
ddma.seed(0);
K = 1e6; Z = -pi + 2 * pi * rand(K, 2); Y = F(Z) + 0.2 * rand(K, 1) - 0.1;
a = -pi + 0.01 * 2 * pi; b = pi - 0.01 * 2 * pi;
edges = ddma.margin_partition([-pi -pi], [pi pi], [a a], [b b], 29);   % 31 x 31 cells
tic; S = ddma.fit(Z, Y, edges, -0.1, 0.1); fprintf('fit: %.2f s\n', toc);
fig = figure('Position', [100 100 1100 450]);
subplot(1, 2, 1); ddma.plot_bounds(S, 1, F); view(-125, 25);
title('TAC Fig. 8: F (solid) and the transparent staircase bounds');
subplot(1, 2, 2);
Sz = S; % zoom on 5 x 5 cells around the origin
Sz.edges = {edges{1}(14:20), edges{2}(14:20)}; Sz.shape = [6 6];
lo = reshape(S.lower, S.shape); hi = reshape(S.upper, S.shape);
Sz.lower = reshape(lo(14:19, 14:19), [], 1); Sz.upper = reshape(hi(14:19, 14:19), [], 1);
ddma.plot_bounds(Sz, 1, F, 0); view(-115, 18); title('zoom on 6 x 6 cells');
exportgraphics_compat(fig, fullfile(outdir, 'tac_fig8_matlab.png'));

% 3-D sets
D = load(fullfile(here, 'figure_data.mat'));
sets = {'tac11', 'TAC Fig. 11: controllable set (contract (39))', {'d', 'v_f', 'v_l'}; ...
        'lorenz', 'TAC Fig. 12: Lorenz maximal controlled invariant (1e8 samples, 101^4 cells)', {'x', 'y', 'z'}; ...
        'adhs6', 'ADHS Fig. 6: controllable set (contract (8))', {'d', 'v_1', '-v_2'}};
for k = 1:size(sets, 1)
    e = D.([sets{k, 1} '_edges']); m = D.([sets{k, 1} '_domain']);
    fig = figure('Position', [100 100 650 520]);
    ddma.plot_set(e, m);
    xlabel(sets{k, 3}{1}); ylabel(sets{k, 3}{2}); zlabel(sets{k, 3}{3}); title(sets{k, 2});
    exportgraphics_compat(fig, fullfile(outdir, [sets{k, 1} '_matlab.png']));
end
fprintf('figures written to %s\n', outdir);

function exportgraphics_compat(fig, file)
if exist('exportgraphics', 'file'), exportgraphics(fig, file, 'Resolution', 150);
else, print(fig, file, '-dpng', '-r150'); end
end
