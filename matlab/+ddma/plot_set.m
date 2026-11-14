function h = plot_set(edges, mask, color)
%PLOT_SET Draw a set of cells (controllable / invariant set), TAC Figs. 11-12, ADHS Fig. 6.
%   ddma.plot_set(edges, mask) with a 3-D partition draws the isosurface of the cell
%   indicator (cell centres as coordinates); with a 2-D partition, filled cells.
%   mask: logical, one entry per cell (column-major order or array of the grid shape).
%   An empty mask draws the text 'empty set'. Rendered in MATLAB R2025 (MATLAB Online);
%   not render-tested in CI (Octave in the CI image has no working graphics).
if nargin < 3, color = [0.16 0.62 0.56]; end
shape = ddma.grid_shape(edges);
M = reshape(logical(mask), [shape, 1]);
c = cellfun(@(e) 0.5 * (e(1:end-1) + e(2:end)), edges(:)', 'UniformOutput', false);
if numel(shape) == 3
    pad = @(v) [2 * v(1) - v(2); v(:); 2 * v(end) - v(end-1)];
    V = zeros(shape + 2); V(2:end-1, 2:end-1, 2:end-1) = M;
    view(3); grid on
    xlim([edges{1}(1) edges{1}(end)]); ylim([edges{2}(1) edges{2}(end)]);
    zlim([edges{3}(1) edges{3}(end)]);
    if ~any(M(:))
        h = text(mean(xlim), mean(ylim), mean(zlim), 'empty set', ...
                 'HorizontalAlignment', 'center', 'Color', [0.35 0.35 0.35]);
        return
    end
    % isosurface(x, y, z, V) expects V(row = y, col = x, page = z); here rows are x1 and
    % columns x2, so pass x = x2, y = x1 and swap the vertex columns back afterwards.
    fv = isosurface(pad(c{2}), pad(c{1}), pad(c{3}), V, 0.5);
    fv.vertices = fv.vertices(:, [2 1 3]);
    % Flat shading with a floor of 45%, as in the Python plot_set_3d: no wall goes black,
    % whatever the view (MATLAB's default camlight leaves faces turned away almost black).
    L = [0.3 -0.5 0.8]; L = L / norm(L);
    t = fv.faces; P = fv.vertices;
    n = cross(P(t(:, 2), :) - P(t(:, 1), :), P(t(:, 3), :) - P(t(:, 1), :), 2);
    n = n ./ max(sqrt(sum(n .^ 2, 2)), 1e-12);
    shade = 0.45 + 0.55 * abs(n * L');
    h = patch('Faces', t, 'Vertices', P, 'FaceVertexCData', min(shade * color, 1), ...
              'FaceColor', 'flat', 'EdgeColor', 'none');
elseif numel(shape) == 2
    [I, J] = find(M);
    if isempty(I)
        axis([edges{1}(1) edges{1}(end) edges{2}(1) edges{2}(end)]);
        h = text(mean(xlim), mean(ylim), 'empty set', 'HorizontalAlignment', 'center', ...
                 'Color', [0.35 0.35 0.35]);
        return
    end
    e1 = edges{1}; e2 = edges{2};
    xs = [e1(I) e1(I + 1) e1(I + 1) e1(I)]'; ys = [e2(J) e2(J) e2(J + 1) e2(J + 1)]';
    h = patch(xs, ys, color, 'EdgeColor', 'none');
    axis([e1(1) e1(end) e2(1) e2(end)]);
else
    error('ddma:plot_set', 'only 2-D and 3-D partitions');
end
end
