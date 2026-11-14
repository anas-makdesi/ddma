function h = plot_bounds(S, output, trueFun, trim)
%PLOT_BOUNDS Transparent staircase bounds of a 2-input map S (from ddma.fit), TAC Fig. 8.
%   ddma.plot_bounds(S) draws the upper and lower bounds of output 1 as transparent
%   piecewise-constant surfaces (one flat top per cell, as kron(.., ones(2)) in the
%   original scripts). ddma.plot_bounds(S, i, trueFun) also draws trueFun(Z) (Z: K x 2)
%   as a solid surface. trim (default 1) drops that many outer cells on each side.
%   (Not render-tested in CI: Octave in the CI image has no working graphics.)
if nargin < 2 || isempty(output), output = 1; end
if nargin < 3, trueFun = []; end
if nargin < 4, trim = 1; end
if numel(S.shape) ~= 2, error('ddma:plot_bounds', 'S must have two inputs'); end
ex = S.edges{1}(:); ey = S.edges{2}(:);
ex = ex(1 + trim:end - trim); ey = ey(1 + trim:end - trim);
lo = reshape(S.lower(:, output), S.shape); hi = reshape(S.upper(:, output), S.shape);
lo = lo(1 + trim:end - trim, 1 + trim:end - trim);
hi = hi(1 + trim:end - trim, 1 + trim:end - trim);
epsx = 1e-9 * (ex(end) - ex(1)); epsy = 1e-9 * (ey(end) - ey(1));
xs = sort([ex(1:end-1); ex(2:end) - epsx]); ys = sort([ey(1:end-1); ey(2:end) - epsy]);
[Xs, Ys] = ndgrid(xs, ys);
h = gobjects_compat(3);
holdState = ishold; hold on
if ~isempty(trueFun)
    [Gx, Gy] = ndgrid(linspace(ex(1), ex(end), 80), linspace(ey(1), ey(end), 80));
    Gz = reshape(trueFun([Gx(:) Gy(:)]), size(Gx));
    h(1) = surf(Gx, Gy, Gz, 'EdgeColor', 'none');
end
h(2) = mesh(Xs, Ys, kron(lo, ones(2)), 'FaceAlpha', 0.3, 'EdgeAlpha', 0.3);
h(3) = mesh(Xs, Ys, kron(hi, ones(2)), 'FaceAlpha', 0.3, 'EdgeAlpha', 0.3);
if ~holdState, hold off; end
view(3); grid on
xlabel('z_1'); ylabel('z_2'); zlabel('\sigma \circ \phi');
end

function h = gobjects_compat(n)
if exist('gobjects', 'builtin') || exist('gobjects', 'file'), h = gobjects(n, 1); else, h = zeros(n, 1); end
end
