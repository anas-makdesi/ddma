function [lo, up] = bd_finalize(R, edges, cells, lp)
%BD_FINALIZE Cell bounds of one output from its stored maps, on the cells with 0-based
%   subscripts cells (k x n). Sign matrices: intersection of the interval hulls
%   (eqs. 3.8-3.9) or, with lp, the hull of the intersection (eqs. 3.6-3.7). Growth:
%   eqs. 3.26-3.27 from the vertex values, or with lp eqs. 3.24-3.25; then the local map.
shape = ddma.grid_shape(edges);
n = numel(edges);
[clo, chi] = ddma.cell_bounds(edges, cells);
f = ddma.flat_index(shape, cells);
if strcmp(R.kind, 'sign')
    a = R.a; slo = R.slo(f, :); shi = R.shi(f, :);
    ap = max(a, 0); am = min(a, 0);
    az_lo = clo * ap' + chi * am';   % interval of a . z over each cell
    az_hi = chi * ap' + clo * am';
    up = min(az_hi + shi, [], 2);
    lo = max(az_lo + slo, [], 2);
    if lp
        up = ddma.max_min_affine(shi, a, clo, chi, up);
        lo = -ddma.max_min_affine(-slo, -a, clo, chi, -lo);
    end
    return
end
gl = R.gl; gh = R.gh;
ne = shape + 1;
k = size(cells, 1);
h = chi - clo;
up = inf(k, 1); lo = -inf(k, 1);
nc = 2 ^ n;
if lp, offu = zeros(k, nc); offl = offu; SU = zeros(nc, n); SL = SU; end
for c = 0:nc-1
    b = bitand(c, 2 .^ (n-1:-1:0)) > 0;   % corner, itertools.product order
    v = ddma.flat_index(ne, cells + b);
    ru = sum(max(0, pick(b, -gl, gh) .* h), 2);
    rl = sum(min(0, pick(b, -gh, gl) .* h), 2);
    up = min(up, R.vup(v) + ru);
    lo = max(lo, R.vlo(v) + rl);
    if lp  % cone from the vertex, inside the cell
        su = pick(b, gl, gh); sl = pick(b, gh, gl);
        vc = clo + b .* h;
        offu(:, c + 1) = R.vup(v) - vc * su';
        offl(:, c + 1) = R.vlo(v) - vc * sl';
        SU(c + 1, :) = su; SL(c + 1, :) = sl;
    end
end
if lp
    up = ddma.max_min_affine(offu, SU, clo, chi, up);
    lo = -ddma.max_min_affine(-offl, -SL, clo, chi, -lo);
end
if isfield(R, 'lup')
    up = min(up, R.lup(f)); lo = max(lo, R.llo(f));
end
end

function r = pick(b, x, y)
r = y; r(b) = x(b);
end
