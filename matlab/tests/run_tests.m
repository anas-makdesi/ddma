function run_tests()
%RUN_TESTS Compare the MATLAB implementation with reference results from Python.
%   Run from the matlab/ folder (MATLAB or Octave):  addpath tests; run_tests
here = fileparts(mfilename('fullpath'));
R = load(fullfile(here, 'reference.mat'));
backends = {'matlab'};
if exist('ddma_fit_mex', 'file') == 3, backends{end + 1} = 'mex'; end
synth = {'matlab'};
if exist('ddma_synth_mex', 'file') == 3, synth{end + 1} = 'mex'; end
nfail = 0;

% 1. fixed-partition maps (both backends) and pointwise minimal simulating maps
fits = R.fits;
for k = 1:numel(fits)
    c = fits{k};
    edges = c.edges;
    sh = ddma.grid_shape(edges);
    for b = 1:numel(backends)
        S = ddma.fit(c.Z, c.Y, edges, c.d_lo, c.d_hi, c.signs, backends{b});
        lo = reshape(S.lower, [sh, size(c.Y, 2)]);
        hi = reshape(S.upper, [sh, size(c.Y, 2)]);
        nfail = nfail + check(isequal(lo(:), c.lower(:)) && isequal(hi(:), c.upper(:)), ...
            sprintf('fit case %d (%s)', k, backends{b}));
    end
    [mlo, mhi] = ddma.msm_bounds(c.Z, c.Y, c.d_lo, c.d_hi, c.zq, c.signs);
    nfail = nfail + check(isequal(mlo, c.msm_lo) && isequal(mhi, c.msm_hi), ...
        sprintf('msm case %d', k));
end

% 2. abstraction, safety and reach-avoid
a = R.abs;
Z = [a.X, a.U];
S = ddma.fit(Z(:, a.features), a.Y, a.fedges, a.d_lo, a.d_hi);
nfail = nfail + check(isequal(S.lower(:), a.lower(:)) && isequal(S.upper(:), a.upper(:)), 'abstraction map');
A = ddma.abstraction(a.Xedges, a.Uedges, S, ddma.affine_bounds(a.A_G), a.features, a.assumption);
shX = A.shapeX; shU = A.shapeU; nx = numel(shX);
first = reshape(A.first, [shX, shU, nx]); last = reshape(A.last, [shX, shU, nx]);
en = reshape(A.enabled, [shX, shU]);
ok = isequal(en, logical(a.enabled));
msk = repmat(en, [ones(1, numel(shX) + numel(shU)), nx]);
ok = ok && isequal(double(first(msk)), double(a.first(msk))) && isequal(double(last(msk)), double(a.last(msk)));
nfail = nfail + check(ok, 'abstraction transitions');
safe = ddma.cells_inside(a.Xedges, [-0.6, -1], [0.6, 1]);
[clo, chi] = ddma.cell_bounds(a.Xedges, ddma.all_subs(shX));
avoid = all(chi >= [-0.2, -1] & clo <= [0, 0], 2);
target = ddma.cells_inside(a.Xedges, [0.3, -0.4], [0.6, 0.4]);
for b = 1:numel(synth)
    [dom, alw] = ddma.safety(A, safe, synth{b});
    good = ddma.box_all_in(A, dom, synth{b});
    nfail = nfail + check(isequal(reshape(dom, shX), logical(a.safety_domain)) && ...
        isequal(alw, good & dom), sprintf('safety (%s)', synth{b}));
    [rdom, ralw, val] = ddma.reach_avoid(A, target, avoid, synth{b});
    if b == 1, ralw1 = ralw; end
    nfail = nfail + check(isequal(reshape(rdom, shX), logical(a.reach_domain)) && ...
        isequal(reshape(val, shX), a.reach_value) && isequal(ralw, ralw1), ...
        sprintf('reach-avoid (%s)', synth{b}));
end

% 3. MEX fixed points vs the MATLAB code on random 1-D to 3-D abstractions
if numel(synth) > 1
    ddma.seed(1);
    shapes = {12, [8 7], [6 5 4]};
    for k = 1:numel(shapes)
        ok = true;
        for t = 0:5
            R = random_abstraction(shapes{k}, 6, mod(t, 2), mod(t, 3));
            nQ = prod(R.shapeX);
            safe = rand(nQ, 1) < 0.95; tgt = rand(nQ, 1) < 0.15; av = rand(nQ, 1) < 0.1;
            [d1, a1] = ddma.safety(R, safe, 'matlab'); [d2, a2] = ddma.safety(R, safe, 'mex');
            [r1, b1, v1] = ddma.reach_avoid(R, tgt, av, 'matlab');
            [r2, b2, v2] = ddma.reach_avoid(R, tgt, av, 'mex');
            ok = ok && isequal(d1, d2) && isequal(a1, a2) && isequal(r1, r2) && ...
                 isequal(b1, b2) && isequal(v1, v2);
        end
        nfail = nfail + check(ok, sprintf('random %d-D abstractions (mex = matlab)', numel(shapes{k})));
    end
end

if nfail == 0
    fprintf('ALL MATLAB TESTS PASSED (%d backends)\n', numel(backends));
else
    error('run_tests:fail', '%d test(s) failed', nfail);
end
end

function f = check(ok, name)
if ok, fprintf('  ok    %s\n', name); f = 0; else, fprintf('  FAIL  %s\n', name); f = 1; end
end

function A = random_abstraction(shX, nP, noise, extent)
% random successor boxes: each input shifts the state by a random step, plus noise
nx = numel(shX); nQ = prod(shX);
sub = ddma.all_subs(shX);                           % 0-based subscripts, nQ x nx
first = zeros(nQ, nP, nx, 'int32'); last = first;
enabled = false(nQ, nP);
for p = 1:nP
    F = sub + randi([-1 1], 1, nx) - randi([0 noise], nQ, nx);
    L = F + randi([0 extent], nQ, nx);
    ok = all(F >= 0 & L <= shX - 1, 2) & rand(nQ, 1) < 0.9;
    F(~ok, :) = 0; L(~ok, :) = 0;
    first(:, p, :) = reshape(int32(F), nQ, 1, nx);
    last(:, p, :) = reshape(int32(L), nQ, 1, nx);
    enabled(:, p) = ok;
end
A = struct('shapeX', shX, 'shapeU', nP, 'first', first, 'last', last, 'enabled', enabled);
end
