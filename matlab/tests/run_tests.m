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
            RA = random_abstraction(shapes{k}, 6, mod(t, 2), mod(t, 3));
            nQ = prod(RA.shapeX);
            safe = rand(nQ, 1) < 0.95; tgt = rand(nQ, 1) < 0.15; av = rand(nQ, 1) < 0.1;
            [d1, a1] = ddma.safety(RA, safe, 'matlab'); [d2, a2] = ddma.safety(RA, safe, 'mex');
            [r1, b1, v1] = ddma.reach_avoid(RA, tgt, av, 'matlab');
            [r2, b2, v2] = ddma.reach_avoid(RA, tgt, av, 'mex');
            ok = ok && isequal(d1, d2) && isequal(a1, a2) && isequal(r1, r2) && ...
                 isequal(b1, b2) && isequal(v1, v2);
        end
        nfail = nfail + check(ok, sprintf('random %d-D abstractions (mex = matlab)', numel(shapes{k})));
    end
end

% 4. bounded-derivative models (thesis Ch. 3), all methods, MATLAB code and MEX
bdb = {'matlab'};
if exist('ddma_growth_mex', 'file') == 3, bdb{end + 1} = 'mex'; end
if isfield(R, 'bd')
    bd = R.bd;
    for b = 1:numel(bdb)
        okAll = true; worst = 0;
        for k = 1:numel(bd)
            c = bd{k};
            sh = ddma.grid_shape(c.edges); m = size(c.Y, 2);
            S = ddma.bd_fit(c.Z, c.Y, c.edges, c.g_lo, c.g_hi, c.d_lo, c.d_hi, c.method, ...
                            c.local, bdb{b});
            lo = reshape(S.lower, [sh, m]); hi = reshape(S.upper, [sh, m]);
            tol = 1e-9; if strncmp(c.method, 'lin_', 4), tol = 1e-6; end
            [ok, d] = close_to(lo(:), c.lower(:), tol);
            [ok2, d2] = close_to(hi(:), c.upper(:), tol);
            if ~(ok && ok2)
                fprintf('        case %d (%s %s): max diff %g\n', k, c.method, c.local, max(d, d2));
            end
            okAll = okAll && ok && ok2; worst = max([worst, d, d2]);
        end
        nfail = nfail + check(okAll, sprintf('bounded-derivative models, %d cases (%s, max diff %.1e)', ...
            numel(bd), bdb{b}, worst));
    end
    % 5. bounded-derivative model + abstraction + safety (pendulum)
    p = R.pend;
    Z = [p.X, p.U];
    for mth = {'all_matrix', 'growth'}
        loc = ''; if strcmp(mth{1}, 'growth'), loc = 'slab'; end
        S = ddma.bd_fit(Z, p.Xn, [p.Xedges(:); p.Uedges(:)]', [p.A_lo, p.B_lo], ...
                        [p.A_hi, p.B_hi], -0.005, 0.005, mth{1}, loc);
        A = ddma.abstraction(p.Xedges, p.Uedges, S);
        safe = ddma.cells_inside(p.Xedges, [-0.8, -1.2], [0.8, 1.2]);
        dom = ddma.safety(A, safe);
        ref = logical(p.([mth{1} '_domain']));
        nfail = nfail + check(isequal(reshape(dom, A.shapeX), ref) && any(ref(:)), ...
            sprintf('pendulum safety with ddma.bd_fit (%s)', mth{1}));
    end
end

% 6. online refinement vs Python: fit, exact update, windowed update, refresh,
%    incremental safety (known_winning)
if isfield(R, 'online')
    o = R.online;
    Z = [o.X, o.U]; Xn = o.Xn;
    edges = [o.Xedges(:); o.Uedges(:)]';
    glo = [o.A_lo, o.B_lo]; ghi = [o.A_hi, o.B_hi];
    safe = logical(o.safe(:));
    for b = 1:numel(bdb)
        S = ddma.bd_fit(Z(1:300, :), Xn(1:300, :), edges, glo, ghi, -0.005, 0.005, ...
                        'growth', 'slab', bdb{b});
        A = ddma.abstraction(o.Xedges, o.Uedges, S);
        d0 = ddma.safety(A, safe);
        S = ddma.bd_update(S, Z(301:20300, :), Xn(301:20300, :));
        sh = [ddma.grid_shape(edges), 2];
        ok = close_to(reshape(S.lower, sh), o.lo1, 1e-9) && close_to(reshape(S.upper, sh), o.hi1, 1e-9);
        [A, S] = ddma.abstraction_refresh(A, S, true(numel(safe), 1));
        d1 = ddma.safety(A, safe, [], d0);
        S = ddma.bd_update_local(S, Z(20301:20330, :), Xn(20301:20330, :), o.window);
        ok = ok && size(S.changed, 1) == 30;
        ok = ok && close_to(reshape(S.lower, sh), o.lo2, 1e-9) && close_to(reshape(S.upper, sh), o.hi2, 1e-9);
        [A, S, mask] = ddma.abstraction_refresh(A, S);
        ok = ok && isequal(mask, logical(o.refreshed(:))) && isempty(S.changed);
        shX = A.shapeX; shU = A.shapeU;
        en = reshape(A.enabled, [shX, shU]);
        msk = repmat(en, [1 1 1 2]);
        first = reshape(A.first, [shX, shU, 2]); last = reshape(A.last, [shX, shU, 2]);
        ok = ok && isequal(en, logical(o.enabled2)) && ...
             isequal(double(first(msk)), double(o.first2(msk))) && ...
             isequal(double(last(msk)), double(o.last2(msk)));
        d2 = ddma.safety(A, safe, [], d1);
        ok = ok && isequal(d0, logical(o.dom0(:))) && isequal(d1, logical(o.dom1(:))) && ...
             isequal(d2, logical(o.dom2(:))) && any(d2) && all(d2 >= d1);
        nfail = nfail + check(ok, sprintf('online: update, update_local, refresh, known_winning (%s)', bdb{b}));
    end
end

% 7. update = refit (all methods), windowed update never tighter, full window = update,
%    known_winning in both synthesis backends
ddma.seed(3);
edges = {linspace(-1, 1, 7)', [-1; -0.3; 0.2; 1]};
Z = 2.4 * rand(260, 2) - 1.2;
Y = [sin(2 * Z(:, 1)) + 0.4 * Z(:, 2), Z(:, 1) .* Z(:, 2)];
glo = [-2 0.3; -1.2 -1.2]; ghi = [2 0.5; 1.2 1.2];
ok = true;
for mth = {'two_matrix', 'all_matrix', 'lin_mono', 'growth', 'lin_growth'}
    locs = {''}; if ~isempty(strfind(mth{1}, 'growth')), locs = {'', 'cell', 'slab'}; end
    for l = 1:numel(locs)
        Sall = ddma.bd_fit(Z, Y, edges, glo, ghi, -0.02, 0.02, mth{1}, locs{l});
        S = ddma.bd_fit(Z(1:150, :), Y(1:150, :), edges, glo, ghi, -0.02, 0.02, mth{1}, locs{l});
        Su = ddma.bd_update(S, Z(151:end, :), Y(151:end, :));
        tol = 1e-9; if strncmp(mth{1}, 'lin_', 4), tol = 1e-6; end
        ok = ok && close_to(Su.lower, Sall.lower, tol) && close_to(Su.upper, Sall.upper, tol);
        if ~isempty(strfind(mth{1}, 'growth'))
            Sw = ddma.bd_update_local(S, Z(151:end, :), Y(151:end, :), 1);
            ok = ok && all(Sw.upper(:) >= Sall.upper(:) - tol) && all(Sw.lower(:) <= Sall.lower(:) + tol);
            Sb = ddma.bd_update_local(S, Z(151:end, :), Y(151:end, :), 100);
            ok = ok && close_to(Sb.lower, Sall.lower, tol) && close_to(Sb.upper, Sall.upper, tol);
        end
    end
end
nfail = nfail + check(ok, 'update = refit (5 methods), windowed update never tighter, full window = update');
if numel(synth) > 1
    ok = true;
    ddma.seed(4);
    for t = 0:5
        RA = random_abstraction([9 8], 6, mod(t, 2), mod(t, 3));
        nQ = prod(RA.shapeX);
        safe = rand(nQ, 1) < 0.95;
        [d, a] = ddma.safety(RA, safe, 'matlab');
        inner = ddma.safety(RA, d & rand(nQ, 1) < 0.5, 'matlab');   % an invariant subset
        [d1, a1] = ddma.safety(RA, safe, 'matlab', inner);
        [d2, a2] = ddma.safety(RA, safe, 'mex', inner);
        ok = ok && isequal(d1, d) && isequal(d2, d) && isequal(a1, a) && isequal(a2, a);
    end
    bad = false;
    try
        ddma.safety(RA, safe, 'mex', safe & ~d);   % not invariant
    catch
        bad = true;
    end
    nfail = nfail + check(ok && (bad || ~any(safe & ~d)), 'safety with known_winning (matlab = mex = plain; checked)');
end

if nfail == 0
    fprintf('ALL MATLAB TESTS PASSED (%d backends)\n', numel(backends));
else
    error('run_tests:fail', '%d test(s) failed', nfail);
end
end

function [ok, d] = close_to(a, b, tol)
% same infinities, finite values within tol
ok = isequal(a == Inf, b == Inf) && isequal(a == -Inf, b == -Inf);
f = isfinite(a) & isfinite(b);
d = 0; if any(f), d = max(abs(a(f) - b(f))); end
ok = ok && d <= tol;
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
