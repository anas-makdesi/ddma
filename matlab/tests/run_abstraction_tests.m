function run_abstraction_tests()
%RUN_ABSTRACTION_TESTS The MEX kernel ddma_abs_mex builds the same abstraction as the
%   MATLAB code, for any number of threads: maps with and without g, features, per-output
%   features, single-precision maps, assumptions, uneven partitions, lazy abstractions
%   and refresh. Run from the matlab/ folder: addpath tests; run_abstraction_tests
if exist('ddma_abs_mex', 'file') ~= 3
    fprintf('run_abstraction_tests: ddma_abs_mex not built, skipped\n');
    return
end
[nmax, openmp] = ddma.get_num_threads();
fprintf('OpenMP: %d, default threads: %d\n', openmp, nmax);
threads = unique([1 2 3 nmax]);
c = onCleanup(@() ddma.internal.mex_opts('reset'));
nfail = 0;
rand('seed', 3); randn('seed', 3); %#ok<RAND>
nen = [];

for t = 1:12
    nx = 1 + mod(t, 3); nu = 1 + mod(t, 2);
    uniform = mod(t, 4) < 2;
    Xedges = make_edges(nx, 4 + mod(3 * t, 6), uniform);
    Uedges = make_edges(nu, 3 + mod(t, 3), uniform);
    A0 = 0.6 * eye(nx) + 0.15 * rand(nx); B0 = 0.3 * rand(nx, nu);
    K = 400;
    X = -1 + 2 * rand(K, nx); U = -1 + 2 * rand(K, nu);
    Y = X * A0' + U * B0' + 0.05 * tanh(X);
    % options of this case
    g = []; feats = []; assumption = [];
    if mod(t, 3) == 1                         % affine g, map learned on (x) only
        g = ddma.affine_bounds(A0, B0);
        Y = Y - X * A0' - U * B0';
        feats = 1:nx;
    elseif mod(t, 3) == 2                     % some other interval function g
        ga = ddma.affine_bounds(0.5 * A0, B0);
        g = @(xl, xh, ul, uh) widen(ga, xl, xh, ul, uh);
        Y = Y - 0.5 * X * A0' - U * B0';
    end
    if mod(t, 4) == 3 && nx >= 2               % assumption on the last coordinate
        e = Xedges{nx};
        assumption = [-Inf(1, nx); Inf(1, nx)];
        assumption(:, nx) = [e(2); e(end - 1)];
    end
    if isempty(feats), Z = [X, U]; feats_used = 1:(nx + nu); else, Z = X(:, feats); feats_used = feats; end
    joint = [Xedges(:); Uedges(:)];
    S = ddma.fit(Z, Y, joint(feats_used), -0.02, 0.02);
    if mod(t, 5) == 0, S = ddma.compact(S); end
    ref = ddma.abstraction(Xedges, Uedges, S, g, feats, assumption, false, 'matlab');
    nen(end + 1) = mean(ref.enabled(:)); %#ok<AGROW>
    ok = true;
    for th = threads
        ddma.set_num_threads(th);
        A = ddma.abstraction(Xedges, Uedges, S, g, feats, assumption, false, 'mex');
        ok = ok && same(ref, A);
    end
    ddma.set_num_threads(0);
    % lazy: transitions on demand (paired MEX calls) = eager rows
    L = ddma.abstraction(Xedges, Uedges, S, g, feats, assumption, true, 'mex');
    nQ = prod(ref.shapeX); nP = prod(ref.shapeU);
    q = randi(nQ, 300, 1); p = randi(nP, 300, 1);
    [en, F, Lst] = ddma.internal.pair_boxes(L, S, q, p);
    idx = sub2ind([nQ nP], q, p);
    rf = reshape(ref.first, nQ * nP, []); rl = reshape(ref.last, nQ * nP, []);
    ok = ok && isequal(en, ref.enabled(idx)) && isequal(F, double(rf(idx, :))) && ...
         isequal(Lst, double(rl(idx, :)));
    % refresh of some rows = the full abstraction
    A = ddma.abstraction(Xedges, Uedges, S, g, feats, assumption, false, 'mex');
    rows = find(rand(nQ, 1) < 0.3);
    A.first(rows, :, :) = 0; A.enabled(rows, :) = false;
    A = ddma.abstraction_refresh(A, S, rows);
    ok = ok && same(ref, A);
    nfail = nfail + check(ok, sprintf('case %d: nx=%d nu=%d uniform=%d g=%d assumption=%d single=%d', ...
        t, nx, nu, uniform, mod(t, 3), ~isempty(assumption), isa(S.lower, 'single')));
end
nfail = nfail + check(mean(nen > 0 & nen < 1) > 0.7, 'cases have enabled and disabled pairs');

% per-output features (bounded-derivative model, as in run_tests)
Xedges = make_edges(2, 12, true); Uedges = make_edges(1, 5, true);
K = 3000; X = -1 + 2 * rand(K, 2); U = -1 + 2 * rand(K, 1);
Xn = [0.9 * X(:, 1) + 0.1 * X(:, 2), X(:, 2) + 0.1 * (-sin(X(:, 1)) + U)];
edges = [Xedges(:); Uedges(:)];
glo = [0.85 0.05 -0.01; -0.15 0.95 0.05]; ghi = [0.95 0.15 0.01; 0.05 1.05 0.15];
Sf = ddma.bd_fit([X, U], Xn, edges, glo, ghi, -0.005, 0.005, 'all_matrix', '', [], {[1 2], [1 2 3]});
ref = ddma.abstraction(Xedges, Uedges, Sf, [], [], [], false, 'matlab');
ok = true;
for th = threads
    ddma.set_num_threads(th);
    ok = ok && same(ref, ddma.abstraction(Xedges, Uedges, Sf, [], [], [], false, 'mex'));
end
ddma.set_num_threads(0);
nfail = nfail + check(ok && any(ref.enabled(:)), 'per-output features (bd_fit)');

% NaN bounds with an assumption: the pair is not enabled (MATLAB code and MEX)
S = ddma.fit([0 0; 1 1], [0.5; 0.5], {[0 0.5 1], [0 1]}, 0, 0);
S.lower(1) = NaN;
a = [0; 1];
Am = ddma.abstraction({[0 0.5 1]}, {[0 1]}, S, [], [], a, false, 'matlab');
Ax = ddma.abstraction({[0 0.5 1]}, {[0 1]}, S, [], [], a, false, 'mex');
nfail = nfail + check(same(Am, Ax) && ~Am.enabled(1), 'NaN bound with an assumption: not enabled');

if nfail > 0
    error('ddma:tests', '%d abstraction test(s) failed', nfail);
end
fprintf('ALL ABSTRACTION TESTS PASSED\n');
end

function [lo, hi] = widen(ga, xl, xh, ul, uh)
[lo, hi] = ga(xl, xh, ul, uh);
lo = lo - 0.01; hi = hi + 0.01;
end

function tf = same(A, B)
tf = isequal(A.enabled, B.enabled) && isequal(int32(A.first), int32(B.first)) && ...
     isequal(int32(A.last), int32(B.last));
end

function edges = make_edges(n, k, uniform)
edges = cell(1, n);
for j = 1:n
    if uniform
        edges{j} = linspace(-1, 1, k + 1);
    else
        edges{j} = [-1, sort(-1 + 2 * rand(1, k - 1)), 1];
    end
end
end

function n = check(ok, name)
if ok
    fprintf('  ok    %s\n', name);
    n = 0;
else
    fprintf('  FAIL  %s\n', name);
    n = 1;
end
end
