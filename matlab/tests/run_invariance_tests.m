function run_invariance_tests()
%RUN_INVARIANCE_TESTS Gridless safe envelopes: compare MATLAB with the Python reference.
%   Run from the matlab/ folder:  addpath tests; run_invariance_tests
here = fileparts(mfilename('fullpath'));
R = load(fullfile(here, 'reference_invariance.mat'));
nfail = 0;
tol = 1e-12;

M = ddma.maximal_points(R.mp_P);
nfail = nfail + check(same_rows(M, R.mp_M, 0), 'maximal_points');
v = ddma.max_min_cones(R.mm_p, R.mm_q, R.mm_alpha, R.mm_beta);
nfail = nfail + check(max(abs(v - R.mm_val)) < 1e-12, 'max_min_cones');

o = struct('w_lo', 0, 'w_hi', 0.15, 'input_signs', -1, 'x_hi', [26 26 26], 'u_lo', 0, 'u_hi', 1);
E = ddma.monotone_envelope(R.r_X, R.r_Xn, R.r_U, o);
nfail = nfail + check(same_rows(E.corners, R.r_corners, tol), 'monotone_envelope corners (rooms)');
nfail = nfail + check(same_rows([E.pairs_T, E.pairs_U], [R.r_pT, R.r_pU], tol), ...
    'monotone_envelope (corner, input) pairs');
nfail = nfail + check(isequal(ddma.envelope_contains(E, R.r_Q), R.r_in == 1), 'contains (rooms)');
u = nan(200, 1);
for k = 1:200
    uk = ddma.envelope_input(E, R.r_Q(k, :), 0);
    if ~isempty(uk), u(k) = uk; end
end
nfail = nfail + check(isequaln(isnan(u), isnan(R.r_u)) && max(abs(u(~isnan(u)) - R.r_u(~isnan(u)))) < tol, ...
    'envelope_input (rooms)');
o2 = struct('w_lo', [-0.02 -0.02], 'w_hi', [0.02 0.02], 'state_signs', [1 -1], ...
            'x_hi', [0.8 inf], 'x_lo', [-inf -0.7]);
E2 = ddma.monotone_envelope(R.a_X, R.a_Xn, [], o2);
nfail = nfail + check(same_rows(E2.corners, R.a_corners, tol), 'monotone_envelope (flipped state)');

oc = struct('w_lo', -0.005, 'w_hi', 0.005, 'x_lo', [0.6 0.6], 'x_hi', [1.1 1.1]);
B = ddma.embedding_envelope(R.c_X, R.c_Xn, R.c_U, R.c_jl, R.c_jh, eye(2), eye(2), oc);
nfail = nfail + check(isequal(B.pairs, R.c_pairs) && max(abs([B.lo(:); B.hi(:)] - [R.c_lo(:); R.c_hi(:)])) < tol ...
    && isequal([B.ulo, B.uhi], [R.c_ulo, R.c_uhi]), 'embedding_envelope boxes (competition)');
nfail = nfail + check(isequal(ddma.envelope_contains(B, R.c_Q), R.c_in == 1), 'contains (boxes)');
uc = nan(200, 2);
for k = 1:200
    uk = ddma.envelope_input(B, R.c_Q(k, :), [-0.02 -0.02]);
    if ~isempty(uk), uc(k, :) = uk; end
end
ok = ~isnan(uc(:, 1));
nfail = nfail + check(isequal(ok, ~isnan(R.c_u(:, 1))) && max(max(abs(uc(ok, :) - R.c_u(ok, :)))) < tol, ...
    'envelope_input (boxes)');

if nfail
    error('run_invariance_tests:failed', '%d invariance test(s) failed', nfail);
end
fprintf('all invariance tests passed\n');
end

function ok = same_rows(A, B, tol)
% same set of rows (up to order); values equal within tol
ok = isequal(size(A), size(B));
if ~ok || isempty(A), return; end
A = sortrows(A); B = sortrows(B);
ok = max(abs(A(:) - B(:))) <= tol;
end

function f = check(ok, name)
if ok, fprintf('  ok    %s\n', name); f = 0; else, fprintf('  FAIL  %s\n', name); f = 1; end
end
