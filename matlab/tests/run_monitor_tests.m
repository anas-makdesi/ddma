function run_monitor_tests()
%RUN_MONITOR_TESTS Assumption monitor: compare MATLAB with the Python reference.
%   Run from the matlab/ folder:  addpath tests; run_monitor_tests
%   (reference written by tests/make_reference_monitor.py; indices there are 0-based)
here = fileparts(mfilename('fullpath'));
R = load(fullfile(here, 'reference_monitor.mat'));
nfail = 0;

% A: Lipschitz bound too small
M = ddma.monitor([-1 -1], [1 1], -0.02, 0.02);
M = ddma.monitor_add(M, R.a_Z(1:60, :), R.a_y(1:60));
[M, v] = ddma.monitor_check(M, R.a_Z(61:end, :), R.a_y(61:end));
nfail = nfail + check(same_viol(v, R.a_viol), 'violations (Lipschitz bound too small)');
nfail = nfail + check(same_need(M, R, 'a_'), 'needed changes (Lipschitz)');
nfail = nfail + check(abs(M.need_hi(1, 1) - 3) < 0.2, 'needed slope close to the true 3');

% B: wrong sign, map on a grid (model bounds checked too)
edges = {linspace(0, 1, 11), linspace(0, 1, 11)};
S = ddma.fit(R.b_Z(1:150, :), R.b_y(1:150), edges, -0.01, 0.01);
M = ddma.monitor_for_model(S, R.b_Z(1:150, :), R.b_y(1:150));
[M, v] = ddma.monitor_check(M, R.b_Z(151:end, :), R.b_y(151:end));
nfail = nfail + check(same_viol(v, R.b_viol), 'violations (wrong sign)');
nfail = nfail + check(same_need(M, R, 'b_'), 'needed changes (wrong sign)');
nfail = nfail + check(~isempty(strfind(ddma.monitor_report(M), 'contradicts ''non-decreasing''')), ...
    'report names the sign');

% C: disturbance grows, widen policy
M = ddma.monitor_monotone([1 1; 1 -1], -0.01, 0.01, struct('on_violation', 'widen', 'margin', 0.1));
M = ddma.monitor_add(M, R.c_Z(1:100, :), R.c_Y(1:100, :));
[M, v] = ddma.monitor_check(M, R.c_Z(101:end, :), R.c_Y(101:end, :));
nfail = nfail + check(same_viol(v, R.c_viol), 'violations (widen policy)');
nfail = nfail + check(max(abs([M.d_lo - R.c_dlo, M.d_hi - R.c_dhi])) < 1e-12 ...
    && numel(M.changes) == R.c_nchanges && M.needs_refit, 'widened disturbance bounds');
nfail = nfail + check(ddma.monitor_consistent(M), 'all data consistent after widening');

% D: missing dependence, then widen that slope bound
M = ddma.monitor_monotone([1 0], -0.01, 0.01, struct('slope_limit', 2));
M = ddma.monitor_add(M, R.d_Z(1:50, :), R.d_y(1:50));
[M, v] = ddma.monitor_check(M, R.d_Z(51:end, :), R.d_y(51:end));
nfail = nfail + check(same_viol(v, R.d_viol), 'violations (missing dependence)');
nfail = nfail + check(same_need(M, R, 'd_'), 'needed changes (missing dependence)');
M = ddma.monitor_widen(M, {'slope_hi', 1, 2});
nfail = nfail + check(isequal(isinf(M.slope_hi), isinf(R.d_slope_hi)) && ...
    abs(M.slope_hi(2) - R.d_slope_hi(2)) < 1e-12 && ddma.monitor_consistent(M), 'widen a slope bound');

% E: growth model, nothing stored: the model bounds catch a jump
S = ddma.bd_fit(R.e_Z, R.e_y, {linspace(-1, 1, 9), linspace(-1, 1, 9)}, [-1 0.3], [1 0.3], ...
                -0.02, 0.02, 'growth');
M = ddma.monitor_for_model(S, [], []);
nfail = nfail + check(isequal(M.slope_lo, R.e_lo) && isequal(M.slope_hi, R.e_hi), 'slopes read from the model');
[M, v] = ddma.monitor_check(M, R.e_Zq, R.e_yq);
nfail = nfail + check(same_viol(v, R.e_viol), 'violations through the model bounds');

% F: collect during a run (levels 1-2, same random samples as Python), flush at base
lo = zeros(2); hi = [inf inf; 0 0]; dlo = [-0.05 -1]; dhi = [0.05 1];
for b = backends()
    M = ddma.monitor(lo, hi, dlo, dhi, struct('full', 'later', 'sample', 150, 'seed', 11, 'backend', b{1}));
    M = ddma.monitor_add(M, R.f_Z(1:400, :), R.f_Y(1:400, :));
    [M, v1] = ddma.monitor_check(M, R.f_Z(401:end, :), R.f_Y(401:end, :));
    [M, v2] = ddma.monitor_flush(M);
    nfail = nfail + check(same_viol(v1, R.f_viol1) && same_viol(v2, R.f_viol2), ...
        ['levels 1-2 during the run, level 3 at flush (' b{1} ')']);
    nfail = nfail + check(same_need(M, R, 'f_') && isequal(double(M.st(1:M.K)), R.f_st), ...
        ['needed changes and quarantine after flush (' b{1} ')']);
end

% G: check_data
for b = backends()
    D = ddma.check_data(R.g_Z, R.g_Y, lo, hi, dlo, dhi, struct('backend', b{1}));
    p = R.g_pair + 1; p(R.g_pair < 0) = nan;
    nd.need_width = D.needed.width; nd.need_lo = D.needed.slope_lo; nd.need_hi = D.needed.slope_hi;
    ok = isequal(D.violating, R.g_count) && isequaln(D.worst_pair, p) && ...
         isequal(D.conflicts, R.g_conf) && max(abs(D.worst(isfinite(D.worst)) - R.g_worst(isfinite(R.g_worst)))) < 1e-12 ...
         && same_need(nd, R, 'g_');
    nfail = nfail + check(ok && ~D.ok && ~isempty(strfind(D.report, 'Most conflicts')), ['check_data (' b{1} ')']);
end

% H, I: widen_then_fallback beyond and under the limit
for key = {'h_', 'i_'}
    k = key{1};
    M = ddma.monitor(0, 2, -0.05, 0.05, struct('on_violation', 'widen_then_fallback'));
    M = ddma.monitor_add(M, R.([k 'Z'])(1:400), R.([k 'y'])(1:400));
    [M, v] = ddma.monitor_check(M, R.([k 'Z'])(401:end), R.([k 'y'])(401:end));
    since = M.fallback_since; if isempty(since), since = -1; else, since = since - 1; end
    ok = same_viol(v, R.([k 'viol'])) && M.fallback_active == R.([k 'fallback']) && since == R.([k 'since']) ...
         && max(abs([M.d_lo - R.([k 'dlo']), M.d_hi - R.([k 'dhi'])])) < 1e-12;
    nfail = nfail + check(ok, ['widen_then_fallback, ' k(1)]);
end

% MEX and MATLAB code: same bits, any number of threads
if exist('ddma_monitor_mex', 'file') == 3
    rand('seed', 3);
    Zs = rand(900, 2); Ys = [0.8 * Zs(:, 1) + 0.3 * Zs(:, 2) + 0.1 * (2 * rand(900, 1) - 1), 3 + 2 * rand(900, 1) - 1];
    Ys(700 + (1:15), 1) = Ys(700 + (1:15), 1) - 0.3 * rand(15, 1);
    ref = run_case(Zs, Ys, 'matlab', true);
    ok = true;
    old = ddma.internal.mex_opts();
    for t = 1:3
        ddma.set_num_threads(t);
        for pr = [false true]
            r = run_case(Zs, Ys, 'mex', pr);
            ok = ok && isequaln(r.v, ref.v) && isequaln(r.need, ref.need) && isequal(r.st, ref.st) ...
                    && isequaln(r.D, ref.D);
        end
    end
    ddma.set_num_threads(old(1));
    nfail = nfail + check(ok, 'MEX = MATLAB code (bits), 1-3 threads, with and without pruning');
end

% input rules
nfail = nfail + check(raises(@() ddma.monitor(0, inf, 0, 1, struct('widen', {{'slope_hi', 1, 1}}), 'slope_limit', 3)), ...
    'a sign cannot be widened');
nfail = nfail + check(raises(@() ddma.monitor(0, 1, 0, 1, struct('widen', {{'slope_hi', 1, 1}}))), ...
    'a slope target needs slope_limit');
nfail = nfail + check(raises(@() ddma.monitor(0, 1, 0, 1, struct('max_points', 5))), 'unknown option');

% fallback latch, reset, callback, quarantine
global ddma_monitor_calls %#ok<GVMIS>
ddma_monitor_calls = 0;
M = ddma.monitor(0, 1, -0.01, 0.01, struct('on_violation', 'fallback', 'callback', @count_call));
M = ddma.monitor_add(M, [0.2; 0.4], [0.2; 0.4]);
[M, v] = ddma.monitor_check(M, 0.5, 2);
M = ddma.monitor_check(M, 0.3, 0.3);  % conflicts only with the quarantined point: no alarm
ok = M.fallback_active && M.fallback_since == 1 && ddma_monitor_calls == 1 && numel(v) == 1;
Q = ddma.monitor_quarantine(M);
[Zc, ~] = ddma.monitor_data(M);
ok = ok && isequal(Q.ids, 1) && size(Zc, 1) == 3;
M = ddma.monitor_reset(M);
nfail = nfail + check(ok && ~M.fallback_active && isempty(M.violating), 'fallback latch, callback, quarantine, reset');

if nfail
    error('run_monitor_tests:failed', '%d monitor test(s) failed', nfail);
end
fprintf('all monitor tests passed\n');
end

function count_call(~)
global ddma_monitor_calls %#ok<GVMIS>
ddma_monitor_calls = ddma_monitor_calls + 1;
end

function ok = same_viol(v, ref)
ok = numel(v) == size(ref, 1);
if ~ok || isempty(v), return; end
levels = {'cells', 'sample', 'full'};
w = [v.witness]'; wr = ref(:, 5); wr(wr >= 0) = wr(wr >= 0) + 1;
ok = isequal([v.index]', ref(:, 1) + 1) && isequal([v.output]', ref(:, 2) + 1) ...
     && isequal({v.level}', levels(ref(:, 3) + 1)') && isequaln(w, wr) ...
     && isequal([v.found_at]', ref(:, 6) + 1) && max(abs([v.gap]' - ref(:, 4))) < 1e-12;
end

function b = backends()
b = {'matlab'};
if exist('ddma_monitor_mex', 'file') == 3, b{end + 1} = 'mex'; end
end

function r = run_case(Z, Y, backend, prune)
o = struct('full', 'later', 'sample', 100, 'seed', 5, 'backend', backend, 'prune', prune);
M = ddma.monitor(zeros(2), [inf inf; 0 0], [-0.1 -1], [0.1 1], o);
M = ddma.monitor_add(M, Z(1:600, :), Y(1:600, :));
[M, v1] = ddma.monitor_check(M, Z(601:end, :), Y(601:end, :));
[M, v2] = ddma.monitor_flush(M);
r.v = [v1, v2]; r.need = ddma.monitor_needed(M); r.st = M.st(1:M.K);
D = ddma.check_data(Z, Y, zeros(2), [inf inf; 0 0], [-0.1 -1], [0.1 1], ...
                    struct('backend', backend, 'prune', prune));
r.D = rmfield(D, 'pairs_evaluated');
end

function tf = raises(f)
tf = false;
try
    f();
catch
    tf = true;
end
end

function ok = same_need(M, R, p)
if isfield(M, 'slope_lo'), n = ddma.monitor_needed(M);
else, n = struct('width', M.need_width, 'slope_lo', M.need_lo, 'slope_hi', M.need_hi); end
a = [n.width(:); n.slope_lo(:); n.slope_hi(:)];
b = [R.([p 'need_w'])(:); R.([p 'need_lo'])(:); R.([p 'need_hi'])(:)];
ok = isequal(isnan(a), isnan(b)) && isequal(isinf(a), isinf(b));
f = isfinite(a) & isfinite(b);
ok = ok && max([0; abs(a(f) - b(f))]) < 1e-12;
end

function f = check(ok, name)
if ok, fprintf('  ok    %s\n', name); f = 0; else, fprintf('  FAIL  %s\n', name); f = 1; end
end
