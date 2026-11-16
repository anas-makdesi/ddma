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
M = ddma.monitor_monotone([1 0], -0.01, 0.01);
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
M = ddma.monitor_for_model(S, [], [], struct('max_points', 0));
nfail = nfail + check(isequal(M.slope_lo, R.e_lo) && isequal(M.slope_hi, R.e_hi), 'slopes read from the model');
[M, v] = ddma.monitor_check(M, R.e_Zq, R.e_yq);
nfail = nfail + check(same_viol(v, R.e_viol), 'violations through the model bounds');

% fallback latch, reset and callback
global ddma_monitor_calls %#ok<GVMIS>
ddma_monitor_calls = 0;
M = ddma.monitor(0, 1, -0.01, 0.01, struct('on_violation', 'fallback', 'callback', @count_call));
M = ddma.monitor_add(M, [0.2; 0.4], [0.2; 0.4]);
[M, v] = ddma.monitor_check(M, 0.5, 2);
M = ddma.monitor_check(M, 0.3, 0.3);  % the bad point stays stored, so this one conflicts too
ok = M.fallback_active && M.fallback_since == 1 && ddma_monitor_calls == 2 && numel(v) == 1;
M = ddma.monitor_reset(M);
nfail = nfail + check(ok && ~M.fallback_active && isempty(M.violating), 'fallback latch, callback, reset');

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
w = [v.witness]'; wr = ref(:, 5); wr(wr >= 0) = wr(wr >= 0) + 1;
ok = isequal([v.index]', ref(:, 1) + 1) && isequal([v.output]', ref(:, 2) + 1) ...
     && isequal(strcmp({v.kind}, 'model')', ref(:, 3) == 1) && isequal(w, wr) ...
     && max(abs([v.gap]' - ref(:, 4))) < 1e-12;
end

function ok = same_need(M, R, p)
n = ddma.monitor_needed(M);
a = [n.width(:); n.slope_lo(:); n.slope_hi(:)];
b = [R.([p 'need_w'])(:); R.([p 'need_lo'])(:); R.([p 'need_hi'])(:)];
ok = isequal(isnan(a), isnan(b)) && isequal(isinf(a), isinf(b));
f = isfinite(a) & isfinite(b);
ok = ok && max([0; abs(a(f) - b(f))]) < 1e-12;
end

function f = check(ok, name)
if ok, fprintf('  ok    %s\n', name); f = 0; else, fprintf('  FAIL  %s\n', name); f = 1; end
end
