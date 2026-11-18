%ASSUMPTION_MONITOR What if the assumptions are wrong? (MATLAB/Octave)
%   Same scenarios as examples/assumption_monitor.py. The cruise-control safety filter of
%   acc_safety_filter.m assumes a speed update monotone in (v, u) and a leader speed in
%   [13, 17] m/s. ddma.check_data first checks every pair of the learning data. At
%   t = 75 s, with the response 'widen_then_fallback' (limit: 2 x the original width):
%     A. the leader slows down to 11 m/s (a disturbance bound is wrong): the needed width
%        stays under the limit, so the monitor widens the bound, and the map and the safe
%        set are recomputed;
%     B. a trailer is hitched (30 % of the acceleration, twice the drag): the needed width
%        is beyond the limit, so the monitor switches to a backup controller.
%   Without a monitor the filter trusts its model until the state leaves the certified
%   domain; outside it (also outside a refit, smaller, safe set), every run uses the
%   backup. The output shows which level caught each event. Run from the matlab/ folder:
%   run examples/assumption_monitor
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));   % the +ddma package
T = 0.5; Mv = 1370; f0 = 51; f1 = 1.2567; f2 = 0.4342;
VL = [13 17]; W = 0.05;
Xlo = [5 0]; Xhi = [60 30]; Ulo = -3; Uhi = 3;
SAFE_LO = [10 2]; SAFE_HI = [50 28];
drag = @(v) (f0 + f1 * v + f2 * v.^2) / Mv;
A_G = [1 -T; 0 0];                       % known part g(x) = A_G x
SIGNS = [0 0; 1 1];                      % gap update constant, speed update monotone
features = [2 3];
Xedges = ddma.uniform_partition(Xlo, Xhi, [110 60]);
Uedges = ddma.uniform_partition(Ulo, Uhi, 24);
safe = ddma.cells_inside(Xedges, SAFE_LO, SAFE_HI);

ddma.seed(0);
K = 50000;
X = Xlo + rand(K, 2) .* (Xhi - Xlo);
U = Ulo + rand(K, 1) * (Uhi - Ulo);
Y = [T * (VL(1) + rand(K, 1) * diff(VL)), X(:, 2) + T * (U - drag(X(:, 2))) + W * (2 * rand(K, 1) - 1)];   % y = x' - g(x)
Z = [X(:, 2) U];
R = ddma.check_data(Z, Y, [], [], [T * VL(1), -W], [T * VL(2), W], struct('signs', SIGNS));
disp(R.report)
S = ddma.fit(Z, Y, {Xedges{2}, Uedges{1}}, [T * VL(1), -W], [T * VL(2), W], SIGNS);
A = ddma.abstraction(Xedges, Uedges, S, ddma.affine_bounds(A_G), features);
[domain, allowed] = ddma.safety(A, safe);
fprintf('certified safe set: %.0f%% of the specification\n', 100 * nnz(domain) / nnz(safe));

N = 500; KE = 150;
nominal = @(x) min(max(0.3 * (22 - x(2)), Ulo), Uhi);
backup = @(x, xp) min(max(1.5 * (xp(2) + (x(1) - xp(1)) / T - x(2)) + 0.2 * (x(1) - 30), Ulo), Uhi);
scen = {'slow leader', 'trailer'};
for s = 1:2
    ddma.seed(1);
    slow = 13 * ones(1, N); if s == 1, slow(KE+1:end) = 11; end
    fast = mod(floor((0:N-1) / 40), 2) == 0;
    vl = min(max(fast * 17 + ~fast .* slow + (rand(1, N) - 0.5), min(slow, 13)), 17);
    for withMon = [false true]
        ddma.seed(2);
        As = A; al = allowed;
        if withMon
            Mon = ddma.monitor_for_model(S, Z, Y, ...
                struct('on_violation', 'widen_then_fallback', 'widen_limit', 2, 'margin', 0.2, ...
                       'input_names', {{'v', 'u'}}, 'output_names', {{'gap update', 'speed update'}}));
        end
        x = [30 15]; xp = x; nbad = 0; alarm = NaN; worst = inf;
        for k = 1:N
            if withMon && Mon.fallback_active
                u = backup(x, xp);
            else
                u = ddma.controller_input(As, al, x, nominal(x));
                if isempty(u), u = backup(x, xp); end
            end
            trailer = s == 2 && k > KE;
            if trailer, acc = 0.3 * u - 2 * drag(x(2)); else, acc = u - drag(x(2)); end
            xn = [x(1) - T * x(2) + T * vl(k), x(2) + T * acc + W * (2 * rand - 1)];
            if withMon
                [Mon, v] = ddma.monitor_check(Mon, [x(2) u], xn - x * A_G');
                if ~isempty(v) && isnan(alarm), alarm = k; end
                if Mon.needs_refit             % widened: refit on all stored data
                    [Zs, Ys] = ddma.monitor_data(Mon);
                    S2 = ddma.fit(Zs, Ys, {Xedges{2}, Uedges{1}}, Mon.d_lo, Mon.d_hi, SIGNS);
                    As = ddma.abstraction(Xedges, Uedges, S2, ddma.affine_bounds(A_G), features);
                    [dom2, al] = ddma.safety(As, safe);
                    Mon.model = S2; Mon.needs_refit = false;
                    fprintf('  refit at t = %.1f s: certified safe set %.0f%%\n', (k - 1) * T, ...
                            100 * nnz(dom2) / nnz(safe));
                end
            end
            xp = x; x = xn; worst = min(worst, x(1));
            nbad = nbad + any(x < SAFE_LO | x > SAFE_HI);
        end
        if withMon, name = 'monitor'; else, name = 'no monitor'; end
        fprintf('--- %s, %s: min gap %.1f m, %d steps outside the specification\n', ...
                scen{s}, name, worst, nbad);
        if withMon
            V = Mon.violations;
            fprintf('first alarm at t = %.1f s (event at t = %.1f s), caught by level ''%s''\n%s\n', ...
                    (alarm - 1) * T, KE * T, V(1).level, ddma.monitor_report(Mon));
        end
    end
end
