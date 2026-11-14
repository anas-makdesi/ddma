%RL_SAFETY_FILTER A learning agent that never violates its constraints (MATLAB/Octave).
%   MATLAB counterpart of examples/rl_safety_filter.py. The cruise-control safety
%   controller is learned from 200,000 transitions (as in acc_safety_filter.m), then
%   put around the environment's step function with ddma.safe_step / ddma.safe_reset,
%   which have the signature of the Reinforcement Learning Toolbox's rlFunctionEnv.
%   With that toolbox, train any agent on the filtered environment:
%       env = rlFunctionEnv(obsInfo, actInfo, safeStep, safeReset);
%   This script needs no toolbox: it runs two exploring "agents" (random actions, and
%   full throttle) for 20 episodes each, with and without the filter, and counts the
%   violations. Run from the matlab/ folder:  run examples/rl_safety_filter
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));   % the +ddma package
addpath(fileparts(mfilename('fullpath')));                     % cruise_control_env
[step0, reset0, P] = cruise_control_env(400);
% 1. safety controller from data (black-box speed dynamics)
ddma.seed(0);
K = 200000;
X = [5 + 55 * rand(K, 1), 30 * rand(K, 1)]; U = 6 * rand(K, 1) - 3;
vl = P.VL(1) + diff(P.VL) * rand(K, 1);
Xn = [X(:, 1) - P.T * X(:, 2) + P.T * vl, ...
      X(:, 2) + P.T * (U - (P.f0 + P.f1 * X(:, 2) + P.f2 * X(:, 2) .^ 2) / P.M) + P.W * (2 * rand(K, 1) - 1)];
A_G = [1 -P.T; 0 0];
Y = Xn - X * A_G';
Xedges = ddma.uniform_partition([5 0], [60 30], [110 60]);
Uedges = ddma.uniform_partition(-3, 3, 24);
S = ddma.fit([X(:, 2) U], Y, {Xedges{2}, Uedges{1}}, [P.T * P.VL(1), -P.W], [P.T * P.VL(2), P.W]);
A = ddma.abstraction(Xedges, Uedges, S, ddma.affine_bounds(A_G), [2 3]);
[domain, allowed] = ddma.safety(A, ddma.cells_inside(Xedges, P.SAFE_LO, P.SAFE_HI));
fprintf('safety controller: %.1f %% of the specification cells\n', ...
        100 * nnz(domain) / nnz(ddma.cells_inside(Xedges, P.SAFE_LO, P.SAFE_HI)));
% 2. the same agents with and without the filter
safeStep = ddma.safe_step(step0, A, allowed);
safeReset = ddma.safe_reset(reset0, A, allowed);
agents = {@() 6 * rand - 3, @() 3};
names = {'random actions', 'full throttle '};
for a = 1:2
    for filt = [false true]
        if filt, st = safeStep; rs = safeReset; else, st = step0; rs = reset0; end
        viol = 0; steps = 0; nf = 0;
        for ep = 1:20
            [~, L] = rs(); done = false;
            while ~done
                [~, ~, done, L] = st(agents{a}(), L);
                steps = steps + 1; viol = viol + L.Violation;
                if filt, nf = nf + L.Filtered; end
            end
        end
        if filt, tag = 'with filter'; else, tag = 'no filter  '; end
        fprintf('%s, %s: %3d violations in 20 episodes (%5d steps, %5d actions corrected)\n', ...
                names{a}, tag, viol, steps, nf);
        if filt && viol > 0, error('violation with the safety filter'); end
    end
end
