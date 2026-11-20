function [step_fcn, reset_fcn, P] = cruise_control_env(horizon)
%CRUISE_CONTROL_ENV Adaptive cruise control as an RL environment (step/reset functions
%   with the signature of the Reinforcement Learning Toolbox's rlFunctionEnv).
%   Same environment as the Python ddma.rl.CruiseControlEnv: state (h, v), action
%   u in [-3, 3]; leader speed switching between about 13 and 17 m/s (not observed);
%   reward 1 - ((v - 22) / 10)^2 per step, -50 and end of episode when the state leaves
%   h in [10, 50], v in [2, 28]; observation [h; v; h - h_prev]. LoggedSignals.Violation
%   flags a violating step.
if nargin < 1, horizon = 400; end
P = struct('T', 0.5, 'M', 1370, 'f0', 51, 'f1', 1.2567, 'f2', 0.4342, 'VL', [13 17], ...
           'W', 0.05, 'SAFE_LO', [10 2], 'SAFE_HI', [50 28], 'V_REF', 22, ...
           'horizon', horizon, 'init_lo', [20 12], 'init_hi', [40 18]);
step_fcn = @(u, L) env_step(P, u, L);
reset_fcn = @() env_reset(P);
end

function [obs, L] = env_reset(P)
L = struct('State', P.init_lo + rand(1, 2) .* (P.init_hi - P.init_lo), 'k', 0, 'dh', 0, ...
           'phase', ddma.internal.rand_int(20, 59), 'fast', rand < 0.5, 'Violation', false);
obs = [L.State(:); 0];
end

function [obs, reward, done, L] = env_step(P, u, L)
u = min(max(double(u(1)), -3), 3);
if mod(L.k, L.phase) == 0 && L.k > 0, L.fast = ~L.fast; end
if L.fast, vl = P.VL(2); else, vl = P.VL(1); end
vl = min(max(vl + (rand - 0.5), P.VL(1)), P.VL(2));
h = L.State(1); v = L.State(2);
hn = h - P.T * v + P.T * vl;
vn = v + P.T * (u - (P.f0 + P.f1 * v + P.f2 * v ^ 2) / P.M) + P.W * (2 * rand - 1);
L.State = [hn vn]; L.dh = hn - h; L.k = L.k + 1;
L.Violation = any(L.State < P.SAFE_LO | L.State > P.SAFE_HI);
if L.Violation, reward = -50; else, reward = 1 - ((vn - P.V_REF) / 10) ^ 2; end
done = L.Violation || L.k >= P.horizon;
obs = [hn; vn; L.dh];
end
