function wrapped = safe_step(step_fcn, A, allowed, state_fcn, penalty)
%SAFE_STEP Certified safety filter around an environment step function.
%   wrapped = ddma.safe_step(step_fcn, A, allowed) returns a step function with the
%   signature of the Reinforcement Learning Toolbox (rlFunctionEnv):
%       [NextObs, Reward, IsDone, LoggedSignals] = wrapped(Action, LoggedSignals)
%   Each action first goes through the safety filter of the controller (A, allowed) from
%   ddma.safety (ddma.controller_input with the action as desired input): a safe action
%   passes unchanged, an unsafe one is replaced by the nearest safe one. As long as the
%   state starts in the controller domain (see ddma.safe_reset), the system stays in the
%   safe set for ever, whatever the agent does, during training as well.
%   state_fcn (optional): handle LoggedSignals -> state x of the abstraction (default:
%   LoggedSignals.State). penalty (optional, default 0): reward penalty per unit of
%   correction. LoggedSignals gains SafeAction, Filtered, Correction, OutsideDomain.
%   Same as the Python ddma.rl.SafetyFilterWrapper.
%
%   With the Reinforcement Learning Toolbox (not needed by ddma):
%       env = rlFunctionEnv(obsInfo, actInfo, ddma.safe_step(@myStep, A, allowed), ...
%                           ddma.safe_reset(@myReset, A, allowed));
%       agent = rlPPOAgent(obsInfo, actInfo); train(agent, env, trainOpts);
if nargin < 4 || isempty(state_fcn), state_fcn = @(L) L.State; end
if nargin < 5 || isempty(penalty), penalty = 0; end
wrapped = @(action, L) filtered_step(step_fcn, A, allowed, state_fcn, penalty, action, L);
end

function [obs, reward, done, L] = filtered_step(step_fcn, A, allowed, state_fcn, penalty, action, L)
u = double(action(:)');
x = state_fcn(L);
s = ddma.controller_input(A, allowed, x, u);
outside = isempty(s);
if outside, s = u; end
corr = norm(s - u);
[obs, reward, done, L] = step_fcn(reshape(s, size(action)), L);
L.SafeAction = s; L.Filtered = corr > 0; L.Correction = corr; L.OutsideDomain = outside;
reward = reward - penalty * corr;
end
