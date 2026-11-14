function wrapped = safe_reset(reset_fcn, A, allowed, state_fcn, max_tries)
%SAFE_RESET Reset function that redraws the initial state until it lies in the domain
%   of the safety controller (A, allowed) from ddma.safety, so that ddma.safe_step
%   guarantees safety from the first step. Signature of the Reinforcement Learning
%   Toolbox (rlFunctionEnv): [InitialObs, LoggedSignals] = wrapped().
%   state_fcn (optional): LoggedSignals -> state (default LoggedSignals.State).
%   Same as the Python SafetyFilterWrapper(reset_in_domain=True).
if nargin < 4 || isempty(state_fcn), state_fcn = @(L) L.State; end
if nargin < 5 || isempty(max_tries), max_tries = 1000; end
wrapped = @() redraw(reset_fcn, A, allowed, state_fcn, max_tries);
end

function [obs, L] = redraw(reset_fcn, A, allowed, state_fcn, max_tries)
for k = 1:max_tries
    [obs, L] = reset_fcn();
    if ~isempty(ddma.controller_input(A, allowed, state_fcn(L))), return; end
end
error('ddma:safe_reset', 'no initial state in the controller domain after %d resets', max_tries);
end
