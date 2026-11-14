function [xs, us] = simulate(step, A, allowed, x0, n_steps, pick, stop_in, nominal)
%SIMULATE Closed-loop simulation of the concrete system x+ = step(x, u).
%   [xs, us] = ddma.simulate(step, A, allowed, x0, n_steps) with the controller (A,
%   allowed) from ddma.safety / ddma.reach_avoid. pick: 'center' (default), 'random';
%   nominal (optional handle x -> desired input): the controller acts as a safety filter
%   around it (ddma.controller_input with the desired input). Stops early if the state
%   leaves the controller domain, or once it enters stop_in (logical state-cell mask).
%   xs: (T+1) x nx states, us: T x nu inputs. Same as the Python simulate.
if nargin < 6 || isempty(pick), pick = 'center'; end
if nargin < 7, stop_in = []; end
if nargin < 8, nominal = []; end
Xlo = cellfun(@(e) e(1), A.Xedges(:)'); Xhi = cellfun(@(e) e(end), A.Xedges(:)');
x = x0(:)';
xs = x; us = zeros(0, numel(A.Uedges));
for k = 1:n_steps
    if ~isempty(stop_in) && all(x >= Xlo & x <= Xhi)
        q = ddma.flat_index(A.shapeX, ddma.locate(A.Xedges, x));
        if stop_in(q), break; end
    end
    if isempty(nominal)
        u = ddma.controller_input(A, allowed, x, pick);
    else
        u = ddma.controller_input(A, allowed, x, nominal(x));
    end
    if isempty(u), break; end
    x = step(x, u); x = x(:)';
    us(end + 1, :) = u; xs(end + 1, :) = x; %#ok<AGROW>
end
end
