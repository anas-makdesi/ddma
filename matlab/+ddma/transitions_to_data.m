function [Z, Y] = transitions_to_data(X, U, Xnext, g)
%TRANSITIONS_TO_DATA Data (z_k, y_k) = ((x_k, u_k), x_k' - g(x_k, u_k)) for learning f + W.
%   g (optional): handle g(X, U) -> K x nx of the known part. Same as the Python function.
Y = Xnext;
if nargin >= 4 && ~isempty(g), Y = Xnext - g(X, U); end
Z = [X, U];
end
