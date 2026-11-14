function c = scenario_confidence(n_w_pairs, beta)
%SCENARIO_CONFIDENCE Proposition 9 (TAC): with probability >= 1 - beta, a fresh pair of
%   samples satisfies the bounds of ddma.estimate_derivative_bounds with probability
%   >= beta^(1 / n_w_pairs). Same as the Python scenario_confidence.
c = beta ^ (1 / n_w_pairs);
end
