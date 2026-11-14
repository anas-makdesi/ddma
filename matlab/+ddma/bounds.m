function [lo, hi] = bounds(S, zq)
%BOUNDS Evaluate a fitted map S (from ddma.fit or ddma.bd_fit) at points zq (M x n):
%   bounds on f(z) + d at the cells of the points.
[lo, hi] = ddma.internal.map_bounds_at(S, ddma.locate(S.edges, zq));
end
