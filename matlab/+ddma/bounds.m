function [lo, hi] = bounds(S, zq)
%BOUNDS Evaluate a fitted map S (from ddma.fit) at points zq (M x n).
f = ddma.flat_index(S.shape, ddma.locate(S.edges, zq));
lo = S.lower(f, :); hi = S.upper(f, :);
end
