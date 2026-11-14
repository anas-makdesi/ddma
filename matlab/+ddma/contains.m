function ok = contains(S, z, y, tol)
%CONTAINS Whether each pair (z(k, :), y(k, :)) satisfies y in the interval of the cell
%   of z (S from ddma.fit or ddma.bd_fit). Same as the Python PartitionedSimulatingMap.contains.
if nargin < 4, tol = 1e-12; end
[lo, hi] = ddma.bounds(S, z);
ok = all(lo - tol <= y & y <= hi + tol, 2);
end
