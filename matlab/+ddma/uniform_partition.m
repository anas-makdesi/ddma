function edges = uniform_partition(lo, hi, ncells)
%UNIFORM_PARTITION Edges of a uniform rectangular partition of [lo, hi].
%   edges = ddma.uniform_partition(lo, hi, ncells) returns a cell array with one
%   increasing edge vector per coordinate (ncells(j) cells along coordinate j).
lo = lo(:); hi = hi(:);
if isscalar(ncells), ncells = ncells * ones(numel(lo), 1); end
edges = cell(numel(lo), 1);
for j = 1:numel(lo)
    edges{j} = ddma.internal.linspace_np(lo(j), hi(j), ncells(j) + 1);
end
end
