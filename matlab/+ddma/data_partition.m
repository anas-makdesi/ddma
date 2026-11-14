function edges = data_partition(Z, lo, hi)
%DATA_PARTITION Data-induced partition of [lo, hi]: one breakpoint per distinct data
%   coordinate inside the box. On it the fixed-partition map of ddma.fit coincides with
%   the minimal simulating map (almost everywhere). Up to (K + 1)^n cells: only for small
%   data sets. Same as the Python RectangularPartition.from_data.
%   (Other partitions: ddma.uniform_partition, ddma.margin_partition; the product of two
%   partitions is the concatenation of their edge cell arrays, [E1(:); E2(:)].)
n = size(Z, 2);
edges = cell(n, 1);
for j = 1:n
    v = unique(Z(:, j));
    v = v(v > lo(j) & v < hi(j));
    edges{j} = [lo(j); v(:); hi(j)];
end
end
