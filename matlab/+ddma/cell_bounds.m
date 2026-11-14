function [lo, hi] = cell_bounds(edges, sub)
%CELL_BOUNDS Lower and upper corners of cells with 0-based subscripts sub (K x n).
n = numel(edges);
lo = zeros(size(sub, 1), n); hi = lo;
for j = 1:n
    e = edges{j};
    lo(:, j) = e(sub(:, j) + 1);
    hi(:, j) = e(sub(:, j) + 2);
end
end
