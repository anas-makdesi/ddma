function q = locate(edges, Z)
%LOCATE Cell subscripts (0-based, K x n) of the points Z (K x n).
%   Cells are [e(c), e(c+1)); the last cell along each axis is closed. Points must lie in
%   the partitioned box.
n = numel(edges);
q = zeros(size(Z, 1), n);
for j = 1:n
    e = edges{j};
    if any(Z(:, j) < e(1) | Z(:, j) > e(end))
        error('ddma:locate', 'points outside the partitioned box');
    end
    q(:, j) = min(max(ddma.searchsorted(e, Z(:, j), 'right') - 1, 0), numel(e) - 2);
end
end
