function c = bd_cells_of(edges, Z)
%BD_CELLS_OF 0-based cell of each point; points outside the box: nearest edge cell.
n = numel(edges);
c = zeros(size(Z, 1), n);
for j = 1:n
    e = edges{j};
    c(:, j) = min(max(ddma.searchsorted(e, Z(:, j), 'right') - 1, 0), numel(e) - 2);
end
end
