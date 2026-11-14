function sub = all_subs(shape)
%ALL_SUBS 0-based subscripts of all cells, in column-major order (prod(shape) x n).
N = prod(shape); n = numel(shape);
sub = zeros(N, n);
r = (0:N-1)';
for j = 1:n
    sub(:, j) = mod(r, shape(j));
    r = floor(r / shape(j));
end
end
