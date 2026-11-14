function sub = all_subs_of(shape, f)
%ALL_SUBS_OF 0-based subscripts of the cells with 1-based linear indices f.
r = f(:) - 1; n = numel(shape);
sub = zeros(numel(r), n);
for j = 1:n
    sub(:, j) = mod(r, shape(j));
    r = floor(r / shape(j));
end
end
