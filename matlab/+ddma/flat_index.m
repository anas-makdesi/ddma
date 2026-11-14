function f = flat_index(shape, sub)
%FLAT_INDEX 1-based column-major linear index of 0-based subscripts (K x n).
stride = [1, cumprod(shape(1:end-1))];
f = sub * stride(:) + 1;
end
