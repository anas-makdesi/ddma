function y = linspace_np(a, b, n)
%LINSPACE_NP n points from a to b, computed exactly as NumPy's linspace (column vector).
%   start + k * ((b - a) / (n - 1)), with the last point set to b. Used for partition edges
%   instead of linspace, whose last-bit rounding differs between Octave versions and MATLAB:
%   a data point on an edge must fall in the same cell everywhere, and as in Python.
if n == 1, y = a; return; end
step = (b - a) / (n - 1);
k = (0:n - 1)';
if step == 0
    y = k / (n - 1) * (b - a) + a;
else
    y = k * step + a;
end
y(end) = b;
end
