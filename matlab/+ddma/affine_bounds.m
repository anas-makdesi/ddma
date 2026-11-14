function f = affine_bounds(A, B, c)
%AFFINE_BOUNDS Interval image of g(x, u) = A x + B u + c over boxes.
%   f = ddma.affine_bounds(A, B, c) returns a handle [glo, ghi] = f(xlo, xhi, ulo, uhi)
%   (rows are boxes) usable as g_bounds in ddma.abstraction.
if nargin < 2 || isempty(B), B = []; end
if nargin < 3 || isempty(c), c = zeros(size(A, 1), 1); end
f = @(xlo, xhi, ulo, uhi) affine_eval(A, B, c, xlo, xhi, ulo, uhi);
end

function [glo, ghi] = affine_eval(A, B, c, xlo, xhi, ulo, uhi)
mid = 0.5 * (xlo + xhi) * A' + c(:)';
rad = 0.5 * (xhi - xlo) * abs(A)';
if ~isempty(B)
    mid = mid + 0.5 * (ulo + uhi) * B';
    rad = rad + 0.5 * (uhi - ulo) * abs(B)';
end
glo = mid - rad; ghi = mid + rad;
end
