function [a, s] = bd_sign_patterns(gl, gh, which)
%BD_SIGN_PATTERNS Slope vectors a (L x n) and signs s of the monotone transforms f - a.z.
%   which = 'two' (approach 1: all-lower and all-upper) or 'all' (every combination;
%   coordinates with zero bounds get one choice). Same order as the Python sign_patterns.
n = numel(gl);
lo = min(gl, 0); hi = max(gh, 0);
zero = gl == 0 & gh == 0;
if strcmp(which, 'two')
    a = [lo; hi]; s = [ones(1, n); -ones(1, n)];
    a(:, zero) = 0; s(:, zero) = 0;
    if isequal(a(1, :), a(2, :)), a = a(1, :); s = s(1, :); end
    return
end
free = find(~zero);
L = 2 ^ numel(free);
a = zeros(L, n); s = zeros(L, n);
for r = 0:L-1  % same order as itertools.product: first free axis varies slowest
    for t = 1:numel(free)
        j = free(t);
        if bitand(r, 2 ^ (numel(free) - t)), a(r + 1, j) = hi(j); s(r + 1, j) = -1;
        else, a(r + 1, j) = lo(j); s(r + 1, j) = 1; end
    end
end
end
