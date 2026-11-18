function s = monitor_cone(t, lo, hi)
%MONITOR_CONE s(t) = hi t (t > 0), lo t (t < 0), 0 (t = 0), entrywise (0 * inf = 0);
%   t (K x n), lo/hi (1 x n). Same values as the C++ and NumPy code.
lo = repmat(lo, size(t, 1), 1); hi = repmat(hi, size(t, 1), 1);
s = zeros(size(t));
p = t > 0; q = t < 0;
s(p) = hi(p) .* t(p);
s(q) = lo(q) .* t(q);
end
