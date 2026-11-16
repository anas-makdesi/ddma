function s = monitor_cone(dz, lo, hi)
%MONITOR_CONE s(t) = max(hi t, lo t) per entry with 0 * inf = 0; dz (K x n), lo/hi (1 x n).
lo = repmat(lo, size(dz, 1), 1); hi = repmat(hi, size(dz, 1), 1);
s = zeros(size(dz));
p = dz > 0; q = dz < 0;
s(p) = hi(p) .* dz(p);
s(q) = lo(q) .* dz(q);
end
