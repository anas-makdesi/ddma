function M = monitor_accumulate(M, i, dz, dy)
%MONITOR_ACCUMULATE Update the needed changes of output i with violating pairs:
%   dz = z_a - z_b (P x n), dy = y_a(i) - y_b(i) (P x 1), a = the point above.
th = M.d_hi(i) - M.d_lo(i);
s = ddma.internal.monitor_cone(dz, M.slope_lo(i, :), M.slope_hi(i, :));
total = sum(s, 2);
r = dy - th - total;
M.need_width(i) = max(M.need_width(i), th + max(r));
rest = (dy - th - total) + s;           % (P x n): gap without coordinate j's term
ratio = rest ./ dz;
for j = 1:M.n
    cur = M.need_hi(i, j);
    if all(dz(:, j) > 0) && ~isnan(cur)
        M.need_hi(i, j) = max(cur, max(ratio(:, j)));
    else
        M.need_hi(i, j) = nan;
    end
    cur = M.need_lo(i, j);
    if all(dz(:, j) < 0) && ~isnan(cur)
        M.need_lo(i, j) = min(cur, min(ratio(:, j)));
    else
        M.need_lo(i, j) = nan;
    end
end
end
