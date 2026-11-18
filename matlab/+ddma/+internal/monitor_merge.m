function [width, nlo, nhi] = monitor_merge(N, width, nlo, nhi, th)
%MONITOR_MERGE Fold a kernel's record N into the needed values (nan = ruled out for good).
for i = find(N.any)
    width(i) = max(width(i), th(i) + N.w(i));
    for j = 1:size(nhi, 2)
        if N.hio(i, j) || isnan(nhi(i, j)), nhi(i, j) = nan;
        else, nhi(i, j) = max(nhi(i, j), N.hi(i, j)); end
        if N.loo(i, j) || isnan(nlo(i, j)), nlo(i, j) = nan;
        else, nlo(i, j) = min(nlo(i, j), N.lo(i, j)); end
    end
end
end
