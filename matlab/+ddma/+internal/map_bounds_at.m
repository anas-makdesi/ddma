function [lo, hi] = map_bounds_at(S, sub)
%MAP_BOUNDS_AT Bounds (k x m) of a map from ddma.fit / ddma.bd_fit at the cells with
%   0-based subscripts sub (k x n, in S's coordinates). Handles per-output features.
if isfield(S, 'features') && ~isempty(S.features)
    m = numel(S.raw);
    lo = zeros(size(sub, 1), m); hi = lo;
    for i = 1:m
        R = S.raw{i};
        f = ddma.flat_index(R.shape, sub(:, R.feat));
        lo(:, i) = R.lo(f); hi(:, i) = R.up(f);
    end
else
    f = ddma.flat_index(S.shape, sub);
    lo = S.lower(f, :); hi = S.upper(f, :);
end
lo = double(lo); hi = double(hi);
end
