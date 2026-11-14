function S = intersect(S, T)
%INTERSECT Intersection of two maps from ddma.fit on the same partition, disturbance
%   bounds and sign pattern: the map of the union of both data sets. If S is compact
%   (single), T is rounded outward first. Same as the Python PartitionedSimulatingMap.intersect.
same = isequal(numel(S.edges), numel(T.edges)) && ...
       all(cellfun(@(a, b) isequal(a(:), b(:)), S.edges(:), T.edges(:))) && ...
       isequal(S.d_lo, T.d_lo) && isequal(S.d_hi, T.d_hi) && isequal(S.signs, T.signs);
if ~same, error('ddma:intersect', 'maps must share partition, disturbance bounds and signs'); end
tlo = T.lower; thi = T.upper;
if isa(S.lower, 'single') && ~isa(tlo, 'single')
    tlo = ddma.internal.round_single(tlo, true); thi = ddma.internal.round_single(thi, false);
end
S.lower = max(S.lower, tlo); S.upper = min(S.upper, thi);
end
