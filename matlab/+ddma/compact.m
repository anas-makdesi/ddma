function S = compact(S)
%COMPACT Store the bounds of a map from ddma.fit as single (half the memory), rounded
%   outward (lower bounds down, upper bounds up), so the map stays sound.
%   Same as the Python PartitionedSimulatingMap.compact / fit(..., compact=True).
S.lower = ddma.internal.round_single(S.lower, true);
S.upper = ddma.internal.round_single(S.upper, false);
end
