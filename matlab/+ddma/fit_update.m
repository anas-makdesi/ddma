function S = fit_update(S, Z, Y)
%FIT_UPDATE Refine a map from ddma.fit with new data, without revisiting the old data:
%   the map of the union of both data sets is the intersection of the two maps (TAC
%   Prop. 3). A compact (single) map stays compact, rounded outward.
%   Same as the Python PartitionedSimulatingMap.update.
new = ddma.fit(Z, Y, S.edges, S.d_lo, S.d_hi, S.signs);
S = ddma.intersect(S, new);
end
