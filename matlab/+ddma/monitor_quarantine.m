function Q = monitor_quarantine(M)
%MONITOR_QUARANTINE The quarantined points: struct with ids (seed: negative; checked:
%   transition index, 1-based), Z, Y and the violations that put them there.
q = find(M.st(1:M.K) == 2);
Q.ids = M.ids(q); Q.Z = M.Z(q, :); Q.Y = M.Y(q, :);
if isempty(M.violations), Q.violations = M.violations;
else, Q.violations = M.violations(ismember([M.violations.index], Q.ids)); end
end
