function [Z, Y] = monitor_data(M, quarantined)
%MONITOR_DATA Stored data, e.g. for a refit: clean points only, or all with
%   ddma.monitor_data(M, true).
if nargin < 2, quarantined = false; end
if quarantined, keep = (1:M.K)'; else, keep = find(M.st(1:M.K) ~= 2); end
Z = M.Z(keep, :); Y = M.Y(keep, :);
end
