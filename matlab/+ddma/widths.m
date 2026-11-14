function w = widths(S)
%WIDTHS upper - lower per cell and output (Inf where unbounded), N x m.
if isfield(S, 'features') && ~isempty(S.features)
    w = cellfun(@(R) R.up - R.lo, S.raw, 'UniformOutput', false);   % one vector per output
else
    w = double(S.upper) - double(S.lower);
end
end
