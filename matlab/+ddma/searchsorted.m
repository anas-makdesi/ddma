function idx = searchsorted(e, z, side)
%SEARCHSORTED Number of edges e < z (side 'left') or e <= z (side 'right').
%   Same result as numpy.searchsorted(e, z, side). e must be increasing.
%   Works in MATLAB and Octave.
if nargin < 3, side = 'right'; end
e = e(:); z = z(:);
ne = numel(e);
if strcmp(side, 'right')
    key = [e, zeros(ne, 1); z, ones(numel(z), 1)];   % ties: edges first
else
    key = [e, ones(ne, 1); z, zeros(numel(z), 1)];   % ties: points first
end
[~, ord] = sortrows(key);
isEdge = ord <= ne;
cnt = cumsum(isEdge);
idx = zeros(numel(z), 1);
idx(ord(~isEdge) - ne) = cnt(~isEdge);
end
