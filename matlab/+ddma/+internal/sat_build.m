function sat = sat_build(shX, W)
%SAT_BUILD n-D summed-area table of the complement of W (cells column-major), padded
%   with a leading zero slice on every axis.
sat = zeros([shX + 1, 1]);
idx = arrayfun(@(s) 2:s + 1, shX, 'UniformOutput', false);
sat(idx{:}) = reshape(~logical(W(:)), [shX, 1]);
for j = 1:numel(shX), sat = cumsum(sat, j); end
end
