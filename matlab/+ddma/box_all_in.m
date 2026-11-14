function good = box_all_in(A, W, backend)
%BOX_ALL_IN good(q, p): input p enabled at q and all successor cells of (q, p) are in W.
%   backend: 'auto' (MEX if compiled, see build_mex) | 'mex' | 'matlab'. Same result.
if isfield(A, 'lazy') && A.lazy
    error('ddma:box_all_in', 'box_all_in needs an eager abstraction (lazy = false)');
end
if nargin < 3, backend = 'auto'; end
if strcmp(backend, 'mex') || (strcmp(backend, 'auto') && exist('ddma_synth_mex', 'file') == 3)
    good = ddma_synth_mex('box_all_in', A.first, A.last, A.enabled, A.shapeX, logical(W(:)));
    return
end
shX = A.shapeX; nx = numel(shX);
sat = zeros([shX + 1, 1]);
idx = arrayfun(@(s) 2:s + 1, shX, 'UniformOutput', false);
sat(idx{:}) = reshape(~W, [shX, 1]);
for j = 1:nx, sat = cumsum(sat, j); end
stride = [1, cumprod(shX(1:end-1) + 1)];
F = double(A.first); L = double(A.last) + 1;
cnt = zeros(size(A.enabled));
for c = 0:2^nx - 1
    lin = ones(size(A.enabled));
    nUp = 0;
    for j = 1:nx
        if bitget(c, j), v = L(:, :, j); nUp = nUp + 1; else, v = F(:, :, j); end
        lin = lin + v * stride(j);
    end
    sgn = (-1) ^ (nx - nUp);
    cnt = cnt + sgn * sat(lin);
end
good = A.enabled & (cnt == 0);
end
