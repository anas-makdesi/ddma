function edges = margin_partition(lo, hi, inner_lo, inner_hi, ninner)
%MARGIN_PARTITION Uniform cells on [inner_lo, inner_hi] plus one margin cell per side.
%   Edges per coordinate: [lo, linspace(inner_lo, inner_hi, ninner+1), hi]. This is the
%   partition used in the experiments of the TAC and ADHS papers.
lo = lo(:); hi = hi(:); inner_lo = inner_lo(:); inner_hi = inner_hi(:);
if isscalar(ninner), ninner = ninner * ones(numel(lo), 1); end
edges = cell(numel(lo), 1);
for j = 1:numel(lo)
    e = ddma.internal.linspace_np(inner_lo(j), inner_hi(j), ninner(j) + 1);
    if lo(j) < inner_lo(j), e = [lo(j); e]; end
    if hi(j) > inner_hi(j), e = [e; hi(j)]; end
    edges{j} = e;
end
end
