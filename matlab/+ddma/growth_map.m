function S = growth_map(Z, y, edges, g_lo, g_hi, d_lo, d_hi, local, final, backend)
%GROWTH_MAP Growth-cone model of one scalar function with bounded derivatives.
%   S = ddma.growth_map(Z, y, edges, g_lo, g_hi, d_lo, d_hi) for an unknown f with
%   df/dz_j in [g_lo(j), g_hi(j)] and data y(k) = f(Z(k, :)) + d, d in [d_lo, d_hi]:
%   for every cell an interval containing f(z) + d for all z in the cell. local =
%   'slab' (default, tightest), 'cell' or ''; final = 'box' (default, approach 4) or
%   'lp' (approach 5). This is ddma.bd_fit with method 'growth' / 'lin_growth' for one
%   output, so ddma.bd_update, ddma.bd_update_local, ddma.bounds and ddma.f_bounds apply.
%   Same as the Python GrowthMap.
if nargin < 8, local = 'slab'; end
if nargin < 9 || isempty(final), final = 'box'; end
if nargin < 10, backend = 'auto'; end
switch final
    case 'box', method = 'growth';
    case 'lp',  method = 'lin_growth';
    otherwise, error('ddma:growth_map', 'final must be ''box'' or ''lp''');
end
S = ddma.bd_fit(Z, y(:), edges, g_lo(:)', g_hi(:)', d_lo, d_hi, method, local, backend);
end
