function M = monitor_for_model(S, Z, Y, opts)
%MONITOR_FOR_MODEL Assumption monitor of a fitted map, seeded with its data (Z, Y).
%   S from ddma.fit (signs), ddma.bd_fit or ddma.growth_map (derivative bounds: [g_lo, g_hi]
%   for the growth methods, [min(g_lo,0), max(g_hi,0)] for the others, 0 outside features).
%   The map's cell bounds are level 1. Monotone maps (ddma.fit) can only widen the
%   disturbance. Same as AssumptionMonitor.for_model.
if nargin < 4, opts = struct(); end
if isfield(S, 'signs')
    if isfield(opts, 'widen') && ~(ischar(opts.widen) && strcmp(opts.widen, 'disturbance'))
        error('ddma:monitor_for_model', ['a monotone model can only widen the disturbance ' ...
              '(widening a sign makes the model non-monotone: use ddma.bd_fit)']);
    end
    s = S.signs;
    m = size(S.upper, 2); n = numel(S.edges);
    if isempty(s), s = ones(m, n); end
    s = s .* ones(m, n);
    lo = zeros(m, n); hi = zeros(m, n);
    lo(s < 0) = -inf; hi(s > 0) = inf;
elseif isfield(S, 'method')
    m = numel(S.raw); n = numel(S.edges);
    lo = zeros(m, n); hi = zeros(m, n);
    growth = any(strcmp(S.method, {'growth', 'lin_growth'}));
    for i = 1:m
        R = S.raw{i};
        if growth
            lo(i, R.feat) = R.gl; hi(i, R.feat) = R.gh;
        else
            lo(i, R.feat) = min(R.gl, 0); hi(i, R.feat) = max(R.gh, 0);
        end
    end
else
    error('ddma:monitor_for_model', 'S must come from ddma.fit, ddma.bd_fit or ddma.growth_map');
end
opts.model = S;
M = ddma.monitor(lo, hi, S.d_lo, S.d_hi, opts);
if nargin >= 3 && ~isempty(Z)
    M = ddma.monitor_add(M, Z, Y);
end
end
