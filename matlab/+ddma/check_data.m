function R = check_data(Z, Y, slope_lo, slope_hi, d_lo, d_hi, opts)
%CHECK_DATA Check every pair of data points against the assumptions, before learning.
%   R = ddma.check_data(Z, Y, slope_lo, slope_hi, d_lo, d_hi, opts)
%   R = ddma.check_data(Z, Y, [], [], d_lo, d_hi, struct('signs', signs))
%   Needs only the data and the assumptions (no model): slope intervals (m x n) or a sign
%   pattern (opts.signs, as for ddma.fit). Exact: every conflicting pair is found.
%   opts: signs, tol (1e-9), prune (true: skip grid cells where no pair can conflict;
%   same result), backend ('auto' | 'mex' | 'matlab'), input_names, output_names.
%   R: ok, n_points, pairs_evaluated, violating (1 x m conflicting pairs per output),
%   worst (1 x m), worst_pair (m x 2, 1-based [later, earlier], nan if none), conflicts
%   (K x 1 partners each point conflicts with), suspects (points with conflicts, most
%   first), needed (as ddma.monitor_needed), report (text). Same as Python ddma.check_data.
if nargin < 7, opts = struct(); end
o = struct('signs', [], 'tol', 1e-9, 'prune', true, 'backend', 'auto', ...
           'input_names', {{}}, 'output_names', {{}});
f = fieldnames(opts);
for k = 1:numel(f)
    if ~isfield(o, f{k}), error('ddma:check_data', 'unknown option %s', f{k}); end
    o.(f{k}) = opts.(f{k});
end
[K, n] = size(Z); m = size(Y, 2);
if size(Y, 1) ~= K, error('ddma:check_data', 'Z and Y must have the same number of rows'); end
if ~all(isfinite([Z(:); Y(:)])), error('ddma:check_data', 'data must be finite'); end
if ~isempty(o.signs) || isempty(slope_lo)
    s = o.signs; if isempty(s), s = ones(m, n); end
    s = s .* ones(m, n);
    slope_lo = zeros(m, n); slope_hi = zeros(m, n);
    slope_lo(s < 0) = -inf; slope_hi(s > 0) = inf;
end
if any(size(slope_lo) ~= [m n]) || any(size(slope_hi) ~= [m n])
    error('ddma:check_data', 'slope bounds must be %d x %d', m, n);
end
if any(slope_lo(:) > slope_hi(:)) || any(isnan([slope_lo(:); slope_hi(:)])) || ...
        any(slope_lo(:) == inf) || any(slope_hi(:) == -inf)
    error('ddma:check_data', 'need slope_lo <= slope_hi, slope_lo < inf, slope_hi > -inf');
end
th = d_hi(:)' .* ones(1, m) - d_lo(:)' .* ones(1, m);
if any(th < 0), error('ddma:check_data', 'disturbance bounds must satisfy d_lo <= d_hi'); end
use_mex = ddma.internal.monitor_use_mex(o.backend);
[N, count, worst, pair, conflicts, pairs] = ddma.internal.monitor_k_all(slope_lo, slope_hi, th, ...
    o.tol, Z, Y, o.prune, use_mex);
[w, nl, nh] = ddma.internal.monitor_merge(N, th, slope_lo, slope_hi, th);
pair(pair < 0) = nan;
R = struct('ok', ~any(count), 'n_points', K, 'pairs_evaluated', pairs, 'violating', count, ...
           'worst', worst, 'worst_pair', pair + 1, 'conflicts', conflicts);
[~, ord] = sort(-conflicts(conflicts > 0));
idx = find(conflicts > 0);
R.suspects = idx(ord);
R.needed = struct('width', w, 'slope_lo', nl, 'slope_hi', nh);
R.report = report(R, th, slope_lo, slope_hi, o);
end

function txt = report(R, th, slope_lo, slope_hi, o)
nm.input_names = o.input_names; nm.output_names = o.output_names;
out = @(i) ddma.internal.monitor_name(nm, 'out', i);
inp = @(j) ddma.internal.monitor_name(nm, 'in', j);
if R.ok
    L = {sprintf('Data check: %d points, all consistent with the assumptions.', R.n_points)};
else
    L = {sprintf('Data check: %d points, %d involved in pairs that contradict the assumptions.', ...
                 R.n_points, sum(R.conflicts > 0))};
end
for i = 1:numel(R.violating)
    if ~R.violating(i), continue; end
    L{end + 1} = sprintf('%s: %d conflicting pairs, worst %.4g beyond the assumptions (points %d and %d).', ...
        out(i), R.violating(i), R.worst(i), R.worst_pair(i, 1), R.worst_pair(i, 2)); %#ok<AGROW>
    L = [L, ddma.internal.monitor_explain(i, R.needed, th, slope_lo, slope_hi, o.tol, out, inp)]; %#ok<AGROW>
end
if ~R.ok
    s = R.suspects(1:min(5, end));
    parts = arrayfun(@(k) sprintf('point %d (%d)', k, R.conflicts(k)), s(:)', 'UniformOutput', false);
    L{end + 1} = ['Most conflicts: ' strjoin(parts, ', ') '.'];
end
txt = strjoin(L, sprintf('\n'));
end
