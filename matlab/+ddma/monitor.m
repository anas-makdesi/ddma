function M = monitor(slope_lo, slope_hi, d_lo, d_hi, opts)
%MONITOR Online check of new data against a model's assumptions (signs, derivative
%   bounds, disturbance bounds). Same as the Python ddma.AssumptionMonitor.
%
%   M = ddma.monitor(slope_lo, slope_hi, d_lo, d_hi, opts)
%   slope_lo, slope_hi (m x n): interval of df_i/dz_j assumed by the model (-inf / inf
%   allowed): sign +1 -> [0, inf], sign -1 -> [-inf, 0], sign 0 or not a feature ->
%   [0, 0], bounded derivatives -> [g_lo, g_hi] (see ddma.monitor_monotone and
%   ddma.monitor_for_model). d_lo, d_hi: disturbance bounds (scalars or 1 x m).
%
%   Two data points a, b are consistent with the assumptions iff for every output i
%       y_a(i) - y_b(i) <= (d_hi(i) - d_lo(i)) + sum_j s_ij(z_a(j) - z_b(j)),
%       s_ij(t) = max(slope_hi(i,j) t, slope_lo(i,j) t).
%   ddma.monitor_check checks each new transition against all stored data with this
%   condition (and against the model's cell bounds if opts.model is given). A violation
%   proves that some assumption is wrong. ddma.monitor_needed gives, for each assumption,
%   the smallest change that alone makes all stored data consistent (exact); nan = it
%   cannot explain the violations. ddma.monitor_report summarises in plain words.
%
%   opts (struct, all optional):
%     on_violation  'alarm' (default) | 'widen' | 'fallback'
%     widen         'disturbance' (default) | {'slope_lo', i, j} | {'slope_hi', i, j}
%     margin        relative extra widening (default 0)
%     max_points    keep at most this many data points (random sample); default inf
%     model         fitted map (ddma.fit / ddma.bd_fit / ddma.growth_map) whose cell
%                   bounds are also checked (inside its grid); set by ddma.monitor_for_model
%     tol           violations below tol are ignored (default 1e-9)
%     callback      function handle called with the violations of a transition
%     input_names, output_names   cell arrays of names for the report
%   Indices are 1-based (transitions in the order checked; seed data -1, -2, ...).
if nargin < 5, opts = struct(); end
if any(size(slope_lo) ~= size(slope_hi)), error('ddma:monitor', 'slope_lo and slope_hi must have the same size'); end
if any(slope_lo(:) > slope_hi(:)) || any(isnan([slope_lo(:); slope_hi(:)]))
    error('ddma:monitor', 'slope bounds must satisfy slope_lo <= slope_hi');
end
[m, n] = size(slope_lo);
d_lo = d_lo(:)' .* ones(1, m); d_hi = d_hi(:)' .* ones(1, m);
if any(d_lo > d_hi), error('ddma:monitor', 'disturbance bounds must satisfy d_lo <= d_hi'); end
o = struct('on_violation', 'alarm', 'widen', 'disturbance', 'margin', 0, 'max_points', inf, ...
           'model', [], 'tol', 1e-9, 'callback', [], 'input_names', {{}}, 'output_names', {{}});
f = fieldnames(opts);
for k = 1:numel(f), o.(f{k}) = opts.(f{k}); end
if ~any(strcmp(o.on_violation, {'alarm', 'widen', 'fallback'}))
    error('ddma:monitor', 'on_violation must be ''alarm'', ''widen'' or ''fallback''');
end
ddma.internal.monitor_target(o.widen);
M = o;
M.slope_lo = slope_lo; M.slope_hi = slope_hi; M.d_lo = d_lo; M.d_hi = d_hi;
M.m = m; M.n = n;
M.Z = zeros(0, n); M.Y = zeros(0, m); M.ids = zeros(0, 1);
M.n_seen = 0; M.n_seed = 0; M.n_checked = 0;
M.violations = struct('index', {}, 'output', {}, 'kind', {}, 'gap', {}, 'witness', {});
M.violating = zeros(0, 1);
M.changes = {};
M.fallback_active = false; M.fallback_since = [];
M.needs_refit = false;
M.need_width = d_hi - d_lo; M.need_lo = slope_lo; M.need_hi = slope_hi;
end
