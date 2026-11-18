function M = monitor(slope_lo, slope_hi, d_lo, d_hi, opts)
%MONITOR Check data against a model's assumptions (signs, derivative bounds, disturbance
%   bounds) while the system runs. Same as the Python ddma.AssumptionMonitor.
%
%   M = ddma.monitor(slope_lo, slope_hi, d_lo, d_hi, opts)
%   slope_lo, slope_hi (m x n): interval of df_i/dz_j assumed by the model (-inf / inf
%   allowed): sign +1 -> [0, inf], sign -1 -> [-inf, 0], sign 0 or not a feature ->
%   [0, 0], bounded derivatives -> [g_lo, g_hi] (see ddma.monitor_monotone and
%   ddma.monitor_for_model). d_lo, d_hi: disturbance bounds (scalars or 1 x m).
%
%   Two data points a, b are consistent with the assumptions iff for every output i
%       y_a(i) - y_b(i) <= (d_hi(i) - d_lo(i)) + sum_j s_ij(z_a(j) - z_b(j)),
%       s_ij(t) = slope_hi(i,j) t (t > 0), slope_lo(i,j) t (t < 0), 0 (t = 0).
%   A violation proves that some assumption is wrong. ddma.monitor_check checks each new
%   transition at three levels: 'cells' (the model's cell bounds, if opts.model),
%   'sample' (a fresh random sample of stored points; only with full = 'later') and
%   'full' (every earlier point). A point that fails is quarantined: kept for the
%   diagnosis, but later points are not blamed for disagreeing with it.
%   ddma.monitor_needed gives, for each assumption, the smallest change that alone makes
%   all stored data consistent; ddma.monitor_report summarises in plain words.
%
%   opts (struct, all optional):
%     on_violation  'alarm' (default) | 'widen' | 'fallback' | 'widen_then_fallback'
%     widen         'disturbance' (default) | {'slope_lo', i, j} | {'slope_hi', i, j}
%                   (a slope bound only if both of its ends are finite, with slope_limit)
%     widen_limit   'widen_then_fallback': largest disturbance width as a multiple of
%                   the original width (default 2; scalar or 1 x m); beyond: fallback
%     slope_limit   absolute limit of a widened slope bound (required for a slope target)
%     margin        relative extra widening (default 0)
%     full          'step' (default): full check of each point in its step; 'later':
%                   levels 1-2 during the run, ddma.monitor_flush checks the rest
%     sample        level 2: points per step (default 2000; full = 'later' only)
%     prune         skip grid cells where no pair can conflict (default true; same results)
%     model         fitted map (ddma.fit / ddma.bd_fit / ddma.growth_map) whose cell
%                   bounds are level 1; set by ddma.monitor_for_model
%     tol           violations below tol are ignored (default 1e-9)
%     callback      function handle called with the violations found at once
%     input_names, output_names   cell arrays of names for the report
%     seed          seed of the level-2 samples (default 0; same samples as Python)
%     backend       'auto' (MEX if built) | 'mex' | 'matlab' (same results)
%   Indices are 1-based (transitions in the order checked; seed data -1, -2, ...).
if nargin < 5, opts = struct(); end
if any(size(slope_lo) ~= size(slope_hi)), error('ddma:monitor', 'slope_lo and slope_hi must have the same size'); end
if any(slope_lo(:) > slope_hi(:)) || any(isnan([slope_lo(:); slope_hi(:)]))
    error('ddma:monitor', 'slope bounds must satisfy slope_lo <= slope_hi');
end
if any(slope_lo(:) == inf) || any(slope_hi(:) == -inf)
    error('ddma:monitor', 'slope_lo must be < inf and slope_hi > -inf');
end
[m, n] = size(slope_lo);
d_lo = d_lo(:)' .* ones(1, m); d_hi = d_hi(:)' .* ones(1, m);
if any(d_lo > d_hi), error('ddma:monitor', 'disturbance bounds must satisfy d_lo <= d_hi'); end
o = struct('on_violation', 'alarm', 'widen', 'disturbance', 'widen_limit', 2, ...
           'slope_limit', [], 'margin', 0, 'full', 'step', 'sample', 2000, 'prune', true, ...
           'model', [], 'tol', 1e-9, 'callback', [], 'input_names', {{}}, ...
           'output_names', {{}}, 'seed', 0, 'backend', 'auto');
f = fieldnames(opts);
for k = 1:numel(f)
    if ~isfield(o, f{k}), error('ddma:monitor', 'unknown option %s', f{k}); end
    o.(f{k}) = opts.(f{k});
end
if ~any(strcmp(o.on_violation, {'alarm', 'widen', 'fallback', 'widen_then_fallback'}))
    error('ddma:monitor', 'on_violation must be ''alarm'', ''widen'', ''fallback'' or ''widen_then_fallback''');
end
if ~any(strcmp(o.full, {'step', 'later'})), error('ddma:monitor', 'full must be ''step'' or ''later'''); end
M = o;
M.slope_lo = slope_lo; M.slope_hi = slope_hi; M.d_lo = d_lo; M.d_hi = d_hi;
M.m = m; M.n = n;
M.widen_limit = o.widen_limit(:)' .* ones(1, m);
ddma.internal.monitor_target(M, o.widen);
M.use_mex = ddma.internal.monitor_use_mex(o.backend);
% storage with spare capacity: data, ids, status (0 clean, 2 quarantined)
M.Z = zeros(64, n); M.Y = zeros(64, m); M.ids = zeros(64, 1); M.st = zeros(64, 1, 'uint8');
M.K = 0; M.n_seed = 0; M.n_checked = 0;
M.pending_from = 0;   % points after this one wait for their full check
M.grid = []; M.grid_K = 0;
M.violations = struct('index', {}, 'output', {}, 'level', {}, 'gap', {}, 'witness', {}, 'found_at', {});
M.violating = zeros(0, 1);
M.changes = {}; M.notes = {};
M.fallback_active = false; M.fallback_since = []; M.fallback_reason = '';
M.needs_refit = false;
M.theta0 = d_hi - d_lo;
M.rng_x = mod(o.seed, 2147483646) + 1; M.rng_pw = [];
M = ddma.monitor_reset(M);
end
