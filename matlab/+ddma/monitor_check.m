function [M, viol] = monitor_check(M, Z, Y)
%MONITOR_CHECK Check new transitions one by one, store them and respond. viol: struct
%   array with fields index (1-based position among the checked transitions), output,
%   level ('cells', 'sample' or 'full'), gap, witness (id of the stored point it
%   conflicts with: seed data -1, -2, ..., checked points their index; nan for 'cells')
%   and found_at (index of the last checked transition when found). With full = 'later'
%   only levels 1-2 run here; ddma.monitor_flush runs level 3.
%   Same as AssumptionMonitor.check.
if size(Z, 2) ~= M.n || size(Y, 2) ~= M.m || size(Z, 1) ~= size(Y, 1)
    error('ddma:monitor', 'expected Z (K x %d) and Y (K x %d)', M.n, M.m);
end
if ~all(isfinite([Z(:); Y(:)])), error('ddma:monitor', 'data must be finite'); end
viol = M.violations([]);
for k = 1:size(Z, 1)
    [M, this] = check_one(M, Z(k, :), Y(k, :));
    viol = [viol, this]; %#ok<AGROW>
end
end

function [M, this] = check_one(M, z, y)
idx = M.n_checked + 1;
this = M.violations([]);
th = M.d_hi - M.d_lo;
% level 1: the model's bounds at the point's cell
if ~isempty(M.model)
    e = M.model.edges;
    inside = true;
    for j = 1:numel(e)
        inside = inside && z(j) >= e{j}(1) && z(j) <= e{j}(end);
    end
    if inside
        [lo, hi] = ddma.bounds(M.model, z);
        gap = max(lo - y, y - hi);
        for i = find(gap > M.tol)
            this(end + 1) = viol_of(idx, i, 'cells', gap(i), nan, idx); %#ok<AGROW>
            M.need_width_model(i) = max(M.need_width_model(i), th(i) + gap(i));
        end
    end
end
% level 2: a fresh random sample of the points whose full check is done
V = M.pending_from;
if strcmp(M.full, 'later') && M.sample > 0 && V > 0
    [M, sidx] = ddma.internal.monitor_sample(M, V);
    [N, worst, wit] = ddma.internal.monitor_k_list(M.slope_lo, M.slope_hi, th, M.tol, ...
        M.Z, M.Y, V, z, y, sidx, M.st, M.use_mex);
    [M.need_width, M.need_lo, M.need_hi] = ddma.internal.monitor_merge(N, M.need_width, ...
        M.need_lo, M.need_hi, th);
    done = [this.output];
    for i = find(worst > M.tol)
        if ~any(done == i)
            this(end + 1) = viol_of(idx, i, 'sample', worst(i), M.ids(wit(i) + 1), idx); %#ok<AGROW>
        end
    end
end
% store (quarantined if levels 1-2 caught it)
M = ddma.internal.monitor_store(M, z, y, idx, 2 * uint8(~isempty(this)));
M.n_checked = idx;
if ~isempty(M.grid) && M.K > 2 * M.grid_K
    M = ddma.monitor_rebuild_index(M);
elseif isempty(M.grid) && M.prune && M.use_mex
    M = ddma.monitor_rebuild_index(M);
end
% level 3 in the same step
if strcmp(M.full, 'step')
    [M, v3] = ddma.internal.monitor_full_check(M);
    this = [this, v3];
end
if ~isempty(this)
    M = ddma.internal.monitor_record(M, this);
    M = ddma.internal.monitor_respond(M, this);
end
end

function v = viol_of(index, output, level, gap, witness, found_at)
v = struct('index', index, 'output', output, 'level', level, 'gap', gap, ...
           'witness', witness, 'found_at', found_at);
end
