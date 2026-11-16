function [M, viol] = monitor_check(M, Z, Y)
%MONITOR_CHECK Check new transitions one by one against all stored data (and the model's
%   cell bounds, if any), then store them. viol: struct array with fields index (1-based
%   position among the checked transitions), output, kind ('data' or 'model'), gap,
%   witness (id of the stored point it conflicts with; seed data -1, -2, ...).
%   The response (on_violation) is applied after each violating transition.
%   Same as AssumptionMonitor.check.
if size(Z, 2) ~= M.n || size(Y, 2) ~= M.m || size(Z, 1) ~= size(Y, 1)
    error('ddma:monitor', 'expected Z (K x %d) and Y (K x %d)', M.n, M.m);
end
viol = struct('index', {}, 'output', {}, 'kind', {}, 'gap', {}, 'witness', {});
th = M.d_hi - M.d_lo;
for k = 1:size(Z, 1)
    z = Z(k, :); y = Y(k, :);
    idx = M.n_checked + 1;
    this = struct('index', {}, 'output', {}, 'kind', {}, 'gap', {}, 'witness', {});
    if ~isempty(M.Z)
        dz = z - M.Z; dy = y - M.Y;
        for i = 1:M.m
            gn = dy(:, i) - th(i) - sum(ddma.internal.monitor_cone(dz, M.slope_lo(i, :), M.slope_hi(i, :)), 2);
            go = -dy(:, i) - th(i) - sum(ddma.internal.monitor_cone(-dz, M.slope_lo(i, :), M.slope_hi(i, :)), 2);
            vn = gn > M.tol; vo = go > M.tol;
            if ~any(vn) && ~any(vo), continue; end
            if any(vn), M = ddma.internal.monitor_accumulate(M, i, dz(vn, :), dy(vn, i)); end
            if any(vo), M = ddma.internal.monitor_accumulate(M, i, -dz(vo, :), -dy(vo, i)); end
            [g, w] = max(max(gn, go));
            this(end + 1) = struct('index', idx, 'output', i, 'kind', 'data', 'gap', g, ...
                                   'witness', M.ids(w)); %#ok<AGROW>
        end
    end
    if ~isempty(M.model)
        e = M.model.edges;
        inside = true;
        for j = 1:numel(e)
            inside = inside && z(j) >= e{j}(1) && z(j) <= e{j}(end);
        end
        if inside
            [lo, hi] = ddma.bounds(M.model, z);
            gap = max(lo - y, y - hi);
            done = [this.output];
            for i = 1:M.m
                if gap(i) > M.tol && ~any(done == i)
                    this(end + 1) = struct('index', idx, 'output', i, 'kind', 'model', ...
                                           'gap', gap(i), 'witness', -1); %#ok<AGROW>
                end
            end
        end
    end
    M = ddma.internal.monitor_store(M, z, y, idx);
    M.n_checked = idx;
    if ~isempty(this)
        viol = [viol, this]; %#ok<AGROW>
        M.violations = [M.violations, this];
        M.violating(end + 1, 1) = idx;
        M = respond(M, this);
        th = M.d_hi - M.d_lo;
    end
end
end

function M = respond(M, this)
if ~isempty(M.callback), M.callback(this); end
if strcmp(M.on_violation, 'fallback') && ~M.fallback_active
    M.fallback_active = true; M.fallback_since = this(1).index;
elseif strcmp(M.on_violation, 'widen')
    if all(strcmp({this.kind}, 'model'))
        for v = this
            M.need_width(v.output) = max(M.need_width(v.output), ...
                M.d_hi(v.output) - M.d_lo(v.output) + v.gap);
        end
    end
    target = M.widen;
    [kind, i, j] = ddma.internal.monitor_target(target);
    if ~strcmp(kind, 'disturbance')
        if strcmp(kind, 'slope_lo'), need = M.need_lo(i, j); else, need = M.need_hi(i, j); end
        if isnan(need), target = 'disturbance'; end
    end
    M = ddma.monitor_widen(M, target, this(1).index);
end
end
