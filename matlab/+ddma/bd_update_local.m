function S = bd_update_local(S, Z, Y, window)
%BD_UPDATE_LOCAL Windowed update of a growth model (thesis Section 3.2, Prop. 13).
%   S = ddma.bd_update_local(S, Z, Y, window): each data point tightens the model only on
%   the cells within window cells of its own cell (scalar, or 1 x n per coordinate), and
%   only those cells are recomputed. Sound; cheaper than ddma.bd_update on large grids and
%   never tighter than it. Meant for online use, one or a few points at a time.
%   The changed cell boxes are appended to S.changed (rows [first, last], 0-based, in the
%   coordinates of S); ddma.abstraction_refresh recomputes the transitions of the
%   affected states. Same model as the Python update_local.
if ~any(strcmp(S.method, {'growth', 'lin_growth'}))
    error('ddma:bd_update_local', 'available for the growth methods');
end
if isempty(Z), return; end
[K, n] = size(Z); m = numel(S.raw);
if size(Y, 1) ~= K || size(Y, 2) ~= m || n ~= numel(S.edges)
    error('ddma:bd_update_local', 'Z must be K x n and Y K x m');
end
wfull = round(double(window(:)')) .* ones(1, n);
useMex = ddma.internal.bd_use_mex(S.backend);
lp = strncmp(S.method, 'lin_', 4);
for k = 1:K
    boxes = zeros(m, 2 * n);
    for i = 1:m
        R = S.raw{i};
        nf = numel(R.feat);
        z = Z(k, R.feat); w = wfull(R.feat);
        c = ddma.internal.bd_cells_of(R.edges, z);
        first = max(c - w, 0); last = min(c + w, R.shape - 1);
        bshape = last - first + 1;
        bedges = cell(1, nf);
        for j = 1:nf, bedges{j} = R.edges{j}(first(j) + 1:last(j) + 2); end
        v = ddma.flat_index(R.shape + 1, ddma.all_subs(bshape + 1) + first);   % window vertices
        csub = ddma.all_subs(bshape) + first;                                % window cells
        f = ddma.flat_index(R.shape, csub);
        y = Y(k, i); theta = R.d_hi - R.d_lo;
        % the point's cone at the window's vertices (separable sum over the axes)
        up = y + theta; lo = y - theta;
        for j = 1:nf
            t = bedges{j}(:) - z(j);
            sh = [ones(1, j - 1), numel(t), 1];
            up = up + reshape(max(R.gh(j) * t, R.gl(j) * t), sh);
            lo = lo + reshape(min(R.gh(j) * t, R.gl(j) * t), sh);
        end
        R.vup(v) = min(R.vup(v), up(:));
        R.vlo(v) = max(R.vlo(v), lo(:));
        if ~isempty(S.local)
            [ll, lu] = ddma.internal.bd_growth_local(z, y, bedges, R.gl, R.gh, theta, ...
                                                     strcmp(S.local, 'slab'), useMex);
            R.llo(f) = max(R.llo(f), ll); R.lup(f) = min(R.lup(f), lu);
        end
        [R.lo(f), R.up(f)] = ddma.internal.bd_finalize(R, R.edges, csub, lp);
        S.raw{i} = R;
        if isempty(S.features), S.lower(f, i) = R.lo(f); S.upper(f, i) = R.up(f); end
        % changed box in the coordinates of S (full range on the other coordinates)
        bf = zeros(1, n); bl = S.shape - 1;
        bf(R.feat) = first; bl(R.feat) = last;
        boxes(i, :) = [bf, bl];
    end
    S.changed = [S.changed; unique(boxes, 'rows', 'stable')];
end
S.n_data = S.n_data + K;
end
