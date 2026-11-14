function S = bd_update(S, Z, Y)
%BD_UPDATE Add data to a model from ddma.bd_fit (exact: same model as a refit on all data).
%   S = ddma.bd_update(S, Z, Y) with Z (K x n), Y (K x m) as in ddma.bd_fit. The stored
%   maps are intersected with those of the new data (sign-matrix methods: the simulating
%   maps; growth methods: the vertex values and local maps), then the final step is
%   recomputed on all cells. The bounds only shrink, so every model along the way is
%   sound. For the 'lin_*' methods the final step solves one LP per cell: for frequent
%   small updates prefer 'growth' with ddma.bd_update_local.
%   Same as the Python BoundedDerivativeModel.update.
if isempty(Z), return; end
[K, n] = size(Z); m = numel(S.raw);
if size(Y, 1) ~= K || size(Y, 2) ~= m || n ~= numel(S.edges)
    error('ddma:bd_update', 'Z must be K x n and Y K x m');
end
useMex = ddma.internal.bd_use_mex(S.backend);
fb = S.backend; if strcmp(fb, 'mex'), fb = 'auto'; end
lp = strncmp(S.method, 'lin_', 4);
for i = 1:m
    R = S.raw{i};
    Zi = Z(:, R.feat);
    theta = R.d_hi - R.d_lo;
    if strcmp(R.kind, 'sign')
        L = size(R.a, 1);
        M = ddma.fit(Zi, Y(:, i) - Zi * R.a', R.edges, R.d_lo * ones(1, L), R.d_hi * ones(1, L), R.s, fb);
        R.slo = max(R.slo, M.lower); R.shi = min(R.shi, M.upper);
    else
        [vl, vu] = ddma.internal.bd_growth_vertices(Zi, Y(:, i), R.edges, R.gl, R.gh, theta, useMex);
        R.vlo = max(R.vlo, vl); R.vup = min(R.vup, vu);
        if ~isempty(S.local)
            [ll, lu] = ddma.internal.bd_growth_local(Zi, Y(:, i), R.edges, R.gl, R.gh, theta, ...
                                                     strcmp(S.local, 'slab'), useMex);
            R.llo = max(R.llo, ll); R.lup = min(R.lup, lu);
        end
    end
    [R.lo, R.up] = ddma.internal.bd_finalize(R, R.edges, ddma.all_subs(R.shape), lp);
    S.raw{i} = R;
    if isempty(S.features), S.lower(:, i) = R.lo; S.upper(:, i) = R.up; end
end
S.n_data = S.n_data + K;
end
