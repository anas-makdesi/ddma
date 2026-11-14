function run_threads_tests()
%RUN_THREADS_TESTS The learning MEX kernels give the same bits for any number of threads
%   and any way of writing the data into the grid (direct, bucketed, private copies,
%   owned parts), and the same as the MATLAB code. Run from the matlab/ folder:
%   addpath tests; run_threads_tests
if exist('ddma_fit_mex', 'file') ~= 3 || exist('ddma_growth_mex', 'file') ~= 3
    fprintf('run_threads_tests: MEX files not built, skipped\n');
    return
end
[nmax, openmp] = ddma.get_num_threads();
fprintf('OpenMP: %d, default threads: %d\n', openmp, nmax);
threads = unique([1 2 nmax]);
modes = 0:4;         % auto, direct, bucketed, private, owned
chunks = [0 997];    % default round size, and small rounds (many rounds)
c = onCleanup(@() ddma.internal.mex_opts('reset'));
nfail = 0;
rand('seed', 1); randn('seed', 1); %#ok<RAND>

% 1. ddma.fit: grids big enough to be cut into several blocks
cases = {{1, 40, 3000, 1}, {2, 150, 40000, 2}, {3, 40, 60000, 3}, {4, 14, 40000, 1}};
for k = 1:numel(cases)
    [n, kc, K, m] = cases{k}{:};
    for uniform = [true false]
        edges = make_edges(n, kc, uniform);
        Z = -1.1 + 2.2 * rand(K, n);
        Y = randn(K, m);
        signs = randi([-1 1], m, n);
        f = @() fit_bounds(Z, Y, edges, signs);
        nfail = nfail + same_everywhere(f, threads, modes, chunks, ...
            sprintf('fit n=%d K=%d m=%d uniform=%d', n, K, m, uniform));
        ref = f();
        S = ddma.fit(Z, Y, edges, 0, 0.05, signs, 'matlab');
        nfail = nfail + check(isequal(ref, {S.lower, S.upper}), ...
            sprintf('fit n=%d m=%d uniform=%d: MEX = MATLAB code', n, m, uniform));
    end
end

% 2. growth cones (vertices, cell, slab) through ddma.internal
cases = {{1, 30, 2000}, {2, 200, 30000}, {3, 25, 20000}, {4, 9, 15000}};
for k = 1:numel(cases)
    [n, kc, K] = cases{k}{:};
    for uniform = [true false]
        edges = make_edges(n, kc, uniform);
        Z = -1.2 + 2.4 * rand(K, n);
        y = sin(sum(Z, 2)) + 0.01 * randn(K, 1);
        gl = -2 + 2.5 * rand(1, n); gh = gl + 2 * rand(1, n);
        name = sprintf('growth n=%d K=%d uniform=%d', n, K, uniform);
        f = @() growth_bounds(Z, y, edges, gl, gh, 0, true);
        nfail = nfail + same_everywhere(f, threads, modes, chunks, [name ' vertices']);
        r = f(); a = growth_bounds(Z, y, edges, gl, gh, 0, false);
        nfail = nfail + check(close_enough(r, a), [name ' vertices: MEX = MATLAB code']);
        for slab = [false true]
            f = @() growth_bounds(Z, y, edges, gl, gh, 1 + slab, true);
            nfail = nfail + same_everywhere(f, threads, modes, chunks, ...
                sprintf('%s local slab=%d', name, slab));
        end
    end
end

% 3. through ddma.bd_fit (growth with slabs) and ddma.growth_map
edges = {linspace(0, 1, 61), linspace(0, 1, 51), linspace(0, 1, 41)};
Z = rand(60000, 3);
Y = [Z(:, 1) + Z(:, 2) .^ 2, sin(3 * Z(:, 3)) - Z(:, 1)];
f = @() bdfit_bounds(Z, Y, edges);
nfail = nfail + same_everywhere(f, threads, modes, [0 997], 'bd_fit growth slab');
f = @() gmap_bounds(Z, Y(:, 2), edges);
nfail = nfail + same_everywhere(f, threads, modes, [0 997], 'growth_map');

% 4. thread setting
ddma.set_num_threads(1);
nfail = nfail + check(ddma.get_num_threads() == 1, 'set_num_threads(1)');
ddma.set_num_threads();
nfail = nfail + check(ddma.get_num_threads() == nmax, 'set_num_threads() = default');
if nfail > 0
    error('run_threads_tests: %d check(s) failed', nfail);
end
fprintf('run_threads_tests: all checks passed\n');
end

function nfail = same_everywhere(f, threads, modes, chunks, name)
ddma.internal.mex_opts('reset');
ddma.set_num_threads(1);
ddma.internal.mex_opts('scatter_mode', 1);
ref = f();
bad = 0;
for t = threads
    for md = modes
        for ch = chunks
            ddma.set_num_threads(t);
            ddma.internal.mex_opts('scatter_mode', md);
            ddma.internal.mex_opts('chunk', ch);
            out = f();
            if ~bits_equal(out, ref)
                bad = bad + 1;
                fprintf('  differs: threads %d, mode %d, chunk %d\n', t, md, ch);
            end
        end
    end
end
ddma.internal.mex_opts('reset');
nfail = check(bad == 0, [name ': same bits for all threads and modes']);
end

function tf = bits_equal(a, b)
tf = numel(a) == numel(b);
for i = 1:numel(a)
    if ~tf, return, end
    x = a{i}; z = b{i};
    tf = isequal(class(x), class(z)) && isequal(size(x), size(z)) && ...
         isequal(typecast(x(:), 'uint8'), typecast(z(:), 'uint8'));
end
end

function out = fit_bounds(Z, Y, edges, signs)
S = ddma.fit(Z, Y, edges, 0, 0.05, signs, 'mex');
out = {S.lower, S.upper};
end

function out = growth_bounds(Z, y, edges, gl, gh, mode, useMex)
if mode == 0
    [a, b] = ddma.internal.bd_growth_vertices(Z, y, edges, gl, gh, 0.01, useMex);
else
    [a, b] = ddma.internal.bd_growth_local(Z, y, edges, gl, gh, 0.01, mode == 2, useMex);
end
out = {a, b};
end

function out = bdfit_bounds(Z, Y, edges)
S = ddma.bd_fit(Z, Y, edges, [-1.5 -0.1 -3.1; -1.1 -0.1 -3.1], [0.1 2.1 3.1; 0.1 0.1 3.1], ...
                -0.01, 0.01, 'growth', 'slab', 'mex');
out = {S.lower, S.upper};
end

function out = gmap_bounds(Z, y, edges)
S = ddma.growth_map(Z, y, edges, [-1.5 -0.1 -3.1], [0.1 0.1 3.1], -0.01, 0.01, 'slab', ...
                    'box', 'mex');
out = {S.lower, S.upper};
end

function edges = make_edges(n, k, uniform)
edges = cell(1, n);
for j = 1:n
    if uniform
        edges{j} = linspace(-1, 1, k + 1)';
    else
        edges{j} = sort(-1 + 2 * rand(k + 1, 1)) + 1e-6 * (0:k)';
    end
end
end

function tf = close_enough(a, b)
% same values up to rounding (the MATLAB code adds the cone terms in another order)
tf = true;
for i = 1:numel(a)
    fa = isfinite(a{i}); fb = isfinite(b{i});
    tf = tf && isequal(fa, fb) && isequal(a{i}(~fa), b{i}(~fb)) && ...
         all(abs(a{i}(fa) - b{i}(fb)) <= 1e-12 * max(1, abs(a{i}(fa))));
end
end

function nfail = check(ok, name)
nfail = ~ok;
if ok, fprintf('ok    %s\n', name); else, fprintf('FAIL  %s\n', name); end
end
