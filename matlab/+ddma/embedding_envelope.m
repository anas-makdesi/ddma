function E = embedding_envelope(X, Xn, U, jx_lo, jx_hi, ju_lo, ju_hi, opts)
%EMBEDDING_ENVELOPE Certified safe envelope of a nonlinear system with bounded Jacobians.
%   E = ddma.embedding_envelope(X, Xn, U, jx_lo, jx_hi, ju_lo, ju_hi, opts) with
%   X, Xn (N x n) states and measured successors, x_k+ = f(x_k, u_k) + w_k, U (N x m)
%   inputs ([] if autonomous), jx_lo/jx_hi (n x n) bounds on df_i/dx_j and ju_lo/ju_hi
%   (n x m) on df_i/du_k ([] if autonomous). opts (struct, optional fields): w_lo, w_hi
%   (disturbance, state units, default 0), x_lo, x_hi (safety box), pairs (P x 2, 1-based;
%   default all i < j in the order of Python's triu_indices), chunk.
%   Every pair spans a state box and an input box; growth cones of the two points bound
%   where the box can go; boxes that provably map into themselves are kept.
%   E.lo, E.hi (K x n), E.ulo, E.uhi (K x m), E.pairs (K x 2, 1-based), E.n_pairs.
%   Method of Makdesi, Zamani & Jafarpour, IFAC WC 2026. Same as the Python
%   ddma.invariance.embedding_envelope.
if nargin < 8, opts = struct(); end
[N, n] = size(X);
if isempty(U), U = zeros(N, 0); end
m = size(U, 2);
jx_lo = expand(jx_lo, n, n); jx_hi = expand(jx_hi, n, n);
if m
    ju_lo = expand(ju_lo, n, m); ju_hi = expand(ju_hi, n, m);
else
    ju_lo = zeros(n, 0); ju_hi = zeros(n, 0);
end
if any(jx_lo(:) > jx_hi(:)) || any(ju_lo(:) > ju_hi(:))
    error('ddma:embedding_envelope', 'derivative bounds must satisfy lo <= hi');
end
a = [jx_lo, ju_lo]; b = [jx_hi, ju_hi];
w_lo = getv(opts, 'w_lo', zeros(1, n), n); w_hi = getv(opts, 'w_hi', zeros(1, n), n);
x_lo = getv(opts, 'x_lo', -inf(1, n), n); x_hi = getv(opts, 'x_hi', inf(1, n), n);
chunk = 20000; if isfield(opts, 'chunk'), chunk = opts.chunk; end
width = w_hi - w_lo;
y_up = bsxfun(@plus, Xn, width); y_lo = bsxfun(@minus, Xn, width);
W = [X, U];
if isfield(opts, 'pairs') && ~isempty(opts.pairs)
    first = opts.pairs(:, 1); second = opts.pairs(:, 2);
else
    [second, first] = find(triu(true(N), 1)');   % row-major order, as numpy triu_indices
end
n_pairs = numel(first);
bp = max(b, 0); am = max(-a, 0); d = size(a, 2);
L = {}; H = {}; UL = {}; UH = {}; PP = {};
for s = 1:chunk:n_pairs
    r = s:min(n_pairs, s + chunk - 1);
    i = first(r); j = second(r);
    lo = min(X(i, :), X(j, :)); hi = max(X(i, :), X(j, :));
    ok = all(bsxfun(@ge, lo, x_lo), 2) & all(bsxfun(@le, hi, x_hi), 2);
    ok = ok & all(y_up(i, :) <= hi & y_up(j, :) <= hi & y_lo(i, :) >= lo & y_lo(j, :) >= lo, 2);
    i = i(ok); j = j(ok); lo = lo(ok, :); hi = hi(ok, :);
    if isempty(i), continue; end
    delta = W(j, :) - W(i, :);
    P = numel(i);
    g_hi = zeros(P, n); g_lo = zeros(P, n);
    ad = abs(delta); pos = delta > 0;
    for k = 1:n
        BP = repmat(bp(k, :), P, 1); AM = repmat(am(k, :), P, 1);
        au = ad .* (pos .* BP + ~pos .* AM); bu = ad .* (pos .* AM + ~pos .* BP);
        g_hi(:, k) = ddma.max_min_cones(y_up(i, k), y_up(j, k), au, bu);
        g_lo(:, k) = -ddma.max_min_cones(-y_lo(i, k), -y_lo(j, k), bu, au);
    end
    ok = all(g_hi <= hi & g_lo >= lo, 2);
    if ~any(ok), continue; end
    L{end + 1} = lo(ok, :); H{end + 1} = hi(ok, :); %#ok<AGROW>
    UL{end + 1} = min(U(i(ok), :), U(j(ok), :)); UH{end + 1} = max(U(i(ok), :), U(j(ok), :)); %#ok<AGROW>
    PP{end + 1} = [i(ok), j(ok)]; %#ok<AGROW>
end
E = struct('kind', 'boxes', 'lo', cat0(L, n), 'hi', cat0(H, n), 'ulo', cat0(UL, m), ...
           'uhi', cat0(UH, m), 'pairs', cat0(PP, 2), 'n_pairs', n_pairs);
end

function A = expand(A, r, c)
if isscalar(A), A = repmat(A, r, c); end
if ~isequal(size(A), [r, c]), error('ddma:embedding_envelope', 'bounds must be %d x %d', r, c); end
end

function v = getv(o, name, default, n)
if isfield(o, name) && ~isempty(o.(name))
    v = reshape(double(o.(name)), 1, []);
    if numel(v) == 1, v = repmat(v, 1, n); end
else
    v = default;
end
end

function A = cat0(C, c)
if isempty(C), A = zeros(0, c); else, A = vertcat(C{:}); end
end
