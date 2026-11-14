function E = monotone_envelope(X, Xn, U, opts)
%MONOTONE_ENVELOPE Certified safe envelope of a monotone system from transitions, no grid.
%   E = ddma.monotone_envelope(X, Xn, U, opts) with X, Xn (N x n) states and measured
%   successors, x_k+ = f(x_k, u_k) + w_k, and U (N x m) inputs ([] if autonomous).
%   opts (struct, all fields optional):
%     w_lo, w_hi    additive disturbance bounds in state units (default 0)
%     x_lo, x_hi    safety box, lower-closed in the flipped coordinates (default +-Inf)
%     state_signs, input_signs   +1/-1 so that f increases in (s.*x, su.*u) (default +1)
%     u_lo, u_hi    input range where f is monotone (default: the data's range)
%     max_states    limit on partial corners kept by the search (default 2e6)
%   E.corners (L x n): maximal corners T (original coordinates); every state below T (in
%   the flipped order) can be kept below it forever. E.pairs_T, E.pairs_U: non-dominated
%   (corner, input limit) pairs. Use ddma.envelope_contains and ddma.envelope_input.
%   Method of Makdesi & Zamani, IFAC WC 2026. Same as Python ddma.invariance.monotone_envelope.
if nargin < 4, opts = struct(); end
[N, n] = size(X);
if ~isequal(size(Xn), [N, n]), error('ddma:monotone_envelope', 'X and Xn must match'); end
if isempty(U), U = zeros(N, 0); end
m = size(U, 2);
s = opt(opts, 'state_signs', ones(1, n), n);
su = opt(opts, 'input_signs', ones(1, m), m);
w_lo = opt(opts, 'w_lo', zeros(1, n), n);
w_hi = opt(opts, 'w_hi', zeros(1, n), n);
if any(w_lo > w_hi), error('ddma:monotone_envelope', 'w_lo must be <= w_hi'); end
x_lo = opt(opts, 'x_lo', -inf(1, n), n);
x_hi = opt(opts, 'x_hi', inf(1, n), n);
if any(isfinite(x_lo) & s > 0) || any(isfinite(x_hi) & s < 0)
    error('ddma:monotone_envelope', ['the safety box must be lower-closed in the ' ...
          'flipped coordinates: x_hi where state_signs = +1, x_lo where it is -1']);
end
if m
    u_lo = opt(opts, 'u_lo', min(U, [], 1), m);
    u_hi = opt(opts, 'u_hi', max(U, [], 1), m);
else
    u_lo = zeros(1, 0); u_hi = zeros(1, 0);
end
max_states = 2e6;
if isfield(opts, 'max_states'), max_states = opts.max_states; end

Z = bsxfun(@times, X, s);
V = bsxfun(@times, U, su);
z_max = x_hi; z_max(s < 0) = -x_lo(s < 0);
Zn_eff = bsxfun(@plus, bsxfun(@times, Xn, s), w_hi - w_lo);

counts = sum(Zn_eff <= Z, 1);
[~, order] = sort(counts);   % stable, as in Python
[T, Uv, n_valid] = search(Z, Zn_eff, V, order, max_states, z_max);
if ~isempty(T)
    TU = ddma.maximal_points([T, Uv]);
    T = TU(:, 1:n); Uv = TU(:, n + 1:end);
    corners = ddma.maximal_points(T);
else
    corners = zeros(0, n); T = zeros(0, n); Uv = zeros(0, m);
end
E = struct('kind', 'monotone', 'corners', bsxfun(@times, corners, s), ...
           'pairs_T', bsxfun(@times, T, s), 'pairs_U', bsxfun(@times, Uv, su), ...
           'state_signs', s, 'input_signs', su, 'u_lo', u_lo, 'u_hi', u_hi, ...
           'n_valid', n_valid);
end

function v = opt(o, name, default, n)
if isfield(o, name) && ~isempty(o.(name))
    v = reshape(double(o.(name)), 1, []);
    if numel(v) == 1, v = repmat(v, 1, n); end
    if numel(v) ~= n, error('ddma:monotone_envelope', '%s must have length %d', name, n); end
else
    v = default;
end
end

function [m, mu, n_valid] = search(Z, Zn_eff, V, order, max_states, z_max)
% same layered search as Python _monotone_search (see its comments)
n = size(Z, 2); mv = size(V, 2); L = numel(order);
cands = cell(1, L);
for i = 1:L
    d = order(i);
    cands{i} = find(Zn_eff(:, d) <= Z(:, d) & Zn_eff(:, d) <= z_max(d));
end
floors = cell(1, L);
for i = 1:L
    rest = cands(i + 1:end);
    if i < L && all(cellfun(@numel, rest) > 0)
        floors{i} = min(Z(vertcat(rest{:}), :), [], 1);
    else
        floors{i} = inf(1, n);
    end
end
m = inf(1, n); mu = inf(1, mv); t = zeros(1, 0);
for layer = 1:L
    dim = order(layer);
    cand = cands{layer};
    if isempty(cand), m = zeros(0, n); mu = zeros(0, mv); n_valid = 0; return; end
    Zc = Z(cand, :); Vc = V(cand, :); tc = Zn_eff(cand, dim);
    Q = numel(cand);
    chosen = order(1:layer);
    block = max(1, floor(2e6 / (Q * (n + mv + 1))));
    nm = {}; nmu = {}; nt = {};
    for s0 = 1:block:size(m, 1)
        rr = s0:min(size(m, 1), s0 + block - 1); P = numel(rr);
        M = min(repmat(permute(m(rr, :), [1 3 2]), [1 Q 1]), repmat(permute(Zc, [3 1 2]), [P 1 1]));
        MU = min(repmat(permute(mu(rr, :), [1 3 2]), [1 Q 1]), repmat(permute(Vc, [3 1 2]), [P 1 1]));
        Tt = cat(3, repmat(permute(t(rr, :), [1 3 2]), [1 Q 1]), repmat(tc', [P 1 1]));
        ok = all(Tt <= M(:, :, chosen), 3);
        M = reshape(M, P * Q, n); MU = reshape(MU, P * Q, mv); Tt = reshape(Tt, P * Q, layer);
        nm{end + 1} = M(ok(:), :); nmu{end + 1} = MU(ok(:), :); nt{end + 1} = Tt(ok(:), :); %#ok<AGROW>
    end
    m = vertcat(nm{:}); mu = vertcat(nmu{:}); t = vertcat(nt{:});
    fl = floors{layer};
    t(bsxfun(@le, t, fl(chosen))) = -inf;   % thresholds that can never bind again
    last = layer == L;
    if last
        inside = all(bsxfun(@le, m, z_max), 2);
        m = m(inside, :); mu = mu(inside, :); t = t(inside, :);
    end
    if isempty(m), m = zeros(0, n); mu = zeros(0, mv); n_valid = 0; return; end
    if last, key = [m, mu]; else, key = [m, mu, -t]; end
    key(key == inf) = 1e300; key(key == -inf) = -1e300;
    [~, first] = unique(key, 'rows', 'first');
    [~, idx] = ddma.maximal_points(key(first, :));
    idx = first(idx);
    m = m(idx, :); mu = mu(idx, :); t = t(idx, :);
    if size(m, 1) > max_states
        error('ddma:monotone_envelope', '%d partial corners; reduce the data', size(m, 1));
    end
end
n_valid = size(m, 1);
end
