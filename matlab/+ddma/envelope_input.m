function u = envelope_input(E, x, u_desired)
%ENVELOPE_INPUT Certified input at state x closest to u_desired ([] outside the envelope).
%   u = ddma.envelope_input(E, x, u_desired) for E from ddma.monotone_envelope or
%   ddma.embedding_envelope (u_desired optional). Same as the Python safe_input().
x = reshape(x, 1, []);
if strcmp(E.kind, 'monotone')
    z = x .* E.state_signs; T = bsxfun(@times, E.pairs_T, E.state_signs);
    lim = E.pairs_U(all(bsxfun(@le, z, T), 2), :);
    if isempty(lim), u = []; return; end
    su = E.input_signs;
    v_lo = min(E.u_lo .* su, E.u_hi .* su);
    if nargin < 3 || isempty(u_desired), u_desired = 0.5 * (E.u_lo + E.u_hi); end
    vd = reshape(u_desired, 1, []) .* su;
    cand = bsxfun(@max, bsxfun(@min, vd, bsxfun(@times, lim, su)), v_lo);
    [~, k] = min(sum(bsxfun(@minus, cand, vd) .^ 2, 2));
    u = cand(k, :) .* su;
else
    k = find(all(bsxfun(@le, E.lo, x) & bsxfun(@le, x, E.hi), 2));
    if isempty(k), u = []; return; end
    lo = E.ulo(k, :); hi = E.uhi(k, :);
    if nargin < 3 || isempty(u_desired), u_desired = 0.5 * (lo(1, :) + hi(1, :)); end
    ud = reshape(u_desired, 1, []);
    cand = min(max(repmat(ud, numel(k), 1), lo), hi);
    [~, j] = min(sum(bsxfun(@minus, cand, ud) .^ 2, 2));
    u = cand(j, :);
end
end
