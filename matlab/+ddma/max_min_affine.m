function out = max_min_affine(offsets, slopes, lo, hi, fallback)
%MAX_MIN_AFFINE Certified upper bounds of max over a box of a minimum of affine functions.
%   out(k) >= max_{z in [lo(k,:), hi(k,:)]} min_l (offsets(k, l) + slopes(l, :) * z')
%   offsets: N x L (entries may be +Inf: constraint absent); slopes: L x n; lo, hi: N x n
%   (finite); fallback: N x 1 sound upper bounds (default +Inf).
%   One small LP per cell (linprog, or glpk in Octave). Its value is not used directly:
%   from the dual multipliers lam (normalized to sum 1) the bound
%       sum_l lam_l off_l + sum_j max(mu_j lo_j, mu_j hi_j),  mu = sum_l lam_l a_l,
%   is an upper bound for any such lam (weak duality), so the result is sound whatever
%   the solver accuracy. Same method as ddma/_box_lp.py.
[N, L] = size(offsets); n = size(slopes, 2);
if nargin < 5 || isempty(fallback), fallback = inf(N, 1); end
out = fallback(:);
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if ~isOctave && ~exist('linprog', 'file')
    error('ddma:max_min_affine', ['the lin_* methods need linprog (Optimization Toolbox) ' ...
          'or Octave''s glpk']);
end
if ~isOctave
    opts = optimoptions('linprog', 'Display', 'none');
end
cost = [zeros(n, 1); -1];          % variables [z; c], maximize c
for k = 1:N
    fin = isfinite(offsets(k, :));
    if ~any(fin), continue; end
    A = [-slopes(fin, :), ones(nnz(fin), 1)];   % c - a_l . z <= off_l
    b = offsets(k, fin)';
    lb = [lo(k, :)'; -Inf]; ub = [hi(k, :)'; Inf];
    if isOctave
        [~, ~, err, extra] = glpk(cost, A, b, lb, ub, repmat('U', 1, nnz(fin)), ...
                                  repmat('C', 1, n + 1), 1);
        if err ~= 0 || ~isfield(extra, 'lambda') || extra.status ~= 5, continue; end
        lam = abs(extra.lambda(:));
    else
        [~, ~, flag, ~, mult] = linprog(cost, A, b, [], [], lb, ub, opts);
        if flag ~= 1 || isempty(mult), continue; end
        lam = max(mult.ineqlin(:), 0);
    end
    tot = sum(lam);
    if ~(tot > 1e-12), continue; end
    lam = lam / tot;
    mu = lam' * slopes(fin, :);
    val = lam' * b + sum(max(mu .* lo(k, :), mu .* hi(k, :)));
    out(k) = min(out(k), val);
end
end
