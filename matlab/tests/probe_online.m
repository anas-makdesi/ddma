% TEMPORARY probe (to be removed before merging): why online_refinement differs on Octave 11
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
printf('octave %s\n', OCTAVE_VERSION);
T = 0.1; D = 0.005;
f = @(X, U) T * (-sin(X(:, 1)) - 0.2 * X(:, 2) + U(:, 1));
A_G = [1 T; 0 1];
Xedges = ddma.uniform_partition([-1 -1.5], [1 1.5], [60 60]);
Uedges = ddma.uniform_partition(-2, 2, 7);
edges = [Xedges(:); Uedges(:)]';
g_lo = [0 0 0; -0.3, -0.1, 0.05]; g_hi = [0 0 0;  0.1,  0.05, 0.15];
safe = ddma.cells_inside(Xedges, [-0.8 -1.2], [0.8 1.2]);
ddma.seed(1); K = 2000; r0 = 0.15;
X = [r0 * (2 * rand(K, 1) - 1), 1.5 * r0 * (2 * rand(K, 1) - 1)]; U = 4 * rand(K, 1) - 2;
dn = D * (2 * rand(K, 1) - 1);
printf('X %.17g U %.17g dn %.17g sin %.17g\n', sum(X(:)), sum(U), sum(dn), sum(sin(X(:, 1))));
Y = [zeros(K, 1), f(X, U) + dn];
printf('Y %.17g\n', sum(Y(:)));
for be = {'mex', 'matlab'}
  S = ddma.bd_fit([X U], Y, edges, g_lo, g_hi, [0 -D], [0 D], 'growth', 'slab', be{1});
  fn = fieldnames(S);
  for i = 1:numel(fn)
    v = S.(fn{i});
    if isnumeric(v) && numel(v) > 1, v = v(isfinite(v)); printf('  S.%s n=%d sum=%.17g\n', fn{i}, numel(v), sum(v(:))); end
  end
  A = ddma.abstraction(Xedges, Uedges, S, ddma.affine_bounds(A_G), [], [], false, be{1});
  [domain, ~] = ddma.safety(A, safe, be{1});
  printf('%s: %.2f %%\n', be{1}, 100 * nnz(domain) / nnz(safe));
end
