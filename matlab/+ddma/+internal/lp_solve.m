function [x, ok] = lp_solve(c, A, b, lb, ub)
%LP_SOLVE min c'x s.t. A x <= b, lb <= x <= ub, with linprog (MATLAB, Optimization
%   Toolbox) or glpk (Octave). ok = false if the solver did not report an optimum.
if exist('OCTAVE_VERSION', 'builtin')
    [x, ~, err, extra] = glpk(c, A, b, lb, ub, repmat('U', 1, size(A, 1)), ...
                              repmat('C', 1, numel(c)), 1);
    ok = err == 0 && extra.status == 5;
elseif exist('linprog', 'file')
    [x, ~, flag] = linprog(c, A, b, [], [], lb, ub, optimoptions('linprog', 'Display', 'none'));
    ok = flag == 1;
else
    error('ddma:lp', 'needs linprog (Optimization Toolbox) or Octave''s glpk');
end
end
