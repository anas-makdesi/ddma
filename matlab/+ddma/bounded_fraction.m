function f = bounded_fraction(S)
%BOUNDED_FRACTION Fraction of cells whose interval is bounded in every output.
w = ddma.widths(S);
if iscell(w), error('ddma:bounded_fraction', 'not defined for per-output features'); end
f = mean(all(isfinite(w), 2));
end
