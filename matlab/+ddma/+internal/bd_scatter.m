function v = bd_scatter(f, vals, N, fun, empty)
%BD_SCATTER Reduce vals into N bins with fun; bins without data get the value empty.
v = repmat(empty, N, 1);
if isempty(f), return; end
r = accumarray(f, vals, [N 1], fun);
hit = accumarray(f, 1, [N 1]) > 0;
v(hit) = r(hit);
end
