function f = round_single(x, down)
%ROUND_SINGLE Round to single outward: the largest single <= x (down) or the smallest
%   single >= x (up). Same as the Python _round32 (nextafter).
f = single(x);
if down, bad = double(f) > x; else, bad = double(f) < x; end
if ~any(bad(:)), return; end
v = f(bad);
b = typecast(v(:), 'int32');
pos = v > 0; neg = v < 0; zer = v == 0;
step = int32(1);
if down
    b(pos) = b(pos) - step; b(neg) = b(neg) + step;   % toward -Inf
    nv = typecast(b, 'single');
    nv(zer) = -realmin('single') * eps('single');       % largest negative denormal
else
    b(pos) = b(pos) + step; b(neg) = b(neg) - step;   % toward +Inf
    nv = typecast(b, 'single');
    nv(zer) = realmin('single') * eps('single');
end
f(bad) = reshape(nv, size(v));
end
