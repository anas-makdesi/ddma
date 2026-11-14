function in = envelope_contains(E, x)
%ENVELOPE_CONTAINS Which states (rows of x) lie in a certified envelope.
%   in = ddma.envelope_contains(E, x) for E from ddma.monotone_envelope or
%   ddma.embedding_envelope. Same as the Python contains() methods.
in = false(size(x, 1), 1);
if strcmp(E.kind, 'monotone')
    if isempty(E.corners), return; end
    z = bsxfun(@times, x, E.state_signs); T = bsxfun(@times, E.corners, E.state_signs);
    for s = 1:4096:size(z, 1)
        r = s:min(size(z, 1), s + 4095);
        in(r) = any(all(bsxfun(@le, permute(z(r, :), [1 3 2]), permute(T, [3 1 2])), 3), 2);
    end
else
    if isempty(E.lo), return; end
    for s = 1:2048:size(x, 1)
        r = s:min(size(x, 1), s + 2047);
        xs = permute(x(r, :), [1 3 2]);
        in(r) = any(all(bsxfun(@le, permute(E.lo, [3 1 2]), xs) & ...
                        bsxfun(@le, xs, permute(E.hi, [3 1 2])), 3), 2);
    end
end
end
