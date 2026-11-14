function [domain, allowed, value] = reach_avoid(A, target, avoid, backend)
%REACH_AVOID States that can be driven to target while avoiding avoid (least fixed point).
%   value(q) = number of steps needed (Inf if unreachable); allowed(q, p) inputs whose
%   successors all have a smaller value.
%   backend: 'auto' (MEX if compiled, see build_mex) | 'mex' | 'matlab'. Same result.
target = logical(target(:));
if nargin < 3 || isempty(avoid), avoid = false(size(target)); end
if nargin < 4, backend = 'auto'; end
avoid = logical(avoid(:));
if strcmp(backend, 'mex') || (strcmp(backend, 'auto') && exist('ddma_synth_mex', 'file') == 3)
    [domain, allowed, value] = ddma_synth_mex('reach_avoid', A.first, A.last, A.enabled, ...
                                              A.shapeX, target, avoid);
    return
end
R = target & ~avoid;
value = inf(size(R)); value(R) = 0;
allowed = false(size(A.enabled));
k = 0;
while true
    k = k + 1;
    good = ddma.box_all_in(A, R, 'matlab');
    new = ~R & ~avoid & any(good, 2);
    if ~any(new), break; end
    allowed(new, :) = good(new, :);
    value(new) = k;
    R = R | new;
end
domain = R;
end
