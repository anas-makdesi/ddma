function opts = mex_opts(field, value)
%MEX_OPTS Options passed to the learning MEX files: [threads, scatter_mode, chunk].
%   opts = ddma.internal.mex_opts() returns them (passed as the last argument of
%   ddma_fit_mex and ddma_growth_mex). Each MEX file is a separate library, so the
%   setting is kept here and sent with every call.
%   ddma.internal.mex_opts('threads', n): see ddma.set_num_threads.
%   For tests and benchmarks only (same results whatever the values):
%   mex_opts('scatter_mode', m): 0 auto, 1 direct, 2 bucketed, 3 private, 4 owned;
%   mex_opts('chunk', c): values (bucketed) or points (owned) per round, 0 = default;
%   mex_opts('reset'): all back to 0 (defaults).
persistent o
if isempty(o), o = [0 0 0]; end
if nargin >= 1
    switch field
        case 'threads', o(1) = value;
        case 'scatter_mode', o(2) = value;
        case 'chunk', o(3) = value;
        case 'reset', o = [0 0 0];
        otherwise, error('ddma:mex_opts', 'unknown option %s', field);
    end
end
opts = o;
end
