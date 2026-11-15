function set_num_threads(n)
%SET_NUM_THREADS Number of threads used by the compiled kernels.
%   ddma.set_num_threads(n) makes the MEX kernels of ddma.fit, ddma.fit_update,
%   ddma.bd_fit / ddma.bd_update (growth methods), ddma.growth_map and ddma.abstraction
%   use n threads.
%   ddma.set_num_threads() or ddma.set_num_threads(0) restores the default: the value of
%   the environment variable OMP_NUM_THREADS if it is set (before MATLAB starts),
%   otherwise all cores.
%   Results are bit-identical for any number of threads. Without the MEX files, or when
%   they were built without OpenMP, everything runs on one thread (same results).
%   See also ddma.get_num_threads, build_mex.
if nargin < 1 || isempty(n), n = 0; end
if ~isscalar(n) || n < 0 || n ~= round(n)
    error('ddma:threads', 'number of threads must be an integer >= 0');
end
ddma.internal.mex_opts('threads', double(n));
end
