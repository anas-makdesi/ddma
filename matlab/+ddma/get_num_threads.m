function [n, openmp] = get_num_threads()
%GET_NUM_THREADS Number of threads the compiled learning kernels will use.
%   [n, openmp] = ddma.get_num_threads(): n threads (1 without the MEX files or without
%   OpenMP); openmp = true if the MEX files were built with OpenMP.
%   See also ddma.set_num_threads.
if exist('ddma_fit_mex', 'file') == 3
    [n, openmp] = ddma_fit_mex('info', ddma.internal.mex_opts());
    openmp = logical(openmp);
else
    n = 1;
    openmp = false;
end
end
