function build_mex(openmp)
%BUILD_MEX Compile the C++ kernels (optional; ddma falls back to MATLAB code):
%   ddma_fit_mex (ddma.fit), ddma_synth_mex (ddma.safety, ddma.reach_avoid) and
%   ddma_growth_mex (ddma.bd_fit, growth methods).
%   build_mex() builds with OpenMP (threads, see ddma.set_num_threads) when the compiler
%   supports it, and without otherwise (e.g. Apple clang): one thread, same results.
%   build_mex(false) builds without OpenMP.
%   -ffp-contract=off keeps floating-point operations exactly as written (no fused
%   multiply-add), as in the Python build.
if nargin < 1, openmp = true; end
here = fileparts(mfilename('fullpath'));
inc = ['-I' fullfile(here, '..', 'cpp')];
old = cd(here);
c = onCleanup(@() cd(old));
for name = {'ddma_fit_mex', 'ddma_synth_mex', 'ddma_growth_mex'}
    src = fullfile(here, [name{1} '.cpp']);
    done = false;
    if openmp
        try
            compile(src, inc, true);
            done = true;
        catch err
            fprintf('note: building %s without OpenMP (%s)\n', name{1}, err.message);
        end
    end
    if ~done, compile(src, inc, false); end
    fprintf('built %s\n', which(name{1}));
end
end

function compile(src, inc, openmp)
if exist('OCTAVE_VERSION', 'builtin')
    args = {'--mex', '-O3', '-std=c++17', '-ffp-contract=off', '-Wno-unknown-pragmas', inc, src};
    if openmp, args = [{'-fopenmp'}, args]; end
    [out, status] = mkoctfile(args{:});
    if status ~= 0, error('ddma:build', '%s', out); end
    return
end
cxx = '-std=c++17 -ffp-contract=off -Wno-unknown-pragmas';
ld = '';
msvc = false;
try
    cc = mex.getCompilerConfigurations('C++', 'Selected');
    msvc = ~isempty(strfind(lower(cc.ShortName), 'msvc'));
catch
end
if msvc
    flags = {'COMPFLAGS=$COMPFLAGS /std:c++17 /fp:precise'};
    if openmp, flags = {'COMPFLAGS=$COMPFLAGS /std:c++17 /fp:precise /openmp'}; end
else
    if openmp, cxx = [cxx ' -fopenmp']; ld = ' -fopenmp'; end
    flags = {['CXXFLAGS=$CXXFLAGS ' cxx], ['LDFLAGS=$LDFLAGS' ld]};
end
mex('-O', flags{:}, inc, src);
end
