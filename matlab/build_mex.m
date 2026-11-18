function build_mex(openmp)
%BUILD_MEX Compile the C++ kernels (optional; ddma falls back to MATLAB code):
%   ddma_fit_mex (ddma.fit), ddma_synth_mex (ddma.safety, ddma.reach_avoid),
%   ddma_growth_mex (ddma.bd_fit, growth methods), ddma_abs_mex (ddma.abstraction) and
%   ddma_monitor_mex (ddma.check_data, ddma.monitor_check).
%   build_mex() builds with OpenMP (threads, see ddma.set_num_threads) when the compiler
%   supports it: GCC/Clang (-fopenmp), Microsoft's compiler (/openmp), and Apple's
%   compiler when Homebrew's libomp is installed (brew install libomp; another folder can
%   be given with the environment variable LIBOMP_PREFIX). Otherwise it builds without
%   OpenMP: one thread, same results.
%   build_mex(false) builds without OpenMP.
%   On Linux and macOS the environment variable DDMA_MEX_CXX can name the C++ compiler, e.g.
%   setenv('DDMA_MEX_CXX', 'g++-10'): MEX files built with an older GCC also load in older
%   MATLAB releases, which ship an older C++ library (used for the pre-built toolbox).
%   -ffp-contract=off keeps floating-point operations exactly as written (no fused
%   multiply-add), as in the Python build.
if nargin < 1, openmp = true; end
here = fileparts(mfilename('fullpath'));
inc = ['-I' fullfile(here, '..', 'cpp')];
old = cd(here);
c = onCleanup(@() cd(old));
if openmp, ways = omp_ways(); else, ways = {}; end
for name = {'ddma_fit_mex', 'ddma_synth_mex', 'ddma_growth_mex', 'ddma_abs_mex', 'ddma_monitor_mex'}
    src = fullfile(here, [name{1} '.cpp']);
    done = false;
    for w = 1:numel(ways)
        try
            compile(src, inc, ways{w});
            done = true;
            break
        catch err
            fprintf('note: %s: OpenMP via %s failed (%s)\n', name{1}, ways{w}.name, ...
                    strtok(err.message, sprintf('\n')));
        end
    end
    if ~done
        compile(src, inc, struct('name', 'none', 'cxx', '', 'ld', '', 'msvc', ''));
        if openmp
            hint = '';
            if ismac, hint = ' (on macOS: brew install libomp, then build_mex again)'; end
            fprintf('note: %s built without OpenMP: one thread%s\n', name{1}, hint);
        end
    end
    fprintf('built %s\n', which(name{1}));
end
end

function ways = omp_ways()
% Ways to switch OpenMP on, best first. The C++ code is the same; only the flags differ.
ways = {struct('name', '-fopenmp', 'cxx', '-fopenmp', 'ld', '-fopenmp', 'msvc', '/openmp')};
if ismac
    p = libomp_prefix();
    if ~isempty(p)
        ways{end + 1} = struct('name', 'Homebrew libomp', ...
            'cxx', ['-Xpreprocessor -fopenmp -I' fullfile(p, 'include')], ...
            'ld', ['-L' fullfile(p, 'lib') ' -lomp -Wl,-rpath,' fullfile(p, 'lib')], ...
            'msvc', '');
    end
end
end

function p = libomp_prefix()
% Folder of Homebrew's libomp (for Apple's compiler), or '' if not found.
cands = {getenv('LIBOMP_PREFIX')};
[status, out] = system('brew --prefix libomp 2>/dev/null');
if status == 0, cands{end + 1} = strtrim(out); end
cands = [cands, {'/opt/homebrew/opt/libomp', '/usr/local/opt/libomp'}];
p = '';
for i = 1:numel(cands)
    if ~isempty(cands{i}) && exist(fullfile(cands{i}, 'include', 'omp.h'), 'file')
        p = cands{i};
        return
    end
end
end

function compile(src, inc, way)
if exist('OCTAVE_VERSION', 'builtin')
    % mkoctfile takes the OpenMP flags through the environment
    oldc = getenv('CXXFLAGS'); oldl = getenv('LDFLAGS');
    c = onCleanup(@() restore_env(oldc, oldl));
    if ~isempty(way.cxx)
        setenv('CXXFLAGS', strtrim([mkoctfile('-p', 'CXXFLAGS') ' ' way.cxx]));
        setenv('LDFLAGS', strtrim([mkoctfile('-p', 'LDFLAGS') ' ' way.ld]));
    end
    args = {'--mex', '-O3', '-std=c++17', '-ffp-contract=off', '-Wno-unknown-pragmas', inc, src};
    [out, status] = mkoctfile(args{:});
    if status ~= 0, error('ddma:build', '%s', out); end
    return
end
msvc = false;
try
    cc = mex.getCompilerConfigurations('C++', 'Selected');
    msvc = ~isempty(strfind(lower(cc.ShortName), 'msvc'));
catch
end
if msvc
    if ~isempty(way.cxx) && isempty(way.msvc), error('ddma:build', 'not for MSVC'); end
    flags = {['COMPFLAGS=$COMPFLAGS /std:c++17 /fp:precise ' way.msvc]};
else
    flags = {['CXXFLAGS=$CXXFLAGS -std=c++17 -ffp-contract=off -Wno-unknown-pragmas ' way.cxx], ...
             ['LDFLAGS=$LDFLAGS ' way.ld]};
    cxx = getenv('DDMA_MEX_CXX');
    if ~isempty(cxx), flags = [flags, {['CXX=' cxx], ['LDXX=' cxx]}]; end
end
mex('-O', flags{:}, inc, src);
end

function restore_env(c, l)
setenv('CXXFLAGS', c);
setenv('LDFLAGS', l);
end
