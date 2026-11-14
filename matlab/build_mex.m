function build_mex()
%BUILD_MEX Compile the C++ kernels (optional; ddma falls back to MATLAB code):
%   ddma_fit_mex (ddma.fit) and ddma_synth_mex (ddma.safety, ddma.reach_avoid).
here = fileparts(mfilename('fullpath'));
inc = ['-I' fullfile(here, '..', 'cpp')];
old = cd(here);
c = onCleanup(@() cd(old));
for name = {'ddma_fit_mex', 'ddma_synth_mex'}
    src = fullfile(here, [name{1} '.cpp']);
    if exist('OCTAVE_VERSION', 'builtin')
        mkoctfile('--mex', '-O2', '-std=c++17', inc, src);
    else
        mex('-O', 'CXXFLAGS=$CXXFLAGS -std=c++17', inc, src);
    end
    fprintf('built %s\n', which(name{1}));
end
end
