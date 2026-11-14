function seed(n)
%SEED Seed the random number generator in both MATLAB and Octave.
if exist('OCTAVE_VERSION', 'builtin')
    rand('seed', n); randn('seed', n); %#ok<RAND>
else
    rng(n, 'twister');   % also resets a legacy generator activated by rand('seed', ...)
end
end
