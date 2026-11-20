function r = rand_int(lo, hi, varargin)
%RAND_INT Random integers in lo..hi from rand only: lo + floor((hi - lo + 1) * rand(...)).
%   Used instead of randi, whose algorithm differs between Octave versions (and MATLAB),
%   so that a seeded example (ddma.seed) draws the same integers in every Octave version.
if isempty(varargin), varargin = {1, 1}; end
r = lo + floor((hi - lo + 1) * rand(varargin{:}));
r = min(r, hi);
end
