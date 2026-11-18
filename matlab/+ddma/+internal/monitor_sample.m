function [M, idx] = monitor_sample(M, V)
%MONITOR_SAMPLE S = M.sample indices in [0, V) (0-based) from the MINSTD sequence
%   x <- 48271 x mod (2^31 - 1): x_t = 48271^t x_0, index = x_t mod V. Same numbers as
%   the Python monitor. Products are split so that every step is exact in doubles.
P = 2147483647; S = M.sample;
if numel(M.rng_pw) ~= S
    pw = zeros(S, 1); x = 1;
    for t = 1:S
        x = mod(x * 48271, P); pw(t) = x;
    end
    M.rng_pw = pw;
end
x = M.rng_x;
a1 = floor(M.rng_pw / 65536); a0 = M.rng_pw - a1 * 65536;
u = mod(mod(a1 * x, P) * 65536 + a0 * x, P);   % (pw * x) mod P, exact
M.rng_x = u(end);
idx = mod(u, V);
end
