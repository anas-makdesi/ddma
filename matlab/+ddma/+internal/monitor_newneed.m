function N = monitor_newneed(m, n)
%MONITOR_NEWNEED Empty record of what each assumption would need (see monitor_accumulate).
N = struct('any', false(1, m), 'w', -inf(1, m), 'hi', -inf(m, n), 'hio', false(m, n), ...
           'lo', inf(m, n), 'loo', false(m, n));
end
