function [M, viol] = monitor_flush(M)
%MONITOR_FLUSH Run the full check (level 3) of all pending points now (full = 'later':
%   e.g. back at base after a run), record and respond. Same as AssumptionMonitor.flush.
[M, viol] = ddma.internal.monitor_full_check(M);
if ~isempty(viol)
    M = ddma.internal.monitor_record(M, viol);
    M = ddma.internal.monitor_respond(M, viol);
end
end
