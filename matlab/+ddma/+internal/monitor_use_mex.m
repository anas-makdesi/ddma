function tf = monitor_use_mex(backend)
%MONITOR_USE_MEX Whether the pair checks run in the MEX kernel ddma_monitor_mex
%   (backend 'auto' (if built), 'mex' or 'matlab'; same results).
switch backend
    case 'auto', tf = exist('ddma_monitor_mex', 'file') == 3;
    case 'mex'
        if exist('ddma_monitor_mex', 'file') ~= 3
            error('ddma:monitor', 'ddma_monitor_mex is not built (run build_mex)');
        end
        tf = true;
    case 'matlab', tf = false;
    otherwise, error('ddma:monitor', 'backend must be ''auto'', ''mex'' or ''matlab''');
end
end
