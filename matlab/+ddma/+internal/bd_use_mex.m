function tf = bd_use_mex(backend)
%BD_USE_MEX Whether the growth MEX kernel is used for this backend setting.
tf = strcmp(backend, 'mex') || (strcmp(backend, 'auto') && exist('ddma_growth_mex', 'file') == 3);
end
