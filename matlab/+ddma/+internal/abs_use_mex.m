function tf = abs_use_mex(A)
%ABS_USE_MEX Whether the transitions of A are computed by the MEX kernel ddma_abs_mex.
b = 'auto';
if isfield(A.opts, 'backend'), b = A.opts.backend; end
tf = strcmp(b, 'mex') || (strcmp(b, 'auto') && exist('ddma_abs_mex', 'file') == 3);
end
