function s = monitor_name(M, which, i)
%MONITOR_NAME Name of output / input i for reports (0-based numbering as in Python).
if strcmp(which, 'out')
    if ~isempty(M.output_names), s = M.output_names{i}; else, s = sprintf('output %d', i - 1); end
else
    if ~isempty(M.input_names), s = M.input_names{i}; else, s = sprintf('input %d', i - 1); end
end
end
