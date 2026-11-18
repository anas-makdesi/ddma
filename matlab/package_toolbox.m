function file = package_toolbox(version, out_dir)
%PACKAGE_TOOLBOX Package the +ddma code and the pre-built MEX files as a MATLAB toolbox.
%   file = package_toolbox('0.12.1') writes ddma-0.12.1.mltbx into the current folder;
%   package_toolbox(version, out_dir) writes it into out_dir. Needs MATLAB R2023a or newer
%   (matlab.addons.toolbox.ToolboxOptions); Octave cannot make .mltbx files.
%
%   The toolbox holds the +ddma package, the examples, and every MEX file found in this
%   folder (CI puts the ones built on Linux, macOS and Windows here before calling this),
%   plus libomp.dylib for the macOS MEX files (OpenMP for Apple's compiler; the MEX files
%   load it from their own folder, so users need no Homebrew). Without MEX files for a
%   system, ddma runs the .m code there: same results, slower.
here = fileparts(mfilename('fullpath'));
if nargin < 2, out_dir = pwd; end

stage = fullfile(tempdir, ['ddma_toolbox_' char(java.util.UUID.randomUUID())]);
mkdir(stage);
cleanup = onCleanup(@() rmdir(stage, 's'));
copyfile(fullfile(here, '+ddma'), fullfile(stage, '+ddma'));
copyfile(fullfile(here, 'examples'), fullfile(stage, 'examples'));
bins = [dir(fullfile(here, '*.mexa64')); dir(fullfile(here, '*.mexmaca64')); ...
        dir(fullfile(here, '*.mexmaci64')); dir(fullfile(here, '*.mexw64')); ...
        dir(fullfile(here, 'libomp.dylib'))];
for k = 1:numel(bins)
    copyfile(fullfile(bins(k).folder, bins(k).name), stage);
end
fprintf('packaging %d binary files: %s\n', numel(bins), strjoin({bins.name}, ', '));

% A fixed identifier: MATLAB recognises a newer .mltbx as an update of the same toolbox.
opts = matlab.addons.toolbox.ToolboxOptions(stage, '55ae1504-29d5-4073-b5de-f2b1c63b1eca');
opts.ToolboxName = 'ddma';
opts.ToolboxVersion = version;
opts.AuthorName = 'Anas Makdesi';
opts.Summary = 'Data-driven models and abstractions with guarantees';
opts.Description = ['Guaranteed models of unknown systems learned from data (monotone ' ...
    'systems, systems with bounded derivatives), symbolic abstractions, safe controllers ' ...
    'and safety filters, gridless safe envelopes and an online assumption monitor. ' ...
    'MATLAB mirror of the ddma Python package, with the same results. ' ...
    'https://github.com/anas-makdesi/ddma'];
opts.ToolboxMatlabPath = {stage};
opts.MinimumMatlabRelease = 'R2023b';
opts.OutputFile = fullfile(out_dir, sprintf('ddma-%s.mltbx', version));
matlab.addons.toolbox.packageToolbox(opts);
file = opts.OutputFile;
fprintf('wrote %s\n', file);
end
