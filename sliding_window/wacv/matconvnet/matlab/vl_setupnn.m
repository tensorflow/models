function vl_setupnn()
%VL_SETUPNN Setup the MatConvNet toolbox.
%   VL_SETUPNN() function adds the MatConvNet toolbox to MATLAB path.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

root = vl_rootnn() ;
addpath(fullfile(root, 'matlab')) ;
addpath(fullfile(root, 'matlab', 'mex')) ;
addpath(fullfile(root, 'matlab', 'simplenn')) ;
addpath(fullfile(root, 'matlab', 'xtest')) ;
addpath(fullfile(root, 'examples')) ;

if ~exist('gather')
  warning('The MATLAB Parallel Toolbox does not seem to be installed. Activating compatibility functions.') ;
  addpath(fullfile(root, 'matlab', 'compatibility', 'parallel')) ;
end

if numel(dir(fullfile(root, 'matlab', 'mex', 'vl_nnconv.mex*'))) == 0
  warning('MatConvNet is not compiled. Consider running `vl_compilenn`.');
end
