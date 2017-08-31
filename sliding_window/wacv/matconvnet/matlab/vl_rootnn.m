function root = vl_rootnn()
%VL_ROOTNN Get the root path of the MatConvNet toolbox.
%   VL_ROOTNN() returns the path to the MatConvNet toolbox.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

root = fileparts(fileparts(mfilename('fullpath'))) ;
