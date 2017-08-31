function removeLayer(obj, layerName)
%REMOVELAYER Remove a layer from the network
%   REMOVELAYER(OBJ, NAME) removes the layer NAME from the DagNN object
%   OBJ. NAME can be a string or a cell array of strings.

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if ischar(layerName), layerName = {layerName}; end;
idxs = obj.getLayerIndex(layerName);
if any(isnan(idxs))
  error('Invalid layer name `%s`', ...
    strjoin(layerName(isnan(idxs)), ', '));
end
obj.layers(idxs) = [] ;
obj.rebuild() ;
