function renameLayer(obj, oldName, newName, varargin)
%RENAMELAYER Rename a layer
%   RENAMELAYER(OLDNAME, NEWNAME) changes the name of the layer
%   OLDNAME into NEWNAME. NEWNAME should not be the name of an
%   existing layer.

opts.quiet = false ;
opts = vl_argparse(opts, varargin) ;

% Find the layer to rename
v = obj.getLayerIndex(oldName) ;
if isnan(v)
  % There is no such layer, nothing to do
  if ~opts.quiet
    warning('There is no layer ''%s''.', oldName) ;
  end
  return ;
end

% Check if newName is an existing layer
newNameExists = any(strcmp(newName, {obj.layers.name})) ;
if newNameExists
  error('There is already a layer ''%s''.', newName) ;
end

% Replace oldName with newName in all the layers
obj.layers(v).name = newName ;
obj.rebuild() ;
