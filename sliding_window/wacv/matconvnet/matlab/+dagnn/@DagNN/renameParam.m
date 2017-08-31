function renameParam(obj, oldName, newName, varargin)
%RENAMELAYER Rename a parameter
%   RENAMEPARAM(OLDNAME, NEWNAME) changes the name of the parameter
%   OLDNAME into NEWNAME. NEWNAME should not be the name of an
%   existing parameter.

opts.quiet = false ;
opts = vl_argparse(opts, varargin) ;

% Find the param to rename
v = obj.getParamIndex(oldName) ;
if isnan(v)
  % There is no such param, nothing to do
  if ~opts.quiet
    warning('There is no parameter ''%s''.', oldName) ;
  end
  return ;
end

% Check if newName is an existing param
newNameExists = any(strcmp(newName, {obj.params.name})) ;
if newNameExists
  error('There is already a layer ''%s''.', newName) ;
end

% Replace oldName with newName in all the layers
for l = 1:numel(obj.layers)
    sel = find(strcmp(oldName, obj.layers(l).params));
    [obj.layers(l).params{sel}] = deal(newName) ;
end

if ~newNameExists
  obj.params(v).name = newName ;
  obj.paramNames.(newName) = v ;
end

obj.rebuild() ;