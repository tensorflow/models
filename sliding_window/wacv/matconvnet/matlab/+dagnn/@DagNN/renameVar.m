function renameVar(obj, oldName, newName, varargin)
%RENAMEVAR Rename a variable
%   RENAMEVAR(OLDNAME, NEWNAME) changes the name of the variable
%   OLDNAME into NEWNAME. NEWNAME should not be the name of an
%   existing variable.

opts.quiet = false ;
opts = vl_argparse(opts, varargin) ;

% Find the variable to rename
v = obj.getVarIndex(oldName) ;
if isnan(v)
  % There is no such a variable, nothing to do
  if ~opts.quiet
    warning('There is no variable ''%s''.', oldName) ;
  end
  return ;
end

% Check if newName is an existing variable
newNameExists = any(strcmp(newName, {obj.vars.name})) ;

% Replace oldName with newName in all the layers
for l = 1:numel(obj.layers)
  for f = {'inputs', 'outputs'}
     f = char(f) ;
     sel = find(strcmp(oldName, obj.layers(l).(f))) ;
     [obj.layers(l).(f){sel}] = deal(newName) ;
  end
end

% If newVariable is a variable in the graph, then there is not
% anything else to do. obj.rebuild() will remove the slot
% in obj.vars() for oldName as that variable becomes unused.
%
% If, however, newVariable is not in the graph already, then
% the slot in obj.vars() is preserved and only the variable name
% is changed.

if ~newNameExists
  obj.vars(v).name = newName ;
  % update variable name hash otherwise rebuild() won't find this var
  % corectly
  obj.varNames.(newName) = v ;
end

obj.rebuild() ;
