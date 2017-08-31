function obj = loadobj(s)
% LOADOBJ  Initialize a DagNN object from a structure.
%   OBJ = LOADOBJ(S) initializes a DagNN objet from the structure
%   S. It is the opposite of S = OBJ.SAVEOBJ().
%   If S is a string, initializes the DagNN object with data
%   from a mat-file S. Otherwise, if S is an instance of `dagnn.DagNN`,
%   returns S.

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if ischar(s) s = load(s); end
if isstruct(s)
  assert(isfield(s, 'layers'), 'Invalid model.');
  if ~isstruct(s.layers)
    warning('The model appears to be `simplenn` model. Using `fromSimpleNN` instead.');
    obj = dagnn.DagNN.fromSimpleNN(s);
    return;
  end
  obj = dagnn.DagNN() ;
  for l = 1:numel(s.layers)
    constr = str2func(s.layers(l).type) ;
    block = constr() ;
    block.load(struct(s.layers(l).block)) ;
    obj.addLayer(...
      s.layers(l).name, ...
      block, ...
      s.layers(l).inputs, ...
      s.layers(l).outputs, ...
      s.layers(l).params,...
      'skipRebuild', true) ;
  end
  obj.rebuild();
  if isfield(s, 'params')
    for f = setdiff(fieldnames(s.params)','name')
      f = char(f) ;
      for i = 1:numel(s.params)
        p = obj.getParamIndex(s.params(i).name) ;
        obj.params(p).(f) = s.params(i).(f) ;
      end
    end
  end
  if isfield(s, 'vars')
    for f = setdiff(fieldnames(s.vars)','name')
      f = char(f) ;
      for i = 1:numel(s.vars)
        p = obj.getVarIndex(s.vars(i).name) ;
        obj.vars(p).(f) = s.vars(i).(f) ;
      end
    end
  end
  for f = setdiff(fieldnames(s)', {'vars','params','layers'})
    f = char(f) ;
    obj.(f) = s.(f) ;
  end
elseif isa(s, 'dagnn.DagNN')
  obj = s ;
else
  error('Unknown data type %s for `loadobj`.', class(s));
end
