function rfs = getVarReceptiveFields(obj, var)
%GETVARRECEPTIVEFIELDS Get the receptive field of a variable
%   RFS = GETVARRECEPTIVEFIELDS(OBJ, VAR) gets the receptivie fields RFS of
%   all the variables of the DagNN OBJ into variable VAR. VAR is a variable
%   name or index.
%
%   RFS has one entry for each variable in the DagNN following the same
%   format as has DAGNN.GETRECEPTIVEFIELDS(). For example, RFS(i) is the
%   receptive field of the i-th variable in the DagNN into variable VAR. If
%   the i-th variable is not a descendent of VAR in the DAG, then there is
%   no receptive field, indicated by `rfs(i).size == []`. If the receptive
%   field cannot be computed (e.g. because it depends on the values of
%   variables and not just on the network topology, or if it cannot be
%   expressed as a sliding window), then `rfs(i).size = [NaN NaN]`.

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi. All rights reserved.
%
% This file is part of the VLFeat library and is made available under the
% terms of the BSD license (see the COPYING file).

if ~isnumeric(var)
  var_n = obj.getVarIndex(var) ;
  if isnan(var_n)
    error('Variable %s not found.', var);
  end
  var = var_n;
end
nv = numel(obj.vars) ;
nw = numel(var) ;
rfs = struct('size', cell(nw, nv), 'stride', cell(nw, nv), 'offset', cell(nw,nv)) ;

for w = 1:numel(var)
  rfs(w,var(w)).size = [1 1] ;
  rfs(w,var(w)).stride = [1 1] ;
  rfs(w,var(w)).offset = [1 1] ;
end

for l = obj.executionOrder
  % visit all blocks and get their receptive fields
  in = obj.layers(l).inputIndexes ;
  out = obj.layers(l).outputIndexes ;
  blockRfs = obj.layers(l).block.getReceptiveFields() ;

  for w = 1:numel(var)
    % find the receptive fields in each of the inputs of the block
    for i = 1:numel(in)
      for j = 1:numel(out)
        rf = composeReceptiveFields(rfs(w, in(i)), blockRfs(i,j)) ;
        rfs(w, out(j)) = resolveReceptiveFields([rfs(w, out(j)), rf]) ;
      end
    end
  end
end
end

% -------------------------------------------------------------------------
function rf = composeReceptiveFields(rf1, rf2)
% -------------------------------------------------------------------------
if isempty(rf1.size) || isempty(rf2.size)
  rf.size = [] ;
  rf.stride = [] ;
  rf.offset = [] ;
  return ;
end

rf.size = rf1.stride .* (rf2.size - 1) + rf1.size ;
rf.stride = rf1.stride .* rf2.stride ;
rf.offset = rf1.stride .* (rf2.offset - 1) + rf1.offset ;
end

% -------------------------------------------------------------------------
function rf = resolveReceptiveFields(rfs)
% -------------------------------------------------------------------------

rf.size = [] ;
rf.stride = [] ;
rf.offset = [] ;

for i = 1:numel(rfs)
  if isempty(rfs(i).size), continue ; end
  if isnan(rfs(i).size)
    rf.size = [NaN NaN] ;
    rf.stride = [NaN NaN] ;
    rf.offset = [NaN NaN] ;
    break ;
  end
  if isempty(rf.size)
    rf = rfs(i) ;
  else
    if ~isequal(rf.stride,rfs(i).stride)
      % incompatible geometry; this cannot be represented by a sliding
      % window RF field and may denotes an error in the network structure
      rf.size = [NaN NaN] ;
      rf.stride = [NaN NaN] ;
      rf.offset = [NaN NaN] ;
      break;
    else
      % the two RFs have the same stride, so they can be recombined
      % the new RF is just large enough to contain both of them
      a = rf.offset - (rf.size-1)/2 ;
      b = rf.offset + (rf.size-1)/2 ;
      c = rfs(i).offset - (rfs(i).size-1)/2 ;
      d = rfs(i).offset + (rfs(i).size-1)/2 ;
      e = min(a,c) ;
      f = max(b,d) ;
      rf.offset = (e+f)/2 ;
      rf.size = f-e+1 ;
    end
  end
end
end


