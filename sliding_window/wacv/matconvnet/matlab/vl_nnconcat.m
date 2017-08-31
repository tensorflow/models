function y = vl_nnconcat(inputs, dim, dzdy, varargin)
%VL_NNCONCAT CNN concatenate multiple inputs.
%  Y = VL_NNCONCAT(INPUTS, DIM) concatenates the inputs in the cell
%  array INPUTS along dimension DIM generating an output Y.
%
%  DZDINPUTS = VL_NNCONCAT(INPUTS, DIM, DZDY) computes the derivatives
%  of the block projected onto DZDY. DZDINPUTS has one element for
%  each element of INPUTS, each of which is an array that has the same
%  dimensions of the corresponding array in INPUTS.

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.inputSizes = [] ;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

if nargin < 2, dim = 3; end;
if nargin < 3, dzdy = []; end;

if isempty(dzdy)
  y = cat(dim, inputs{:});
else
  if isempty(opts.inputSizes)
    opts.inputSizes = cellfun(@(inp) [size(inp,1),size(inp,2),size(inp,3),size(inp,4)], inputs, 'UniformOutput', false) ;
  end
  start = 1 ;
  y = cell(1, numel(opts.inputSizes)) ;
  s.type = '()' ;
  s.subs = {':', ':', ':', ':'} ;
  for i = 1:numel(opts.inputSizes)
    stop = start + opts.inputSizes{i}(dim) ;
    s.subs{dim} = start:stop-1 ;
    y{i} = subsref(dzdy,s) ;
    start = stop ;
  end
end
