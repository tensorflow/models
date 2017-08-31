function eval(obj, inputs, derOutputs, varargin)
%EVAL Evaluate the DAGNN
%   EVAL(obj, inputs) evaluates the DaG for the specified input
%   values. `inputs` is a cell array of the type `{'inputName',
%   inputValue, ...}`. This call results in a forward pass through the
%   graph, computing the values of the output variables. These can
%   then be accessed using the `obj.vars(outputIndex)` property of the
%   DaG object. The index of an output can be obtained using the
%   `obj.getOutputIndex(outputName)` call.
%
%   EVAL(obj, inputs, derOutputs) evaluates the DaG forward and then
%   backward, performing backpropagation. Similar to `inputs`,
%   `derOutputs` is a cell array of the type {'outputName',
%   outputDerValue, ...} of output derivatives.
%
%   # Understanding backpropagation
%
%   Only those outputs for which an `outputDerValue` which is
%   non-empty are involved in backpropagation, while the others are
%   ignored. This is useful to attach to the graph auxiliary layers to
%   compute errors or other statistics, without however involving them
%   in backpropagation.
%
%   Usually one starts backpropagation from scalar outptus,
%   corresponding to loss functions. In this case `outputDerValue` can
%   be interpreted as the weight of that output and is usually set to
%   one. For example: `{'objective', 1}` backpropagates from the
%   `'objective'` output variable with a weight of 1.
%
%   However, in some cases the DaG may contain more than one such
%   node, for example because one has more than one loss function.  In
%   this case `{'objective1', w1, 'objective2', w2, ...}` allows to
%   balance the different objectives.
%
%   Finally, one can backpropagate from outputs that are *not*
%   scalars. While this is unusual, it is possible by specifying a
%   value of `outputDerValue` that has the same dimensionality as the
%   output; in this case, this value is used as a matrix of weights,
%   or projection.
%
%   # Factors affecting evaluation
%
%   There are several factors affecting evaluation:
%
%   * The *evaluation mode* can be either `normal` or `test`. Layers
%     may behave differently depending on the mode. For example,
%     dropout becomes a pass-through layer in test mode and batch
%     normalization use fixed moments (this usually improves the test
%     performance significantly).
%
%   * By default, the DaG aggressively conserves memory. This is
%     particularly important on the GPU, where memory is
%     scarce. However, this also means that the values of most
%     variables and of their derivatives are dropped during the
%     computation. For debugging purposes, it may be interesting to
%     observe these variables; in this case you can set the
%     `obj.conserveMemory` property of the DaG to `false`. It is also
%     possible to preserve individual variables by setting the
%     property `obj.vars(v).precious` to `true`.

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.holdOn = false ;
opts = vl_argparse(opts,varargin) ;

obj.computingDerivative = nargin > 2 && ~isempty(derOutputs) ;

if ~iscell(inputs), error('INPUTS is not a cell array.') ; end
if obj.computingDerivative && ~iscell(derOutputs), error('DEROUTPUTS is not a cell array.') ; end

% -------------------------------------------------------------------------
% Forward pass
% -------------------------------------------------------------------------

% set the input values
v = obj.getVarIndex(inputs(1:2:end)) ;
if any(isnan(v))
  broken = find(isnan(v)) ;
  error('No variable of name ''%s'' could be found in the DAG.', inputs{2*broken(1)-1}) ;
end
[obj.vars(v).value] = deal(inputs{2:2:end}) ;
inputs = [] ;

obj.numPendingVarRefs = [obj.vars.fanout] ;
for l = obj.executionOrder
  time = tic ;
  obj.layers(l).block.forwardAdvanced(obj.layers(l)) ;
  obj.layers(l).forwardTime = toc(time) ;
end

% -------------------------------------------------------------------------
% Backward pass
% -------------------------------------------------------------------------

if ~obj.computingDerivative, return ; end

obj.holdOn = opts.holdOn ;

% set output derivatives
derOutputsNames = derOutputs(1:2:end);
v = obj.getVarIndex(derOutputsNames) ;
if isnan(v)
  error('Invalid `derOutputs`, variables {%s} do not exist in the network.', ...
    strjoin(derOutputsNames(isnan(v)), ', '));
end
[obj.vars(v).der] = deal(derOutputs{2:2:end}) ;
derOutputs = [] ;

obj.numPendingVarRefs = zeros(1, numel(obj.vars)) ;
obj.numPendingParamRefs = zeros(1, numel(obj.params)) ;
for l = fliplr(obj.executionOrder)
  time = tic ;
  obj.layers(l).block.backwardAdvanced(obj.layers(l)) ;
  obj.layers(l).backwardTime = toc(time) ;
end
