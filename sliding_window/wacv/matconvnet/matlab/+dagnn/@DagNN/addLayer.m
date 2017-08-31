function addLayer(obj, name, block, inputs, outputs, params, varargin)
%ADDLAYER  Adds a layer to a DagNN
%   ADDLAYER(NAME, LAYER, INPUTS, OUTPUTS, PARAMS) adds the
%   specified layer to the network. NAME is a string with the layer
%   name, used as a unique indentifier. BLOCK is the object
%   implementing the layer, which should be a subclass of the
%   Layer. INPUTS, OUTPUTS are cell arrays of variable names, and
%   PARAMS of parameter names.
%
%   See Also REMOVELAYER().
opts.skipRebuild = false;
opts = vl_argparse(opts, varargin);

index = find(strcmp(name, {obj.layers.name})) ;
if ~isempty(index), error('There is already a layer with name ''%s''.', name), end
index = numel(obj.layers) + 1 ;

if nargin < 6, params = {} ; end
if ischar(inputs), inputs = {inputs} ; end
if ischar(outputs), outputs = {outputs} ; end
if ischar(params), params = {params} ; end

obj.layers(index) = struct(...
  'name', {name}, ...
  'inputs', {inputs}, ...
  'outputs', {outputs}, ...
  'params', {params}, ...
  'inputIndexes', {[]}, ...
  'outputIndexes', {[]}, ...
  'paramIndexes', {[]}, ...
  'forwardTime', {[]}, ...
  'backwardTime', {[]}, ...
  'block', {block}) ;
obj.layers(index).block.attach(obj, index) ;
if ~opts.skipRebuild, obj.rebuild() ; end ;
