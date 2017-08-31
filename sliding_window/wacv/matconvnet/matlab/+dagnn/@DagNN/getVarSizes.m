function sizes = getVarSizes(obj, inputSizes)
%GETVARSIZES  Get the size of the variables
%   SIZES = GETVARSIZES(OBJ, INPUTSIZES) computes the SIZES of the
%   DagNN variables given the size of the inputs. `inputSizes` is
%   a cell array of the type `{'inputName', inputSize, ...}`
%   Returns a cell array with sizes of all network variables.
%
%   Example, compute the storage needed for a batch size of 256 for an
%   imagenet-like network:
%   ```
%   batch_size = 256; single_num_bytes = 4;
%   input_size = [net.meta.normalization.imageSize, batch_size];
%   var_sizes = net.getVarSizes({'data', input_size});
%   fprintf('Network activations will take %.2fMiB in single.\n', ...
%     sum(prod(cell2mat(var_sizes, 1))) * single_num_bytes ./ 1024^3);
%   ```

% Copyright (C) 2015 Andrea Vedaldi, Karel Lenc.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

nv = numel(obj.vars) ;
sizes = num2cell(NaN(nv, 4),2)' ;

for i = 1:2:numel(inputSizes)
  v = obj.getVarIndex(inputSizes{i}) ;
  if isnan(v)
    error('Variable `%s` not found in the network.', inputSizes{i});
  end;
  if isempty(inputSizes{i+1})
    sizes{v} = [0 0 0 0] ;
  else
    sizes{v} = [inputSizes{i+1}(:)' ones(1, 4 - numel(inputSizes{i+1}))] ;
  end
end

for layer = obj.layers(obj.executionOrder)
  in = layer.inputIndexes ;
  out = layer.outputIndexes ;
  sizes(out) = layer.block.getOutputSizes(sizes(in)) ;
end
