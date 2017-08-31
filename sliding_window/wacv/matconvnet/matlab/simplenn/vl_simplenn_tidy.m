function tnet = vl_simplenn_tidy(net)
%VL_SIMPLENN_TIDY  Fix an incomplete or outdated SimpleNN network.
%   NET = VL_SIMPLENN_TIDY(NET) takes the NET object and upgrades
%   it to the current version of MatConvNet. This is necessary in
%   order to allow MatConvNet to evolve, while maintaining the NET
%   objects clean. This function ignores custom layers.
%
%   The function is also generally useful to fill in missing default
%   values in NET.
%
%   See also: VL_SIMPLENN().

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

tnet = struct('layers', {{}}, 'meta', struct()) ;

% copy meta information in net.meta subfield
if isfield(net, 'meta')
  tnet.meta = net.meta ;
end

if isfield(net, 'classes')
  tnet.meta.classes = net.classes ;
end

if isfield(net, 'normalization')
  tnet.meta.normalization = net.normalization ;
end

% Adjust for the new version of vl_imreadjpeg
if  isfield(tnet, 'meta') && isfield(tnet.meta, 'normalization') && ...
   ~isfield(tnet.meta.normalization, 'cropSize') && ...
    isfield(tnet.meta.normalization, 'border') && ...
    isfield(tnet.meta.normalization, 'imageSize')
  insz = tnet.meta.normalization.imageSize(1:2);
  bigimSz = insz + tnet.meta.normalization.border;
  tnet.meta.normalization.cropSize = insz ./ bigimSz;
end

% copy layers
for l = 1:numel(net.layers)
  defaults = {'name', sprintf('layer%d', l), 'precious', false};
  layer = net.layers{l} ;

  % Ignore custom layers (e.g. for classes the `isfield` does not work)
  % The only interface requirement for custom layers is forward and
  % backward function.
  if strcmp(layer.type, 'custom')
    tnet.layers{l} = layer ;
    continue;
  end

  % check weights format
  switch layer.type
    case {'conv', 'convt', 'bnorm'}
      if ~isfield(layer, 'weights')
        layer.weights = {...
          layer.filters, ...
          layer.biases} ;
        layer = rmfield(layer, 'filters') ;
        layer = rmfield(layer, 'biases') ;
      end
  end
  if ~isfield(layer, 'weights')
    layer.weights = {} ;
  end

  % Check that weights include moments in batch normalization.
  if strcmp(layer.type, 'bnorm')
    if numel(layer.weights) < 3
      layer.weights{3} = ....
        zeros(numel(layer.weights{1}),2,'single') ;
    end
  end

  % Fill in missing values.
  switch layer.type
    case 'conv'
      defaults = [ defaults {...
        'pad', 0, ...
        'stride', 1, ...
        'dilate', 1, ...
        'opts', {}}] ;

    case 'pool'
      defaults = [ defaults {...
        'pad', 0, ...
        'stride', 1, ...
        'opts', {}}] ;

    case 'convt'
      defaults = [ defaults {...
        'crop', 0, ...
        'upsample', 1, ...
        'numGroups', 1, ...
        'opts', {}}] ;

    case {'pool'}
      defaults = [ defaults {...
        'method', 'max', ...
        'pad', 0, ...
        'stride', 1, ...
        'opts', {}}] ;

    case 'relu'
      defaults = [ defaults {...
        'leak', 0}] ;

    case 'dropout'
      defaults = [ defaults {...
        'rate', 0.5}] ;

    case {'normalize', 'lrn'}
      defaults = [ defaults {...
        'param', [5 1 0.0001/5 0.75]}] ;

    case {'pdist'}
      defaults = [ defaults {...
        'noRoot', false, ...
        'aggregate', false, ...
        'p', 2, ...
        'epsilon', 1e-3, ...
        'instanceWeights', []} ];

    case {'bnorm'}
      defaults = [ defaults {...
        'epsilon', 1e-5 } ] ;
  end

  for i = 1:2:numel(defaults)
    if ~isfield(layer, defaults{i})
      layer.(defaults{i}) = defaults{i+1} ;
    end
  end

  % save back
  tnet.layers{l} = layer ;
end
