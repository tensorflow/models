function simplenn_caffe_deploy(net, caffeModelBaseName, varargin)
% SIMPLENN_CAFFE_DEPLOY export a simplenn network to Caffe model
%   SIMPLENN_CAFFE_DEPLOY(NET, CAFFE_BASE_MODELNAME)
%   Export a simplenn network NET to a Caffe model.
%   The caffe model is stored in the following files:
%
%     [CAFFE_BASE_MODELNAME '.prototxt']   - Network definition file
%     [CAFFE_BASE_MODELNAME '.caffemodel'] - Binary Caffe model
%     [CAFFE_BASE_MODELNAME '_mean_image.binaryproto'] (optional) -
%         The average image (if set in net.normalization.averageImage)
%
%   Compiled MatCaffe (usually located in `<caffe_dir>/matlab`, built
%   with the `matcaffe` target) must be in path.
%
%   Only a limited subset of layers is currently supported and those are:
%
%     Conv, ReLU, Pool, LRN, SoftMax, SoftMaxLogLoss, DropOut
%
%   Please note that thanks to different implementations, the outputs of
%   simplenn and Caffe models are not neccessarily identical.
%
%   SIMPLENN_CAFFE_DEPLOY(NET, CAFFE_BASE_MODELNAME, 'OPT', VAL, ...)
%   takes the following options:
%
%   `removeDropout`:: `true`
%      When true, do not deploy dropout layers.
%
%   `replaceSoftMaxLoss`:: `true`
%      Replace SoftMax log loss with SoftMax.
%
%   `doTest`:: `true`
%      Compare the caffe model to the simplenn model using
%      `simplenn_caffe_compare`.
%
%   `testData`:: Random
%      Perform the test on the given data.
%
%   `inputBlobName`:: 'data'
%      Name of the input data blob in the final protobuf.
%
%   `labelBlobName`:: 'label'
%      Name of the input label blob in the final protobuf.
%
%   `outputBlobName`:: 'prob'
%      Name of the output blob in the resulting protobuf.
%
%   `silent`:: false
%      When true, suppresses all output to stdout.
%
%  See Also: simplenn_caffe_compare

% Copyright (C) 2015-16 Zohar Bar-Yehuda, Karel Lenc
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.inputBlobName = 'data';
opts.outputBlobName = 'prob';
opts.labelBlobName = 'label';
opts.removeDropout = true;
opts.replaceSoftMaxLoss = true;
opts.doTest = true;
opts.testData = [];
opts.silent = false;
opts = vl_argparse(opts, varargin);
if ~exist('caffe.Net', 'class'), error('MatCaffe not in path.'); end

info = @(varargin) fprintf(1, varargin{:});
if opts.silent, info = @(varargin) []; end;

info('Exporting simplenn model to caffe model %s\n', caffeModelBaseName);
[modelDir, name] = fileparts(caffeModelBaseName);
[~,~,~] = mkdir(modelDir);

% -------------------------------------------------------------------------
%                                                          Tidy the network
% -------------------------------------------------------------------------
net = vl_simplenn_tidy(net);
net = vl_simplenn_move(net, 'cpu');
% Remove dropout layers
if opts.removeDropout
  net.layers(cellfun(@(l) strcmp(l.type, 'dropout'), net.layers)) = [];
end

if opts.replaceSoftMaxLoss
  % If last layer is softmax loss, replace it with softmax
  ll = net.layers{end};
  if strcmp(ll.type, 'softmaxloss') || ...
      (strcmp(ll.type, 'loss') && strcmp(ll.loss, 'softmaxlog'))
    net.layers{end}.type = 'softmax';
  elseif isequal(net.layers{end}.type, 'loss')
    error('Unsupported loss function: %s', net.layers{end}.loss);
  end
end

for idx = 1:numel(net.layers)
  % Add missing layer names
  if ~isfield(net.layers{idx}, 'name')
    net.layers{idx}.name = sprintf('layer%d', idx);
  end
end

avImage = [];
if isfield(net.meta, 'normalization') && ...
    isfield(net.meta.normalization, 'imageSize')
  imSize = net.meta.normalization.imageSize;
  if isfield(net.meta.normalization, 'averageImage')
    avImage = net.meta.normalization.averageImage;
    if numel(avImage) == imSize(3)
      avImage = reshape(avImage, 1, 1, imSize(3));
    end
  end
else
  error('Missing image size. Please set `net.normalization.imageSize`.');
end

% -------------------------------------------------------------------------
%                                                           Export prototxt
% -------------------------------------------------------------------------
prototxtFilename = [caffeModelBaseName '.prototxt'];
fid = fopen(prototxtFilename, 'w');

fprintf(fid, 'name: "%s"\n\n', name); % Network name

% Export input dimensions
fprintf(fid, 'input: "data"\n');
fprintf(fid, 'input_dim: 1\n');
fprintf(fid, 'input_dim: %d\n', imSize(3));
fprintf(fid, 'input_dim: %d\n', imSize(1));
fprintf(fid, 'input_dim: %d\n\n', imSize(2));

dummyData = zeros(imSize, 'single'); % Keep track of data size at each layer;

isFullyConnected = false(size(net.layers));
for idx = 1:numel(net.layers)
  % write layers
  fprintf(fid,'layer {\n');
  fprintf(fid,'  name: "%s"\n', net.layers{idx}.name); % Layer name
  layerInputSize = size(dummyData);
  if numel(layerInputSize) == 2
    layerInputSize(3) = 1;
  end
  switch net.layers{idx}.type
    case 'conv'
      filtH = size(net.layers{idx}.weights{1},1);
      filtW = size(net.layers{idx}.weights{1},2);
      if filtH < layerInputSize(1) || filtW < layerInputSize(2)
        % Convolution layer
        fprintf(fid, '  type: "Convolution"\n');
        write_connection(fid, net.layers, idx);
        fprintf(fid, '  convolution_param {\n');
        write_kernel(fid, [filtH, filtW]);
        fprintf(fid, '    num_output: %d\n', size(net.layers{idx}.weights{1},4));
        write_stride(fid, net.layers{idx}.stride);
        if isfield(net.layers{idx}, 'pad') && numel(net.layers{idx}.pad) == 4
          % Make sure pad is symmetrical
          if any(net.layers{idx}.pad([1, 3]) ~= net.layers{idx}.pad([2, 3]))
            error('Caffe only supports symmetrical padding');
          end
        end
        write_pad(fid, net.layers{idx}.pad);
        numGroups = layerInputSize(3) / size(net.layers{idx}.weights{1}, 3);
        assert(mod(numGroups, 1) == 0);
        if numGroups > 1
          fprintf(fid, '    group: %d\n', numGroups);
        end

        fprintf(fid, '  }\n');
      elseif filtH == layerInputSize(1) && filtW == layerInputSize(2)
        % Fully connected layer
        isFullyConnected(idx) = true;
        fprintf(fid, '  type: "InnerProduct"\n');
        write_connection(fid, net.layers, idx);
        fprintf(fid, '  inner_product_param {\n');
        fprintf(fid, '    num_output: %d\n', ...
          size(net.layers{idx}.weights{1}, 4));
        fprintf(fid, '  }\n');
      else
        error('Filter size (%d,%d) is larger than input size (%d,%d)', ...
          filtH, filtW, layerInputSize(1), layerInputSize(2))
      end

    case 'relu'
      fprintf(fid, '  type: "ReLU"\n');
      write_connection(fid, net.layers, idx);

    case 'pool'
      fprintf(fid, '  type: "Pooling"\n');
      % Check padding compatability with caffe. See:
      % http://www.vlfeat.org/matconvnet/matconvnet-manual.pdf
      % for more details.
      if numel(net.layers{idx}.pad) == 1
        net.layers{idx}.pad = repmat(net.layers{idx}.pad, 1, 4);
      end
      if numel(net.layers{idx}.stride) == 1
        net.layers{idx}.stride = repmat(net.layers{idx}.stride, 1, 2);
      end
      if numel(net.layers{idx}.pool) == 1
        net.layers{idx}.pool = repmat(net.layers{idx}.pool, 1, 2);
      end
      pad = net.layers{idx}.pad;
      if pad([2, 4]) == net.layers{idx}.pool - 1
        pad([2, 4]) = 0;
      else
        pad([2, 4]) = pad([2, 4]) - net.layers{idx}.stride + 1;
      end
      % Some older versions did not use these upper bounds
      pad = max(pad, 0);

      write_connection(fid, net.layers, idx);
      fprintf(fid, '  pooling_param {\n');
      switch (net.layers{idx}.method)
        case 'max'
          caffe_pool = 'MAX';
        case 'avg'
          caffe_pool = 'AVE';
        otherwise
          error('Unknown pooling type');
      end
      fprintf(fid, '    pool: %s\n', caffe_pool);
      write_kernel(fid, net.layers{idx}.pool);
      write_stride(fid, net.layers{idx}.stride);
      write_pad(fid, pad);
      fprintf(fid, '  }\n');

    case {'normalize', 'lrn'}
      % MATLAB param = [local_size, kappa, alpha/local_size, beta]
      fprintf(fid, '  type: "LRN"\n');
      write_connection(fid, net.layers, idx);
      fprintf(fid, '  lrn_param {\n');
      fprintf(fid, '    local_size: %d\n', net.layers{idx}.param(1));
      fprintf(fid, '    k: %f\n', net.layers{idx}.param(2));
      fprintf(fid, '    alpha: %f\n', net.layers{idx}.param(3)*net.layers{idx}.param(1));
      fprintf(fid, '    beta: %f\n', net.layers{idx}.param(4));
      fprintf(fid, '  }\n');

    case 'softmax'
      fprintf(fid, '  type: "Softmax"\n');
      write_connection(fid, net.layers, idx);

    case {'loss', 'softmaxloss'}
      fprintf(fid, '  type: "SoftmaxWithLoss"\n');
      write_connection(fid, net.layers, idx, true);

    case 'dropout'
      fprintf(fid, '  type: "Dropout"\n');
      write_connection(fid, net.layers, idx);
      fprintf(fid, '  dropout_param {\n');
      fprintf(fid, '    dropout_ratio: %d\n', net.layers{idx}.rate);
      fprintf(fid, '  }\n');

    otherwise
      error('Unknown layer type: %s', net.layers{idx}.type);
  end
  fprintf(fid,'}\n\n');
  layer = struct('layers', {net.layers(idx)});
  res = vl_simplenn(layer, dummyData);
  dummyData = res(end).x;
end
fclose(fid);
info('Network definition exported to: %s.\n', prototxtFilename);

% -------------------------------------------------------------------------
%                                                         Export caffemodel
% -------------------------------------------------------------------------
caffe.set_mode_cpu();
caffeNet = caffe.Net(prototxtFilename, 'test');
firstConv = true;
for idx = 1:numel(net.layers)
  layer_type = net.layers{idx}.type;
  layer_name = net.layers{idx}.name;
  switch layer_type
    case 'conv'
      filters = net.layers{idx}.weights{1};
      % Convert from HxWxCxN to WxHxCxN per Caffe's convention
      filters = permute(filters, [2 1 3 4]);
      if firstConv
        if size(filters, 3) == 3
          % We assume this is RGB Conv., need to convert RGB to BGR
          filters = filters(:,:, [3 2 1], :);
        end
        firstConv = false; % Do this only for first convolution;
      end
      if isFullyConnected(idx)
        % Fully connected layer, squeeze to 2 dims
        filters = reshape(filters, [], size(filters, 4));
      end
      biases = net.layers{idx}.weights{2}(:);
      caffeNet.layers(layer_name).params(1).set_data(filters); % set weights
      caffeNet.layers(layer_name).params(2).set_data(biases); % set bias
    case {'relu', 'normalize', 'lrn', 'pool', 'softmax'}
      % No weights - nothing to do
    otherwise
      error('Unknown layer type %s', layer_type)
  end
end
modelFilename = [caffeModelBaseName '.caffemodel'];
caffeNet.save(modelFilename);
delete(caffeNet);
info('Model file exported to: %s.\n', modelFilename);

% -------------------------------------------------------------------------
%                                                        Export mean image
% -------------------------------------------------------------------------
if ~isempty(avImage)
  if size(avImage, 1) == 1 && size(avImage, 2) == 1
    % Single value, we'll duplicate it to im_size
    avImage = repmat(avImage, imSize(1), imSize(2));
  end
  avImage = matlab_img_to_caffe(avImage);
  meanFilename = [caffeModelBaseName, '_mean_image.binaryproto'];
  caffe.io.write_mean(avImage, meanFilename)
  info('Mean image exported to: %s.\n', meanFilename);
end

if opts.doTest
  simplenn_caffe_compare(net, caffeModelBaseName, opts.testData);
end


  function write_stride(fid, stride)
      if numel(stride) == 1
        fprintf(fid, '    stride: %d\n', stride);
      elseif numel(stride) == 2
        fprintf(fid, '    stride_h: %d\n', stride(1));
        fprintf(fid, '    stride_w: %d\n', stride(2));
      end
  end

  function write_kernel(fid, kernelSize)
    if numel(kernelSize) == 1
      fprintf(fid, '    kernel_size: %d\n', kernelSize);
    elseif numel(kernelSize) == 2
      fprintf(fid, '    kernel_h: %d\n', kernelSize(1));
      fprintf(fid, '    kernel_w: %d\n', kernelSize(2));
    end
  end


  function write_pad(fid, pad)
    if numel(pad) == 1
      fprintf(fid, '    pad: %d\n', pad);
    elseif numel(pad) == 4
      fprintf(fid, '    pad_h: %d\n', pad(1));
      fprintf(fid, '    pad_w: %d\n', pad(2));
    else
      error('pad vector size must be 1 or 4')
    end
  end

  function write_connection(fid, layers, idx, isLoss)
    if idx == 1
      bottom_name = opts.inputBlobName;
    else
      bottom_name = layers{idx-1}.name;
    end
    top_name = layers{idx}.name;
    if idx == numel(layers) && ~isempty(opts.outputBlobName)
      top_name = opts.outputBlobName;
    end
    fprintf(fid, '  bottom: "%s"\n', bottom_name);
    if nargin > 3 && isLoss
      fprintf(fid, '  bottom: "%s"\n', opts.labelBlobName);
    end
    fprintf(fid, '  top: "%s"\n', top_name);
  end


  function img = matlab_img_to_caffe(img)
    img = single(img);
    % Convert from HxWxCxN to WxHxCxN per Caffe's convention
    img = permute(img, [2 1 3 4]);
    if size(img,3) == 3
      % Convert from RGB to BGR channel order per Caffe's convention
      img = img(:,:, [3 2 1], :);
    end
  end
end
