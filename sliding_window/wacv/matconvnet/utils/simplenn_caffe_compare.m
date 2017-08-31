function diffStats = simplenn_caffe_compare( net, caffeModelBaseName, testData, varargin)
% SIMPLENN_CAFFE_COMPARE compare the simplenn network and caffe models
%   SIMPLENN_CAFFE_COMPARE(NET, CAFFE_BASE_MODELNAME) Evaluates a forward
%   pass of a simplenn network NET and caffe models stored in
%   CAFFE_BASE_MODELNAME and numerically compares the network outputs using
%   a random input data.
%
%   SIMPLENN_CAFFE_COMPARE(NET, CAFFE_BASE_MODELNAME, TEST_DATA) Evaluates
%   the simplenn network and Caffe model on a given data. If TEST_DATA is
%   an empty array, uses a random input.
%
%   RES = SIMPLENN_CAFFE_COMPARE(...) returns a structure with the
%   statistics of the differences where each field of a structure RES is
%   named after a blob and contains basic statistics:
%     `[MIN_DIFF, MEAN_DIFF, MAX_DIFF]`
%
%   This script attempts to match the NET layer names and caffe blob names
%   and shows the MIN, MEAN and MAX difference between the outputs. For
%   caffe model, the mean image stored with the caffe model is used (see
%   `simplenn_caffe_deploy` for details). Furthermore the script compares
%   the execution time of both networks.
%
%   Compiled MatCaffe (usually located in `<caffe_dir>/matlab`, built
%   with the `matcaffe` target) must be in path.
%
%   SIMPLENN_CAFFE_COMPARE(..., 'OPT', VAL, ...) takes the following
%   options:
%
%   `numRepetitions`:: `1`
%      Evaluate the network multiple times. Useful to compare the execution
%      time.
%
%   `device`:: `cpu`
%      Evaluate the network on the specified device (CPU or GPU). For GPU
%      evaluation, the current GPU is used for both Caffe and simplenn.
%
%   `silent`:: `false`
%      When true, supress all outputs to stdin.
%
%  See Also: simplenn_caffe_deploy

% Copyright (C) 2016 Karel Lenc, Zohar Bar-Yehuda
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.numRepetitions = 1;
opts.randScale = 100;
opts.device = 'cpu';
opts.silent = false;
opts = vl_argparse(opts, varargin);

info = @(varargin) fprintf(1, varargin{:});
if opts.silent, info = @(varargin) []; end;

if ~exist('caffe.Net', 'class'), error('MatCaffe not in path.'); end
prototxtFilename = [caffeModelBaseName '.prototxt'];
if ~exist(prototxtFilename, 'file')
  error('Caffe net definition `%s` not found', prototxtFilename);
end;
modelFilename = [caffeModelBaseName '.caffemodel'];
if ~exist(prototxtFilename, 'file')
  error('Caffe net model `%s` not found', modelFilename);
end;
meanFilename = [caffeModelBaseName, '_mean_image.binaryproto'];

net = vl_simplenn_tidy(net);
net = vl_simplenn_move(net, opts.device);

netBlobNames = [{'data'}, cellfun(@(l) l.name, net.layers, ...
  'UniformOutput', false)];

% Load the Caffe model
caffeNet = caffe.Net(prototxtFilename, modelFilename, 'test');
switch opts.device
  case 'cpu'
    caffe.set_mode_cpu();
  case 'gpu'
    caffe.set_mode_gpu();
    gpuDev = gpuDevice();
    caffe.set_device(gpuDev.Index - 1);
end
caffeBlobNames = caffeNet.blob_names';
[caffeLayerFound, caffe2netres] = ismember(caffeBlobNames, netBlobNames);
info('Found %d matches between simplenn layers and caffe blob names.\n',...
  sum(caffeLayerFound));

% If testData not supplied, use random input
imSize = net.meta.normalization.imageSize;
if ~exist('testData', 'var') || isempty(testData)
  testData = rand(imSize, 'single') * opts.randScale;
end
if ischar(testData), testData = imread(testData); end
testDataSize = [size(testData), 1, 1];
assert(all(testDataSize(1:3) == imSize(1:3)), 'Invalid test data size.');

testData = single(testData);
dataCaffe = matlab_img_to_caffe(testData);
if isfield(net.meta.normalization, 'averageImage') && ...
    ~isempty(net.meta.normalization.averageImage)
  avImage = net.meta.normalization.averageImage;
  if numel(avImage) == imSize(3)
    avImage = reshape(avImage, 1, 1, imSize(3));
  end
  testData = bsxfun(@minus, testData, avImage);
end

% Test MatConvNet model
stime = tic;
for rep = 1:opts.numRepetitions
  res = vl_simplenn(net, testData, [], [], 'ConserveMemory', false);
end
info('MatConvNet %s time: %.1f ms.\n', opts.device, ...
  toc(stime)/opts.numRepetitions*1000);

if ~isempty(meanFilename) && exist(meanFilename, 'file')
  mean_img_caffe = caffe.io.read_mean(meanFilename);
  dataCaffe = bsxfun(@minus, dataCaffe, mean_img_caffe);
end

% Test Caffe model
stime = tic;
for rep = 1:opts.numRepetitions
  caffeNet.forward({dataCaffe});
end
info('Caffe %s time: %.1f ms.\n', opts.device, ...
  toc(stime)/opts.numRepetitions*1000);

diffStats = struct();
for li = 1:numel(caffeBlobNames)
  blob = caffeNet.blobs(caffeBlobNames{li});
  caffeData = permute(blob.get_data(), [2, 1, 3, 4]);
  if li == 1 && size(caffeData, 3) == 3
    caffeData = caffeData(:, :, [3, 2, 1]);
  end
  mcnData = gather(res(caffe2netres(li)).x);
  diff = abs(caffeData(:) - mcnData(:));
  diffStats.(caffeBlobNames{li}) = [min(diff), mean(diff), max(diff)]';
end

if ~opts.silent
  pp = '% 10s % 10s % 10s % 10s\n';
  precp = '% 10.2e';
  fprintf(pp, 'Layer name', 'Min', 'Mean', 'Max');
  for li = 1:numel(caffeBlobNames)
    lstats = diffStats.(caffeBlobNames{li});
    fprintf(pp, caffeBlobNames{li}, sprintf(precp, lstats(1)), ...
      sprintf(precp, lstats(2)), sprintf(precp, lstats(3)));
  end
  fprintf('\n');
end
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
