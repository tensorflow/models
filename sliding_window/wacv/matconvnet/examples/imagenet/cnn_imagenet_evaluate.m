function info = cnn_imagenet_evaluate(varargin)
% CNN_IMAGENET_EVALUATE   Evauate MatConvNet models on ImageNet

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.dataDir = fullfile('data', 'ILSVRC2012') ;
opts.expDir = fullfile('data', 'imagenet12-eval-vgg-f') ;
opts.modelPath = fullfile('data', 'models', 'imagenet-vgg-f.mat') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.networkType = [] ;
opts.lite = false ;
opts.numFetchThreads = 12 ;
opts.train.batchSize = 128 ;
opts.train.numEpochs = 1 ;
opts.train.gpus = [] ;
opts.train.prefetch = true ;
opts.train.expDir = opts.expDir ;

opts = vl_argparse(opts, varargin) ;
display(opts);

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
  imdb.imageDir = fullfile(opts.dataDir, 'images');
else
  imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

net = load(opts.modelPath) ;
if isfield(net, 'net') ;
  net = net.net ;
end
% Cannot use isa('dagnn.DagNN') because it is not an object yet
isDag = isfield(net, 'params') ;

if isDag
  opts.networkType = 'dagnn' ;
  net = dagnn.DagNN.loadobj(net) ;
  trainfn = @cnn_train_dag ;

  % Drop existing loss layers
  drop = arrayfun(@(x) isa(x.block,'dagnn.Loss'), net.layers) ;
  for n = {net.layers(drop).name}
    net.removeLayer(n) ;
  end

  % Extract raw predictions from softmax
  sftmx = arrayfun(@(x) isa(x.block,'dagnn.SoftMax'), net.layers) ;
  predVar = 'prediction' ;
  for n = {net.layers(sftmx).name}
    % check if output
    l = net.getLayerIndex(n) ;
    v = net.getVarIndex(net.layers(l).outputs{1}) ;
    if net.vars(v).fanout == 0
      % remove this layer and update prediction variable
      predVar = net.layers(l).inputs{1} ;
      net.removeLayer(n) ;
    end
  end

  % Add custom objective and loss layers on top of raw predictions
  net.addLayer('objective', dagnn.Loss('loss', 'softmaxlog'), ...
               {predVar,'label'}, 'objective') ;
  net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
               {predVar,'label'}, 'top1err') ;
  net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
                                     'opts', {'topK',5}), ...
               {predVar,'label'}, 'top5err') ;

  % Make sure that the input is called 'input'
  v = net.getVarIndex('data') ;
  if ~isnan(v)
    net.renameVar('data', 'input') ;
  end

  % Swtich to test mode
  net.mode = 'test' ;
else
  opts.networkType = 'simplenn' ;
  net = vl_simplenn_tidy(net) ;
  trainfn = @cnn_train ;
  net.layers{end}.type = 'softmaxloss' ; % softmax -> softmaxloss
end

% Synchronize label indexes used in IMDB with the ones used in NET
imdb = cnn_imagenet_sync_labels(imdb, net);

% Run evaluation
[net, info] = trainfn(net, imdb, getBatchFn(opts, net.meta), ...
                      opts.train, ...
                      'train', NaN, ...
                      'val', find(imdb.images.set==2)) ;

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------

if isfield(meta.normalization, 'keepAspect')
  keepAspect = meta.normalization.keepAspect ;
else
  keepAspect = true ;
end

if numel(meta.normalization.averageImage) == 3
  mu = double(meta.normalization.averageImage(:)) ;
else
  mu = imresize(single(meta.normalization.averageImage), ...
                meta.normalization.imageSize(1:2)) ;
end

useGpu = numel(opts.train.gpus) > 0 ;

bopts.test = struct(...
  'useGpu', useGpu, ...
  'numThreads', opts.numFetchThreads, ...
  'imageSize',  meta.normalization.imageSize(1:2), ...
  'cropSize', max(meta.normalization.imageSize(1:2)) / 256, ...
  'subtractAverage', mu, ...
  'keepAspect', keepAspect) ;

fn = @(x,y) getBatch(bopts,useGpu,lower(opts.networkType),x,y) ;

% -------------------------------------------------------------------------
function varargout = getBatch(opts, useGpu, networkType, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
if ~isempty(batch) && imdb.images.set(batch(1)) == 1
  phase = 'train' ;
else
  phase = 'test' ;
end
data = getImageBatch(images, opts.(phase), 'prefetch', nargout == 0) ;
if nargout > 0
  labels = imdb.images.label(batch) ;
  switch networkType
    case 'simplenn'
      varargout = {data, labels} ;
    case 'dagnn'
      varargout{1} = {'input', data, 'label', labels} ;
  end
end
