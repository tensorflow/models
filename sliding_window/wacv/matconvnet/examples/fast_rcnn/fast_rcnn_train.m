function [net, info] = fast_rcnn_train(varargin)
%FAST_RCNN_TRAIN  Demonstrates training a Fast-RCNN detector

% Copyright (C) 2016 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;
addpath(fullfile(vl_rootnn,'examples','fast_rcnn','bbox_functions'));
addpath(fullfile(vl_rootnn,'examples','fast_rcnn','datasets'));

opts.dataDir   = fullfile(vl_rootnn, 'data') ;
opts.sswDir    = fullfile(vl_rootnn, 'data', 'SSW');
opts.expDir    = fullfile(vl_rootnn, 'data', 'fast-rcnn-vgg16-pascal07') ;
opts.imdbPath  = fullfile(opts.expDir, 'imdb.mat');
opts.modelPath = fullfile(opts.dataDir, 'models', ...
  'imagenet-vgg-verydeep-16.mat') ;

opts.piecewise = true;  % piecewise training (+bbox regression)
opts.train.gpus = [] ;
opts.train.batchSize = 2 ;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.prefetch = false ; % does not help for two images in a batch
opts.train.learningRate = 1e-3 / 64 * [ones(1,6) 0.1*ones(1,6)];
opts.train.weightDecay = 0.0005 ;
opts.train.numEpochs = 12 ;
opts.train.derOutputs = {'losscls', 1, 'lossbbox', 1} ;
opts.lite = false  ;
opts.numFetchThreads = 2 ;

opts = vl_argparse(opts, varargin) ;
display(opts);

opts.train.expDir = opts.expDir ;
opts.train.numEpochs = numel(opts.train.learningRate) ;

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
net = fast_rcnn_init(...
  'piecewise',opts.piecewise,...
  'modelPath',opts.modelPath);

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------
if exist(opts.imdbPath,'file') == 2
  fprintf('Loading imdb...');
  imdb = load(opts.imdbPath) ;
else
  if ~exist(opts.expDir,'dir')
    mkdir(opts.expDir);
  end
  fprintf('Setting VOC2007 up, this may take a few minutes\n');
  imdb = cnn_setup_data_voc07_ssw(...
    'dataDir', opts.dataDir, ...
    'sswDir', opts.sswDir, ...
    'addFlipped', true, ...
    'useDifficult', true) ;
  save(opts.imdbPath,'-struct', 'imdb','-v7.3');
  fprintf('\n');
end
fprintf('done\n');

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
% use train + val split to train
imdb.images.set(imdb.images.set == 2) = 1;

% minibatch options
bopts = net.meta.normalization;
bopts.useGpu = numel(opts.train.gpus) >  0 ;
bopts.numFgRoisPerImg = 16;
bopts.numRoisPerImg = 64;
bopts.maxScale = 1000;
bopts.scale = 600;
bopts.bgLabel = numel(imdb.classes.name)+1;
bopts.visualize = 0;
bopts.interpolation = net.meta.normalization.interpolation;
bopts.numThreads = opts.numFetchThreads;
bopts.prefetch = opts.train.prefetch;

[net,info] = cnn_train_dag(net, imdb, @(i,b) ...
                           getBatch(bopts,i,b), ...
                           opts.train) ;

% --------------------------------------------------------------------
%                                                               Deploy
% --------------------------------------------------------------------
modelPath = fullfile(opts.expDir, 'net-deployed.mat');
if ~exist(modelPath,'file')
  net = deployFRCNN(net,imdb);
  net_ = net.saveobj() ;
  save(modelPath, '-struct', 'net_') ;
  clear net_ ;
end

% --------------------------------------------------------------------
function inputs = getBatch(opts, imdb, batch)
% --------------------------------------------------------------------
opts.visualize = 0;

if isempty(batch)
  return;
end

images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
opts.prefetch = (nargout == 0);

[im,rois,labels,btargets] = fast_rcnn_train_get_batch(images,imdb,...
  batch, opts);

if opts.prefetch, return; end

nb = numel(labels);
nc = numel(imdb.classes.name) + 1;

% regression error only for positives
instance_weights = zeros(1,1,4*nc,nb,'single');
targets = zeros(1,1,4*nc,nb,'single');

for b=1:nb
  if labels(b)>0 && labels(b)~=opts.bgLabel
    targets(1,1,4*(labels(b)-1)+1:4*labels(b),b) = btargets(b,:)';
    instance_weights(1,1,4*(labels(b)-1)+1:4*labels(b),b) = 1;
  end
end

rois = single(rois);

if opts.useGpu > 0
  im = gpuArray(im) ;
  rois = gpuArray(rois) ;
  targets = gpuArray(targets) ;
  instance_weights = gpuArray(instance_weights) ;
end

inputs = {'input', im, 'label', labels, 'rois', rois, 'targets', targets, ...
  'instance_weights', instance_weights} ;

% --------------------------------------------------------------------
function net = deployFRCNN(net,imdb)
% --------------------------------------------------------------------
% function net = deployFRCNN(net)
for l = numel(net.layers):-1:1
  if isa(net.layers(l).block, 'dagnn.Loss') || ...
      isa(net.layers(l).block, 'dagnn.DropOut')
    layer = net.layers(l);
    net.removeLayer(layer.name);
    net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
  end
end

net.rebuild();

pfc8 = net.getLayerIndex('predcls') ;
net.addLayer('probcls',dagnn.SoftMax(),net.layers(pfc8).outputs{1},...
  'probcls',{});

net.vars(net.getVarIndex('probcls')).precious = true ;

idxBox = net.getLayerIndex('predbbox') ;
if ~isnan(idxBox)
  net.vars(net.layers(idxBox).outputIndexes(1)).precious = true ;
  % incorporate mean and std to bbox regression parameters
  blayer = net.layers(idxBox) ;
  filters = net.params(net.getParamIndex(blayer.params{1})).value ;
  biases = net.params(net.getParamIndex(blayer.params{2})).value ;
  
  boxMeans = single(imdb.boxes.bboxMeanStd{1}');
  boxStds = single(imdb.boxes.bboxMeanStd{2}');
  
  net.params(net.getParamIndex(blayer.params{1})).value = ...
    bsxfun(@times,filters,...
    reshape([boxStds(:)' zeros(1,4,'single')]',...
    [1 1 1 4*numel(net.meta.classes.name)]));

  biases = biases .* [boxStds(:)' zeros(1,4,'single')];
  
  net.params(net.getParamIndex(blayer.params{2})).value = ...
    bsxfun(@plus,biases, [boxMeans(:)' zeros(1,4,'single')]);
end

net.mode = 'test' ;
