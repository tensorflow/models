function net = fast_rcnn_init(varargin)
%FAST_RCNN_INIT  Initialize a Fast-RCNN

% Copyright (C) 2016 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.piecewise = 1;
opts.modelPath = fullfile('data', 'models','imagenet-vgg-verydeep-16.mat');
opts = vl_argparse(opts, varargin) ;
display(opts) ;

% Load an imagenet pre-trained cnn model.
net = load(opts.modelPath);
net = vl_simplenn_tidy(net);

% Add drop-out layers.
relu6p = find(cellfun(@(a) strcmp(a.name, 'relu6'), net.layers)==1);
relu7p = find(cellfun(@(a) strcmp(a.name, 'relu7'), net.layers)==1);

drop6 = struct('type', 'dropout', 'rate', 0.5, 'name','drop6');
drop7 = struct('type', 'dropout', 'rate', 0.5, 'name','drop7');
net.layers = [net.layers(1:relu6p) drop6 net.layers(relu6p+1:relu7p) drop7 net.layers(relu7p+1:end)];

% Change loss for FC layers.
nCls = 21;
fc8p = find(cellfun(@(a) strcmp(a.name, 'fc8'), net.layers)==1);
net.layers{fc8p}.name = 'predcls';
net.layers{fc8p}.weights{1} = 0.01 * randn(1,1,size(net.layers{fc8p}.weights{1},3),nCls,'single');
net.layers{fc8p}.weights{2} = zeros(1, nCls, 'single');

% Skip pool5.
pPool5 = find(cellfun(@(a) strcmp(a.name, 'pool5'), net.layers)==1);
net.layers = net.layers([1:pPool5-1,pPool5+1:end-1]);

% Convert to DagNN.
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

% Add ROIPooling layer.
vggdeep = false;
pRelu5 = find(arrayfun(@(a) strcmp(a.name, 'relu5'), net.layers)==1);
if isempty(pRelu5)
  vggdeep = true;
  pRelu5 = find(arrayfun(@(a) strcmp(a.name, 'relu5_3'), net.layers)==1);
  if isempty(pRelu5)
    error('Cannot find last relu before fc');
  end
end

pFc6 = (arrayfun(@(a) strcmp(a.name, 'fc6'), net.layers)==1);
if vggdeep
  net.addLayer('roipool', dagnn.ROIPooling('method','max','transform',1/16,...
    'subdivisions',[7,7],'flatten',0), ...
    {net.layers(pRelu5).outputs{1},'rois'}, 'xRP');
else
  net.addLayer('roipool', dagnn.ROIPooling('method','max','transform',1/16,...
    'subdivisions',[6,6],'flatten',0), ...
    {net.layers(pRelu5).outputs{1},'rois'}, 'xRP');
end

pRP = (arrayfun(@(a) strcmp(a.name, 'roipool'), net.layers)==1);
net.layers(pFc6).inputs{1} = net.layers(pRP).outputs{1};

% Add softmax loss layer.
pFc8 = (arrayfun(@(a) strcmp(a.name, 'predcls'), net.layers)==1);
net.addLayer('losscls',dagnn.Loss(), ...
  {net.layers(pFc8).outputs{1},'label'}, ...
  'losscls',{});

% Add bbox regression layer.
if opts.piecewise
  pparFc8 = (arrayfun(@(a) strcmp(a.name, 'predclsf'), net.params)==1);
  pdrop7 = (arrayfun(@(a) strcmp(a.name, 'drop7'), net.layers)==1);
  net.addLayer('predbbox',dagnn.Conv('size',[1 1 size(net.params(pparFc8).value,3) 84],'hasBias', true), ...
    net.layers(pdrop7).outputs{1},'predbbox',{'predbboxf','predbboxb'});

  net.params(end-1).value = 0.001 * randn(1,1,size(net.params(pparFc8).value,3),84,'single');
  net.params(end).value = zeros(1,84,'single');

  net.addLayer('lossbbox',dagnn.LossSmoothL1(), ...
    {'predbbox','targets','instance_weights'}, ...
    'lossbbox',{});
end

net.rebuild();

% No decay for bias and set learning rate to 2
for i=2:2:numel(net.params)
  net.params(i).weightDecay = 0;
  net.params(i).learningRate = 2;
end

% Change image-mean as in fast-rcnn code
net.meta.normalization.averageImage = ...
  reshape([122.7717 102.9801 115.9465],[1 1 3]);

net.meta.normalization.interpolation = 'bilinear';

net.meta.classes.name = {'aeroplane', 'bicycle', 'bird', ...
    'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', ...
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', ...
    'sofa', 'train', 'tvmonitor', 'background' };
  
net.meta.classes.description = {};
