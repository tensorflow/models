function cnn_imagenet_googlenet()
%CNN_IMAGENET_GOOGLENET  Demonstrates how to use GoogLeNet

run matlab/vl_setupnn
modelPath = 'data/models/imagenet-googlenet-dag.mat' ;

if ~exist(modelPath)
  mkdir(fileparts(modelPath)) ;
  urlwrite(...
  'http://www.vlfeat.org/matconvnet/models/imagenet-googlenet-dag.mat', ...
    modelPath) ;
end

net = dagnn.DagNN.loadobj(load(modelPath)) ;

im = imread('peppers.png') ;
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = im_ - net.meta.normalization.averageImage ;
net.eval({'data', im_}) ;

% show the classification result
scores = squeeze(gather(net.vars(end).value)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;
