function net = xnet_init(param)

a  = param.a;
b  = param.b;
c  = param.c;
d  = param.d;
p1 = param.p1;
p2 = param.p2;
p3 = param.p3;
p4 = param.p4;



import dagnn.*

net = DagNN();

%Meta parameters
bs = 256;
net.meta.normalization.imageSize = [32, 32, 1] ;
net.meta.inputSize = net.meta.normalization.imageSize ;
net.meta.normalization.border = 32 - net.meta.normalization.imageSize(1:2) ;
net.meta.normalization.interpolation = 'bicubic' ;
net.meta.normalization.averageImage = [] ;
net.meta.normalization.keepAspect = true ;
net.meta.augmentation.rgbVariance = zeros(0,1) ;
net.meta.augmentation.transformation = 'stretch' ;

% if ~opts.batchNormalization
lr = logspace(-2, -4, 60) ;
% else
%   lr = logspace(-1, -4, 20) ;
% end

net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.batchSize = bs ;
net.meta.trainOpts.weightDecay = 0.0005 ;

sc=1/100 ;

net.addLayer('conv1', ...
  dagnn.Conv('size', [p1 p1 1 a], 'pad', 0, 'stride', 1), ...
  'input', 'x1', {'conv1f','conv1b'});

f = net.getParamIndex('conv1f');
net.params(f).value = sc*randn(p1, p1, 1, a, 'single');
net.params(f).learningRate = 1;
net.params(f).weightDecay = 1;

f = net.getParamIndex('conv1b');
net.params(f).value = zeros(a, 1, 'single');
net.params(f).learningRate = 2;
net.params(f).weightDecay = 0;


net.addLayer('pool1',...
  dagnn.Pooling('method', 'max', 'poolSize', [2 2], 'pad', 0, 'stride', 2),...
  'x1','x2');


net.addLayer('conv2', ...
  dagnn.Conv('size', [p2 p2 a b], 'pad', 0, 'stride', 1), ...
  'x2', 'x3', {'conv2f','conv2b'});

f = net.getParamIndex('conv2f');
net.params(f).value = sc*randn(p2, p2, a, b, 'single');
net.params(f).learningRate = 1;
net.params(f).weightDecay = 1;

f = net.getParamIndex('conv2b');
net.params(f).value = zeros(b, 1, 'single');
net.params(f).learningRate = 2;
net.params(f).weightDecay = 0;


net.addLayer('pool2',...
  dagnn.Pooling('method', 'max', 'poolSize', [2 2], 'pad', 0, 'stride', 2),...
  'x3','x4');


net.addLayer('conv3', ...
  dagnn.Conv('size', [p3 p3 b c], 'pad', 0, 'stride', 1), ...
  'x4', 'x5', {'conv3f','conv3b'});

f = net.getParamIndex('conv3f');
net.params(f).value = sc*randn(p3, p3, b, c, 'single');
net.params(f).learningRate = 1;
net.params(f).weightDecay = 1;

f = net.getParamIndex('conv3b');
net.params(f).value = zeros(c, 1, 'single');
net.params(f).learningRate = 2;
net.params(f).weightDecay = 0;


net.addLayer('relu3',...
  dagnn.ReLU(),...
  'x5','x6');


net.addLayer('dropout1',...
  dagnn.DropOut(),...
  'x6','x7');


net.addLayer('conv4', ...
  dagnn.Conv('size', [p4 p4 c d], 'pad', 0, 'stride', 1), ...
  'x7', 'x8', {'conv4f','conv4b'});

f = net.getParamIndex('conv4f');
net.params(f).value = sc*randn(p4, p4, c, d, 'single');
net.params(f).learningRate = 1;
net.params(f).weightDecay = 1;

f = net.getParamIndex('conv4b');
net.params(f).value = zeros(d, 1, 'single');
net.params(f).learningRate = 2;
net.params(f).weightDecay = 0;


net.addLayer('conv5', ...
  dagnn.Conv('size', [1 1 d 2], 'pad', 0, 'stride', 1), ...
  'x8', 'prediction', {'conv5f','conv5b'});

f = net.getParamIndex('conv5f');
net.params(f).value = sc*randn(1, 1, d, 2, 'single');
net.params(f).learningRate = 1;
net.params(f).weightDecay = 1;

f = net.getParamIndex('conv5b');
net.params(f).value = zeros(2, 1, 'single');
net.params(f).learningRate = 2;
net.params(f).weightDecay = 0;


net.addLayer('loss', ...
  dagnn.Loss('loss', 'softmaxlog'), ...
  {'prediction','label'}, 'objective');


net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
  {'prediction','label'}, 'top1err') ;