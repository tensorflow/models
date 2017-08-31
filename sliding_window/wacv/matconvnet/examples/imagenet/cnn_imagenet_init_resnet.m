function net = cnn_imagenet_init_resnet(varargin)
%CNN_IMAGENET_INIT_RESNET  Initialize the ResNet-50 model for ImageNet classification

opts.classNames = {} ;
opts.classDescriptions = {} ;
opts.averageImage = zeros(3,1) ;
opts.colorDeviation = zeros(3) ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts = vl_argparse(opts, varargin) ;

net = dagnn.DagNN() ;

lastAdded.var = 'input' ;
lastAdded.depth = 3 ;

function Conv(name, ksize, depth, varargin)
% Helper function to add a Convolutional + BatchNorm + ReLU
% sequence to the network.
  args.relu = true ;
  args.downsample = false ;
  args.bias = false ;
  args = vl_argparse(args, varargin) ;
  if args.downsample, stride = 2 ; else stride = 1 ; end
  if args.bias, pars = {[name '_f'], [name '_b']} ; else pars = {[name '_f']} ; end
  net.addLayer([name  '_conv'], ...
               dagnn.Conv('size', [ksize ksize lastAdded.depth depth], ...
                          'stride', stride, ....
                          'pad', (ksize - 1) / 2, ...
                          'hasBias', args.bias, ...
                          'opts', {'cudnnworkspacelimit', opts.cudnnWorkspaceLimit}), ...
               lastAdded.var, ...
               [name '_conv'], ...
               pars) ;
  net.addLayer([name '_bn'], ...
               dagnn.BatchNorm('numChannels', depth, 'epsilon', 1e-5), ...
               [name '_conv'], ...
               [name '_bn'], ...
               {[name '_bn_w'], [name '_bn_b'], [name '_bn_m']}) ;
  lastAdded.depth = depth ;
  lastAdded.var = [name '_bn'] ;
  if args.relu
    net.addLayer([name '_relu'] , ...
                 dagnn.ReLU(), ...
                 lastAdded.var, ...
                 [name '_relu']) ;
    lastAdded.var = [name '_relu'] ;
  end
end

% -------------------------------------------------------------------------
% Add input section
% -------------------------------------------------------------------------

Conv('conv1', 7, 64, ...
     'relu', true, ...
     'bias', false, ...
     'downsample', true) ;

net.addLayer(...
  'conv1_pool' , ...
  dagnn.Pooling('poolSize', [3 3], ...
                'stride', 2, ...
                'pad', 1,  ...
                'method', 'max'), ...
  lastAdded.var, ...
  'conv1') ;
lastAdded.var = 'conv1' ;

% -------------------------------------------------------------------------
% Add intermediate sections
% -------------------------------------------------------------------------

for s = 2:5

  switch s
    case 2, sectionLen = 3 ;
    case 3, sectionLen = 4 ; % 8 ;
    case 4, sectionLen = 6 ; % 23 ; % 36 ;
    case 5, sectionLen = 3 ;
  end

  % -----------------------------------------------------------------------
  % Add intermediate segments for each section
  for l = 1:sectionLen
    depth = 2^(s+4) ;
    sectionInput = lastAdded ;
    name = sprintf('conv%d_%d', s, l)  ;

    % Optional adapter layer
    if l == 1
      Conv([name '_adapt_conv'], 1, 2^(s+6), 'downsample', s >= 3, 'relu', false) ;
    end
    sumInput = lastAdded ;

    % ABC: 1x1, 3x3, 1x1; downsample if first segment in section from
    % section 2 onwards.
    lastAdded = sectionInput ;
    %Conv([name 'a'], 1, 2^(s+4), 'downsample', (s >= 3) & l == 1) ;
    %Conv([name 'b'], 3, 2^(s+4)) ;
    Conv([name 'a'], 1, 2^(s+4)) ;
    Conv([name 'b'], 3, 2^(s+4), 'downsample', (s >= 3) & l == 1) ;
    Conv([name 'c'], 1, 2^(s+6), 'relu', false) ;

    % Sum layer
    net.addLayer([name '_sum'] , ...
                 dagnn.Sum(), ...
                 {sumInput.var, lastAdded.var}, ...
                 [name '_sum']) ;
    net.addLayer([name '_relu'] , ...
                 dagnn.ReLU(), ...
                 [name '_sum'], ...
                 name) ;
    lastAdded.var = name ;
  end
end

net.addLayer('prediction_avg' , ...
             dagnn.Pooling('poolSize', [7 7], 'method', 'avg'), ...
             lastAdded.var, ...
             'prediction_avg') ;

net.addLayer('prediction' , ...
             dagnn.Conv('size', [1 1 2048 1000]), ...
             'prediction_avg', ...
             'prediction', ...
             {'prediction_f', 'prediction_b'}) ;

net.addLayer('loss', ...
             dagnn.Loss('loss', 'softmaxlog') ,...
             {'prediction', 'label'}, ...
             'objective') ;

net.addLayer('top1error', ...
             dagnn.Loss('loss', 'classerror'), ...
             {'prediction', 'label'}, ...
             'top1error') ;

net.addLayer('top5error', ...
             dagnn.Loss('loss', 'topkerror', 'opts', {'topK', 5}), ...
             {'prediction', 'label'}, ...
             'top5error') ;

% -------------------------------------------------------------------------
%                                                           Meta parameters
% -------------------------------------------------------------------------

net.meta.normalization.imageSize = [224 224 3] ;
net.meta.inputSize = [net.meta.normalization.imageSize, 32] ;
net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 256 ;
net.meta.normalization.averageImage = opts.averageImage ;

net.meta.classes.name = opts.classNames ;
net.meta.classes.description = opts.classDescriptions ;

net.meta.augmentation.jitterLocation = true ;
net.meta.augmentation.jitterFlip = true ;
net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;
net.meta.augmentation.jitterAspect = [3/4, 4/3] ;
net.meta.augmentation.jitterScale  = [0.4, 1.1] ;
%net.meta.augmentation.jitterSaturation = 0.4 ;
%net.meta.augmentation.jitterContrast = 0.4 ;

net.meta.inputSize = {'input', [net.meta.normalization.imageSize 32]} ;

%lr = logspace(-1, -3, 60) ;
lr = [0.1 * ones(1,30), 0.01*ones(1,30), 0.001*ones(1,30)] ;
net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.momentum = 0.9 ;
net.meta.trainOpts.batchSize = 256 ;
net.meta.trainOpts.numSubBatches = 4 ;
net.meta.trainOpts.weightDecay = 0.0001 ;

% Init parameters randomly
net.initParams() ;

% For uniformity with the other ImageNet networks, t
% the input data is *not* normalized to have unit standard deviation,
% whereas this is enforced by batch normalization deeper down.
% The ImageNet standard deviation (for each of R, G, and B) is about 60, so
% we adjust the weights and learing rate accordingly in the first layer.
%
% This simple change improves performance almost +1% top 1 error.
p = net.getParamIndex('conv1_f') ;
net.params(p).value = net.params(p).value / 100 ;
net.params(p).learningRate = net.params(p).learningRate / 100^2 ;

for l = 1:numel(net.layers)
  if isa(net.layers(l).block, 'dagnn.BatchNorm')
    k = net.getParamIndex(net.layers(l).params{3}) ;
    net.params(k).learningRate = 0.3 ;
  end
end

end
