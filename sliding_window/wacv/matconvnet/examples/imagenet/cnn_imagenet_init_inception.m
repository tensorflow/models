function net = cnn_imagenet_init_inception(varargin)
% CNN_IMAGENET_INIT  Initialize a standard CNN for ImageNet

opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts = vl_argparse(opts, varargin) ;

net = dagnn.DagNN() ;

net.meta.inputSize = [299 299 3 1] ;
net.meta.normalization.imageSize = net.meta.inputSize(1:3) ;

stack = {} ;

  function dup()
    stack{end+1} = stack{end} ;
  end

  function swap()
    stack([end-1 end]) = stack([end end-1]) ;
  end

  function Conv(name, ksize, out, varargin)
    copts.stride = [1 1] ;
    copts.pad = (ksize-1)/2 ;
    copts = vl_argparse(copts, varargin) ;
    if isempty(stack)
      inputVar = 'input' ;
      in = 3 ;
    else
      prev = stack{end} ;
      stack(end) = [] ;
      i = net.getLayerIndex(prev) ;
      inputVar = net.layers(i).outputs{1} ;
      sizes = net.getVarSizes({'input', net.meta.inputSize}) ;
      j = net.getVarIndex(inputVar) ;
      in = sizes{j}(3) ;
    end
    if numel(ksize) == 1, ksize = [ksize ksize] ; end
    net.addLayer(name , ...
      dagnn.Conv('size', [ksize in out], ...
      'stride', copts.stride, ....
      'pad', copts.pad, ...
      'opts', {'cudnnworkspacelimit', opts.cudnnWorkspaceLimit}), ...
      inputVar, ...
      [name '_conv'], ...
      {[name '_f'], [name '_b']}) ;
    net.addLayer([name '_bn'], ...
      dagnn.BatchNorm('numChannels', out), ...
      [name '_conv'], ...
      [name '_bn'], ...
      {[name '_bn_w'], [name '_bn_b'], [name '_bn_m']}) ;
    net.addLayer([name '_relu'] , ...
      dagnn.ReLU(), ...
      [name '_bn'], ...
      name) ;
    stack{end+1} = [name '_relu'] ;
  end

  function Pool(name, ksize, varargin)
    copts.stride = [1 1] ;
    copts.pad = (ksize-1)/2 ;
    copts.method = 'max' ;
    copts = vl_argparse(copts, varargin) ;

    prev = stack{end} ;
    stack(end) = [] ;
    i = net.getLayerIndex(prev) ;
    inputVar = net.layers(i).outputs{1} ;

    if numel(ksize) == 1, ksize = [ksize ksize] ; end
    net.addLayer(name , ...
      dagnn.Pooling('poolSize', ksize, ...
      'method', copts.method, ...
      'stride', copts.stride, ....
      'pad', copts.pad), ...
      inputVar, ...
      [name '_pool']) ;
    stack{end+1} = name ;
  end

  function Concat(name, num)
    inputVars = {} ;
    for layer = stack(end-num+1:end)
      prev = char(layer) ;
      i = net.getLayerIndex(prev) ;
      inputVars{end+1} = net.layers(i).outputs{1} ;
    end
    stack(end-num+1:end) = [] ;
    net.addLayer(name , ...
      dagnn.Concat(), ...
      inputVars, ...
      name) ;
    stack{end+1} = name ;
  end

  function Pred(name, out, varargin)
    prev = stack{end} ;
    stack(end) = [] ;
    i = net.getLayerIndex(prev) ;
    inputVar = net.layers(i).outputs{1} ;
    sizes = net.getVarSizes({'input', net.meta.inputSize}) ;
    j = net.getVarIndex(inputVar) ;
    in = sizes{j}(3) ;

    net.addLayer([name '_dropout'] , ...
      dagnn.DropOut('rate', 0.2), ...
      inputVar, ...
      [name '_dropout']) ;

    net.addLayer(name, ...
      dagnn.Conv('size', [1 1 in out]), ...
      [name '_dropout'], ...
      name, ...
      {[name '_f'], [name '_b']}) ;

    net.addLayer([name '_loss'], ...
      dagnn.Loss('loss', 'softmaxlog'), ...
      {name, 'label'}, ...
      [name '_loss']) ;

    net.addLayer([name '_top1error'], ...
      dagnn.Loss('loss', 'classerror'), ...
      {name, 'label'}, ...
      [name '_top1error']) ;

    net.addLayer([name '_top5error'], ...
      dagnn.Loss('loss', 'topkerror', 'opts', {'topK', 5}), ...
      {name, 'label'}, ...
      [name '_top5error']) ;
  end

% Pre-inception
Conv('conv', 3, 32, 'stride', 2, 'pad', 0) ;
Conv('conv1', 3, 32, 'pad', 0) ;
Conv('conv2', 3, 64) ;
Pool('pool', 3, 'stride', 2, 'pad', 0) ;
Conv('conv3', 1, 80, 'pad', 0) ;
Conv('conv4', 3, 192, 'pad', 0) ;
Pool('pool1', 3, 'stride', 2, 'pad', 0) ;

% Inception fig. 5 x 3
for t = 1:3
  pfx = sprintf('inception5_%d', t) ;
  dup() ;
  Conv([pfx '_a1'], 1, 64) ;
  swap() ; dup() ;
  Conv([pfx '_b1'], 1, 48) ;
  Conv([pfx '_b2'], 5, 64) ;
  swap() ; dup() ;
  Conv([pfx '_c1'], 1, 64) ;
  Conv([pfx '_c2'], 3, 96) ;
  Conv([pfx '_c3'], 3, 96) ;
  swap() ;
  Pool([pfx '_d1'], 3, 'method', 'avg') ;
  Conv([pfx '_d2'], 1, 64) ;
  Concat(pfx, 4) ;
end

% Inception fig. 5 down
pfx = 'inception5_4' ;
dup() ;
Conv([pfx '_a1'], 3, 384, 'stride', 2, 'pad', 0) ;
swap() ; dup() ;
Conv([pfx '_b1'], 1, 64) ;
Conv([pfx '_b2'], 3, 96) ;
Conv([pfx '_b3'], 3, 96, 'stride', 2, 'pad', 0) ;
swap() ;
Pool([pfx '_c1'], 3, 'method', 'max', 'stride', 2, 'pad', 0) ;
Concat(pfx, 3) ;

% Inpcetion fig. 6 x 4
for t = 1:4
  pfx = sprintf('inception6_%d', t) ;
  dup() ;
  Conv([pfx '_a1'], 1, 192) ;
  swap() ; dup() ;
  Conv([pfx '_b1'], 1, 160) ;
  Conv([pfx '_b2'], [1 7], 160) ;
  Conv([pfx '_b3'], [7 1], 192) ;
  swap() ; dup() ;
  Conv([pfx '_c1'], 1, 160) ;
  Conv([pfx '_c2'], [7 1], 160) ;
  Conv([pfx '_c3'], [1 7], 160) ;
  Conv([pfx '_c4'], [7 1], 160) ;
  Conv([pfx '_c5'], [1 7], 192) ;
  swap() ;
  Pool([pfx '_d1'], 3, 'method', 'avg') ;
  Conv([pfx '_d2'], 1, 192) ;
  Concat(pfx, 4) ;
end

% Inception fig. 6 down
pfx = 'inception6_5' ;
dup() ;
Conv([pfx '_a1'], 1, 192) ;
Conv([pfx '_a2'], 3, 320, 'stride', 2, 'pad', 0) ;
swap() ; dup() ;
Conv([pfx '_b1'], 1, 192) ;
Conv([pfx '_b2'], [1 7], 192) ;
Conv([pfx '_b3'], [7 1], 192) ;
Conv([pfx '_b4'], 3, 192, 'stride', 2, 'pad', 0) ;
swap() ;
Pool([pfx '_c1'], 3, 'method', 'max', 'stride', 2, 'pad', 0) ;
Concat(pfx, 3) ;

% Inception fig. 7 x 2
for t = 1:2
  pfx = sprintf('inception7_%d',t) ;
  dup() ;
  Conv([pfx '_a1'], 1, 320) ;
  swap() ; dup() ;
  Conv([pfx '_b1'], 1, 384) ;
  Conv([pfx '_b2'], [1 3], 384) ;
  Conv([pfx '_b3'], [3 1], 384) ;
  swap() ; dup() ;
  Conv([pfx '_c1'], 1, 448) ;
  Conv([pfx '_c2'], 3, 384) ;
  Conv([pfx '_c3'], [1 3], 384) ;
  Conv([pfx '_c4'], [3 1], 384) ;
  swap() ;
  Pool([pfx '_d1'], 3, 'method', 'avg') ;
  Conv([pfx '_d2'], 1, 192) ;
  Concat(pfx, 4) ;
end

% Average pooling and loss
Pool('pool_2', 8, 'method', 'avg', 'pad', 0) ;
Pred('prediction', 1000) ;

% Meta parameters
net.meta.normalization.fullImageSize = 310 ;
net.meta.normalization.averageImage = [] ;
net.meta.augmentation.rgbSqrtCovariance = zeros(3,'single') ;
net.meta.augmentation.jitter = true ;
net.meta.augmentation.jitterLight = 0.1 ;
net.meta.augmentation.jitterBrightness = 0.4 ;
net.meta.augmentation.jitterSaturation = 0.4 ;
net.meta.augmentation.jitterContrast = 0.4 ;

net.meta.inputSize = {'input', [net.meta.normalization.imageSize 32]} ;

net.meta.trainOpts.learningRate = logspace(-1, -3, 60) ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.batchSize = 256 ;
net.meta.trainOpts.numSubBatches = 3 ;
net.meta.trainOpts.weightDecay = 0.0001 ;

% Init parameters randomly
net.initParams() ;
end
