function test_examples()
%TEST_EXAMPLES  Test some of the examples in the `examples/` directory

addpath examples/mnist ;
addpath examples/cifar ;

trainOpts.gpus = [] ;
trainOpts.continue = true ;
num = 1 ;

exps = {} ;
for networkType = {'dagnn', 'simplenn'}
  for index = 1:4
    clear ex ;
    ex.trainOpts = trainOpts ;
    ex.networkType = char(networkType) ;
    ex.index = index ;
    exps{end+1} = ex ;
  end
end

if num > 1
  if isempty(gcp('nocreate')),
    parpool('local',num) ;
  end
  parfor e = 1:numel(exps)
    test_one(exps{e}) ;
  end
else
  for e = 1:numel(exps)
    test_one(exps{e}) ;
  end
end

% ------------------------------------------------------------------------
function test_one(ex)
% -------------------------------------------------------------------------

suffix = ['-' ex.networkType] ;
switch ex.index
  case 1
    cnn_mnist(...
      'expDir', ['data/test-mnist' suffix], ...
      'batchNormalization', false, ...
      'networkType', ex.networkType, ...
      'train', ex.trainOpts) ;

  case 2
    cnn_mnist(...
      'expDir', ['data/test-mnist-bnorm' suffix], ...
      'batchNormalization', true, ...
      'networkType', ex.networkType, ...
      'train', ex.trainOpts) ;

  case 3
    cnn_cifar(...
      'expDir', ['data/test-cifar-lenet' suffix], ...
      'modelType', 'lenet', ...
      'networkType', ex.networkType, ...
      'train', ex.trainOpts) ;

  case 4
    cnn_cifar(...
      'expDir', ['data/test-cifar-nin' suffix], ...
      'modelType', 'nin', ...
      'networkType', ex.networkType, ...
      'train', ex.trainOpts) ;
end
