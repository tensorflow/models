classdef nnmnist < nntest
  properties (TestParameter)
    networkType = {'simplenn', 'dagnn'}
  end

  methods (TestClassSetup)
    function init(test)
      addpath(fullfile(vl_rootnn, 'examples', 'mnist'));
    end
  end

  methods (Test)
    function valErrorRate(test, networkType)
      clear mex ; % will reset GPU, remove MCN to avoid crashing
                  % MATLAB on exit (BLAS issues?)
      if strcmp(test.currentDataType, 'double'), return ; end
      rng(0);  % fix random seed, for reproducible tests
      switch test.currentDevice
        case 'cpu'
          gpus = [];
        case 'gpu'
          gpus = 1;
      end
      trainOpts = struct('numEpochs', 1, 'continue', false, 'gpus', gpus, ...
        'plotStatistics', false);
      if strcmp(networkType, 'simplenn')
        trainOpts.errorLabels = {'error', 'top5err'} ;
      end
      [~, info] = cnn_mnist('train', trainOpts, 'networkType', networkType);
      test.verifyLessThan(info.train.error, 0.08);
      test.verifyLessThan(info.val.error, 0.025);
    end
  end
end
