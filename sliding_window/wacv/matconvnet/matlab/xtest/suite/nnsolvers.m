classdef nnsolvers < nntest
  properties (TestParameter)
    networkType = {'simplenn', 'dagnn'}
    solver = {[], @solver.adagrad, @solver.adadelta, @solver.rmsprop, @solver.adam}
  end
  properties
    imdb
    init_w
    init_b
  end

  methods (TestClassSetup)
    function data(test, dataType)
      % synthetic data, 2 classes of gaussian samples with different means
      rng(0) ;
      sz = [15, 10, 5] ;  % input size
      x1 = 2 * randn([sz, 100], dataType) ;  % place mean at the origin
      x2 = bsxfun(@plus, 2 * randn(sz, dataType), 2 * randn([sz, 100], dataType)) ;  % place mean randomly
      
      test.imdb.x = cat(4, x1, x2) ;
      test.imdb.y = [ones(100, 1, dataType); 2 * ones(100, 1, dataType)] ;
      
      test.init_w = 1e-3 * randn([sz, 2], dataType) ;  % initial parameters
      test.init_b = zeros([2, 1], dataType) ;
    end
  end

  methods (Test)
    function basic(test, networkType, solver)
      clear mex ; % will reset GPU, remove MCN to avoid crashing
                  % MATLAB on exit (BLAS issues?)

      if strcmp(networkType, 'simplenn') && strcmp(test.currentDataType, 'double')
        return  % simplenn does not work well with doubles
      end

      % a simple logistic regression network
      net.layers = {struct('type','conv', 'weights',{{test.init_w, test.init_b}}), ...
                    struct('type','softmaxloss')} ;
      
      switch test.currentDevice
        case 'cpu', gpus = [];
        case 'gpu', gpus = 1;
      end

      switch networkType
        case 'simplenn',
          trainfn = @cnn_train ;
          getBatch = @(imdb, batch) deal(imdb.x(:,:,:,batch), imdb.y(batch)) ;
          
        case 'dagnn',
          trainfn = @cnn_train_dag ;
          
          if isempty(gpus)
            getBatch = @(imdb, batch) ...
                {'input',imdb.x(:,:,:,batch), 'label',imdb.y(batch)} ;
          else
            getBatch = @(imdb, batch) ...
                {'input',gpuArray(imdb.x(:,:,:,batch)), 'label',imdb.y(batch)} ;
          end
          
          net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
          net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
                      {'prediction','label'}, 'top1err') ;
      end

      % train 1 epoch with small batches and check convergence
      [~, info] = trainfn(net, test.imdb, getBatch, ...
        'train', 1:numel(test.imdb.y), 'val', 1, ...
        'solver', solver, 'batchSize', 10, 'numEpochs',1, ...
        'continue', false, 'gpus', gpus, 'plotStatistics', false) ;
      
      test.verifyLessThan(info.train.top1err, 0.35);
      test.verifyLessThan(info.train.objective, 0.5);
    end
  end
end
