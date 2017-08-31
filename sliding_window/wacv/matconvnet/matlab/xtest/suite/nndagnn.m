classdef nndagnn < nntest

  properties
    net;
    x;
    class;
    inputs;
    outputs;
  end

  methods (TestClassSetup)
    function initNet(test, device)
      test.net = [];
      test.net.layers = {} ;
      test.net.layers{end+1} = struct('type', 'conv', ...
        'weights', {{test.randn(5,5,1,20), test.zeros(1, 20)}}, ...
        'stride', 1, ...
        'pad', 0) ;
      test.net.layers{end+1} = struct('type', 'pool', ...
        'method', 'max', ...
        'pool', [2 2], ...
        'stride', 2, ...
        'pad', 0) ;
      test.net.layers{end+1} = struct('type', 'conv', ...
        'weights', {{test.randn(5,5,20,50),test.zeros(1,50)}}, ...
        'stride', 1, ...
        'pad', 0) ;
      test.net.layers{end+1} = struct('type', 'pool', ...
        'method', 'max', ...
        'pool', [2 2], ...
        'stride', 2, ...
        'pad', 0) ;
      test.net.layers{end+1} = struct('type', 'conv', ...
        'weights', {{test.randn(4,4,50,500),test.zeros(1,500)}}, ...
        'stride', 1, ...
        'pad', 0) ;
      test.net.layers{end+1} = struct('type', 'relu') ;
      test.net.layers{end+1} = struct('type', 'conv', ...
        'weights', {{test.randn(1,1,500,10),test.zeros(1,10)}}, ...
        'stride', 1, ...
        'pad', 0) ;
      test.net.layers{end+1} = struct('type', 'softmaxloss') ;
      % Fill the missing values
      test.net = dagnn.DagNN.fromSimpleNN(vl_simplenn_tidy(test.net));
      test.inputs = test.net.getInputs();
      test.outputs = test.net.getOutputs();

      test.x = test.randn(32, 32, 1, 20) ;
      test.class = test.toDevice(randi(10, 20, 1));
      test.net.move(device) ;
    end
  end

  methods (Test)
    function simpleRun(test)
      % Verify the forget functionality for the forward pass
      test.net.conserveMemory = true;
      test.forward();
      for ri = 1:numel(test.net.vars)
        if ismember(test.net.vars(ri).name, test.outputs)
          test.verifyNotEmpty(test.net.vars(ri).value);
        else
          test.verifyEmpty(test.net.vars(ri).value);
        end
      end
      % Verify the forget functionality for the backward pass
      test.backward();
      for ri = 1:numel(test.net.vars)
        if ismember(test.net.vars(ri).name, test.inputs)
          test.verifyNotEmpty(test.net.vars(ri).value);
        else
          test.verifyEmpty(test.net.vars(ri).value);
        end
      end
      for pi = 1:numel(test.net.params)
        test.verifyNotEmpty(test.net.params(pi).der);
      end
    end

    function conserveMemory(test)
      % Verify the forget functionality for the forward pass
      test.net.conserveMemory = false;
      test.forward();
      for ri = 1:numel(test.net.vars)
          test.verifyNotEmpty(test.net.vars(ri).value);
      end
      % Verify the forget functionality for the backward pass
      test.backward();
      for ri = 1:numel(test.net.vars)
        test.verifyNotEmpty(test.net.vars(ri).value);
        if ~ismember(test.net.vars(ri).name, test.inputs)
          test.verifyNotEmpty(test.net.vars(ri).der);
        end
      end
    end

    function precious(test)
      % Verify that the precious argument works
      selLayer = 3;
      outputIdx = test.net.layers(selLayer).outputIndexes;
      test.net.vars(outputIdx).precious = true;
      test.net.conserveMemory = true;
      test.forward();
      for ri = 1:numel(test.net.vars)
        if ismember(test.net.vars(ri).name, test.outputs) || ri == outputIdx
          test.verifyNotEmpty(test.net.vars(ri).value);
        else
          test.verifyEmpty(test.net.vars(ri).value);
        end
      end
      test.net.vars(outputIdx).precious = false;
    end

    function getReceptiveFields(test)
      % Just test if it does not crash
      for vi = 1:numel(test.net.vars)
        test.net.getVarReceptiveFields(test.net.vars(vi).name);
      end
    end
  end

  methods
    function forward(test)
      test.net.reset();
      test.net.eval({'x0', test.x, 'label', test.class});
    end
    function backward(test)
      test.net.reset();
      test.net.eval({'x0', test.x, 'label', test.class}, {'x8', 1});
    end
  end
end
