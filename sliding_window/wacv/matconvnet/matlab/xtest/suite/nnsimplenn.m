classdef nnsimplenn < nntest

  properties
    net;
    x;
    class;
  end

  methods (TestClassSetup)
    function initNet(test, device)
      test.net.layers = {} ;
      test.net.layers{end+1} = struct('type', 'conv', ...
        'weights', {{test.randn(5,5,1,20)/test.range, test.zeros(1, 20)}}, ...
        'stride', 1, ...
        'pad', 0) ;
      test.net.layers{end+1} = struct('type', 'pool', ...
        'method', 'max', ...
        'pool', [2 2], ...
        'stride', 2, ...
        'pad', 0) ;
      test.net.layers{end+1} = struct('type', 'conv', ...
        'weights', {{test.randn(5,5,20,50)/test.range, test.zeros(1,50)}}, ...
        'stride', 1, ...
        'pad', 0) ;
      test.net.layers{end+1} = struct('type', 'pool', ...
        'method', 'max', ...
        'pool', [2 2], ...
        'stride', 2, ...
        'pad', 0) ;
      test.net.layers{end+1} = struct('type', 'conv', ...
        'weights', {{test.randn(4,4,50,500)/test.range, test.zeros(1,500)}}, ...
        'stride', 1, ...
        'pad', 0) ;
      test.net.layers{end+1} = struct('type', 'relu') ;
      test.net.layers{end+1} = struct('type', 'conv', ...
        'weights', {{test.randn(1,1,500,10)/test.range, test.zeros(1,10)}}, ...
        'stride', 1, ...
        'pad', 0) ;
      test.net.layers{end+1} = struct('type', 'softmaxloss') ;
      % Fill the missing values
      test.net = vl_simplenn_tidy(test.net);

      test.x = test.randn(32, 32, 1, 20)/test.range ;
      test.class = test.toDevice(randi(10, 20, 1)) ;
      test.net = vl_simplenn_move(test.net, device);
    end
  end

  methods (Test)
    function simpleRun(test)
      % Verify the forget functionality for the forward pass
      test.net.layers{end}.class = test.class;
      res = vl_simplenn(test.net, test.x, [], [], 'conserveMemory', true);
      for ri = 1:numel(res)
        if ri == numel(res)
          test.verifyNotEmpty(res(ri).x);
        else
          test.verifyEmpty(res(ri).x);
        end
      end
      % Verify the forget functionality for the backward pass
      res = vl_simplenn(test.net, test.x, 1, [], 'conserveMemory', true);
      for ri = 2:numel(res) - 1
        test.verifyEmpty(res(ri).x);
        test.verifyEmpty(res(ri).dzdx);
      end
      % The values are kept in the first and last results
      for ri = 1:numel(res) - 1
        if strcmp(test.net.layers{ri}.type, 'conv')
          test.verifyNotEmpty(res(ri).dzdw);
        end
      end
    end

    function conserveMemory(test)
      % Verify conserve memory argument
      test.net.layers{end}.class = test.class;
      res = vl_simplenn(test.net, test.x, [], [], 'conserveMemory', false);
      for ri = 1:numel(res)
        test.verifyNotEmpty(res(ri).x);
      end
      res = vl_simplenn(test.net, test.x, 1, [], 'conserveMemory', false);
      for ri = 1:numel(res)
        test.verifyNotEmpty(res(ri).x);
        test.verifyNotEmpty(res(ri).dzdx);
      end
    end

    function precious(test)
      % Verify that the precious argument works
      selLayer = 3;
      net_ = test.net;
      net_.layers{end}.class = test.class;
      net_.layers{selLayer}.precious = true;
      res = vl_simplenn(net_, test.x, [], [], 'conserveMemory', true);
      for ri = 1:numel(res)
        if ri - 1 == selLayer || ri == numel(res)
          test.verifyNotEmpty(res(ri).x);
        else
          test.verifyEmpty(res(ri).x);
        end
      end

      res = vl_simplenn(net_, test.x, 1, [], 'conserveMemory', true);
      for ri = 2:numel(res) - 1
        if ri - 1 == selLayer
          test.verifyNotEmpty(res(ri).x);
          test.verifyNotEmpty(res(ri).dzdx);
        else
          test.verifyEmpty(res(ri).x);
          test.verifyEmpty(res(ri).dzdx);
        end
      end
    end

    function backPropDepth(test)
      backPropDepth = 3;
      net_ = test.net;
      net_.layers{end}.class = test.class;
      params = {'conserveMemory', true, 'backPropDepth', backPropDepth};
      res = vl_simplenn(net_, test.x, [], [], params{:});
      % Should be ignored for forward pass
      for ri = 1:numel(res) - 1
        test.verifyEmpty(res(ri).x);
      end
      res = vl_simplenn(net_, test.x, 1, [], params{:});
      for ri = 2:numel(res) - 1
        test.verifyEmpty(res(ri).x);
        test.verifyEmpty(res(ri).dzdx);
      end
      for ri = 1:numel(res) - 1
        if strcmp(test.net.layers{ri}.type, 'conv')
          if ri >= numel(res) - backPropDepth + 1
            test.verifyNotEmpty(res(ri).dzdw);
          else
            test.verifyEmpty(res(ri).dzdw);
          end
        end
      end
    end

    function skipForward(test)
      % Verify that the skipForward argument works
      test.net.layers{end}.class = test.class;
      res = vl_simplenn(test.net, test.x, 1);
      res_ = vl_simplenn(test.net, test.x, 1, res, 'skipForward', true);
      for li = 1:numel(res)
        % Might still slightly different on GPUs due to
        % non-deterministic operation order
        test.eq(res(li).x, res_(li).x);
        test.eq(res(li).dzdx, res_(li).dzdx);
      end
      % Verify that it throws the specified errors for 'skipForward'
      test.verifyError(...
        @() vl_simplenn(test.net, test.x, [], [], 'skipForward', true), ...
        'simplenn:skipForwardNoBackwPass');
      test.verifyError(...
        @() vl_simplenn(test.net, test.x, 1, [], 'skipForward', true), ...
        'simplenn:skipForwardEmptyRes');
    end
  end
end
