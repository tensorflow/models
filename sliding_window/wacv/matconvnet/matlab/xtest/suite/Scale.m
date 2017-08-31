classdef Scale < nntest
  properties
    x
    a
    b
  end

  properties (TestParameter)
    dim = {1 2 3 4}
  end

  methods (TestClassSetup)
    function data(test,device)
      test.x = test.randn(15,14,3,2) ;
      test.a = test.randn(15,14,3,2) ;
      test.b = test.randn(15,14,3,2) ;
    end
  end

  methods (Test)
    function data_and_parameters(test, dim)
      x = test.x ;
      a = test.a ;
      b = test.b ;

      a = sum(a, dim) ;
      b = sum(b, dim) ;

      scale = dagnn.Scale('size', size(a), 'hasBias', true) ;

      output = scale.forward({x}, {a,b}) ;
      dzdy = test.randn(size(output{1})) ;
      [derInputs, derParams] = scale.backward({x}, {a,b}, {dzdy}) ;

      pick = @(x) x{1} ;
      dzdx = derInputs{1} ;
      dzda = derParams{1} ;
      dzdb = derParams{2} ;

      test.der(@(x) pick(scale.forward({x},{a,b})), x, dzdy, dzdx, 1e-2 * test.range) ;
      test.der(@(a) pick(scale.forward({x},{a,b})), a, dzdy, dzda, 1e-2 * test.range) ;
      test.der(@(b) pick(scale.forward({x},{a,b})), b, dzdy, dzdb, 1e-2 * test.range) ;
    end

    function data_only(test, dim)
      x = test.x ;
      a = test.a ;
      b = test.b ;

      a = sum(a, dim) ;
      b = sum(b, dim) ;

      scale = dagnn.Scale('size', size(a), 'hasBias', true) ;

      output = scale.forward({x,a,b}, {}) ;
      dzdy = test.randn(size(output{1})) ;
      [derInputs, derParams] = scale.backward({x,a,b}, {}, {dzdy}) ;

      pick = @(x) x{1} ;
      dzdx = derInputs{1} ;
      dzda = derInputs{2} ;
      dzdb = derInputs{3} ;

      test.der(@(x) pick(scale.forward({x,a,b},{})), x, dzdy, dzdx, 1e-2 * test.range) ;
      test.der(@(a) pick(scale.forward({x,a,b},{})), a, dzdy, dzda, 1e-2 * test.range) ;
      test.der(@(b) pick(scale.forward({x,a,b},{})), b, dzdy, dzdb, 1e-2 * test.range) ;
    end
  end
end
