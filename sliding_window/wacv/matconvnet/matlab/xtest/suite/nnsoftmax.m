classdef nnsoftmax < nntest
  properties (TestParameter)
    h = {1 2 3}
    w = {1 2}
  end
  methods (Test)
    function basic(test,h,w)
      d = 10 ;
      n = 3 ;
      x = test.randn(h,w,d,n)/test.range ;
      y = vl_nnsoftmax(x) ;
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnsoftmax(x, dzdy) ;
      test.der(@(x) vl_nnsoftmax(x), x, dzdy, dzdx, 1e-2) ;
    end
  end
end
