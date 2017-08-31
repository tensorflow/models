classdef nnsigmoid < nntest
  methods (Test)
   function basic(test)
      x = test.randn(5,5,1,1)/test.range ;
      y = vl_nnsigmoid(x) ;
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnsigmoid(x,dzdy) ;
      test.der(@(x) vl_nnsigmoid(x), x, dzdy, dzdx, 1e-3) ;
    end
  end
end
