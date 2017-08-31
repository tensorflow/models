classdef nnoffset < nntest
  methods (Test)
    function basic(test)
      param = [.34, .5] ;
      x = test.randn(4,5,10,3) ;
      y = vl_nnnoffset(x,param) ;
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnnoffset(x,param,dzdy) ;
      test.der(@(x) vl_nnnoffset(x,param), x, dzdy, dzdx, 1e-3*test.range) ;
    end
  end
end