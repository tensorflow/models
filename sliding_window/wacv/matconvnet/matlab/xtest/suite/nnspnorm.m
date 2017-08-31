classdef nnspnorm < nntest
  methods (Test)
   function basic(test)
      h = 13 ;
      w = 17 ;
      d = 4 ;
      n = 5 ;
      param = [3, 3, 0.1, 0.75] ;
      x = test.randn(h,w,d,n) ;
      y = vl_nnspnorm(x, param) ;
      dzdy = test.rand(h, w, d, n) ;
      dzdx = vl_nnspnorm(x, param, dzdy) ;
      test.der(@(x) vl_nnspnorm(x,param), x, dzdy, dzdx, test.range * 1e-3) ;
    end
  end
end
