classdef nndropout < nntest
  methods (Test)
    function basic(test)
      x = test.randn(4,5,10,3) ;
      [y,mask] = vl_nndropout(x) ;
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nndropout(x,dzdy,'mask',mask) ;
      test.der(@(x) vl_nndropout(x,'mask',mask), x, dzdy, dzdx, 1e-3*test.range) ;
    end
  end
end

