classdef nnbnorm < nntest
  properties (TestParameter)
    rows = {2 8 13}
    cols = {2 8 17}
    numDims = {1 3 4}
    batchSize = {2 7}
  end
  methods (Test)
    function basic(test, rows, cols, numDims, batchSize)
      r = rows ;
      c = cols ;
      nd = numDims ;
      bs = batchSize ;
      x = test.randn(r, c, nd, bs) ;
      %g = test.randn(1, 1, nd, 1) ;
      %b = test.randn(1, 1, nd, 1) ;
      g = test.randn(nd, 1) / test.range ;
      b = test.randn(nd, 1) / test.range ;

      y = vl_nnbnorm(x,g,b) ;
      dzdy = test.randn(size(y)) ;
      [dzdx,dzdg,dzdb] = vl_nnbnorm(x,g,b,dzdy) ;

      test.der(@(x) vl_nnbnorm(x,g,b), x, dzdy, dzdx, test.range * 1e-3) ;
      test.der(@(g) vl_nnbnorm(x,g,b), g, dzdy, dzdg, 1e-2) ;
      test.der(@(b) vl_nnbnorm(x,g,b), b, dzdy, dzdb, 1e-3) ;
    end
  end
end