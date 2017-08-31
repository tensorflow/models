classdef nnconcat < nntest
  methods (Test)
    function basic(test)
      pick = @(i,x) x{i} ;
      sz = [4,5,10,3] ;
      for dim = 1:3
        sz1 = sz ; sz1(dim) = 3 ;
        sz2 = sz ; sz2(dim) = 7 ;
        sz3 = sz ; sz3(dim) = 2 ;
        x1 = test.randn(sz1) ;
        x2 = test.randn(sz2) ;
        x3 = test.randn(sz3) ;

        y = vl_nnconcat({x1, x2, x3}, dim) ;
        test.verifyEqual(size(y,dim), size(x1,dim)+size(x2,dim)+size(x3,dim)) ;
        dzdy = test.randn(size(y)) ;
        dzdx = vl_nnconcat({x1, x2, x3} ,dim, dzdy) ;

        test.der(@(x1) vl_nnconcat({x1, x2, x3},dim), x1, dzdy, dzdx{1}, 1e-3*test.range) ;
        test.der(@(x2) vl_nnconcat({x1, x2, x3},dim), x2, dzdy, dzdx{2}, 1e-3*test.range) ;
        test.der(@(x3) vl_nnconcat({x1, x2, x3},dim), x3, dzdy, dzdx{3}, 1e-3*test.range) ;
      end
    end

    function by_size(test)
      pick = @(i,x) x{i} ;
      sz = [4,5,10,3] ;
      for dim = 1:3
        sz1 = sz ; sz1(dim) = 3 ;
        sz2 = sz ; sz2(dim) = 7 ;
        sz3 = sz ; sz3(dim) = 2 ;
        x1 = test.randn(sz1) ;
        x2 = test.randn(sz2) ;
        x3 = test.randn(sz3) ;

        y = vl_nnconcat({x1, x2, x3}, dim) ;
        test.verifyEqual(size(y,dim), size(x1,dim)+size(x2,dim)+size(x3,dim)) ;
        dzdy = test.randn(size(y)) ;
        dzdx = vl_nnconcat({}, dim, dzdy, 'inputSizes', {sz1, sz2, sz3}) ;

        test.der(@(x1) vl_nnconcat({x1, x2, x3},dim), x1, dzdy, dzdx{1}, 1e-3*test.range) ;
        test.der(@(x2) vl_nnconcat({x1, x2, x3},dim), x2, dzdy, dzdx{2}, 1e-3*test.range) ;
        test.der(@(x3) vl_nnconcat({x1, x2, x3},dim), x3, dzdy, dzdx{3}, 1e-3*test.range) ;
      end
    end
  end
end
