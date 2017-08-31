classdef nnpdist < nntest
  properties (TestParameter)
    oneToOne = {false, true}
    noRoot = {false, true}
    p = {.5 1 2 3}
    aggregate = {false, true}
  end
  methods (Test)
    function basic(test,oneToOne, noRoot, p, aggregate)
      if aggregate
        % make it smaller to avoid numerical derivative issues with
        % float
        h = 2 ;
        w = 2 ;
      else
        h = 13 ;
        w = 17 ;
      end
      d = 4 ;
      n = 5 ;
      x = test.randn(h,w,d,n) ;
      if oneToOne
        x0 = test.randn(h,w,d,n) ;
      else
        x0 = test.randn(1,1,d,n) ;
      end
      opts = {'noRoot', noRoot, 'aggregate', aggregate} ;

      y = vl_nnpdist(x, x0, p, opts{:}) ;

      % make sure they are not too close in any dimension as this may be a
      % problem for the finite difference dereivatives as one could
      % approach 0 which is not differentiable for some p-norms

      s = abs(bsxfun(@minus, x, x0)) < test.range*1e-1 ;
      x(s) = x(s) + 5*test.range ;

      dzdy = test.rand(size(y)) ;
      [dzdx, dzdx0] = vl_nnpdist(x,x0,p,dzdy,opts{:}) ;
      test.der(@(x) vl_nnpdist(x,x0,p,opts{:}), x, dzdy, dzdx, test.range * 1e-3) ;
      if oneToOne
        % Pdist does not implement backprop of the bsxfun
        test.der(@(x0) vl_nnpdist(x,x0,p,opts{:}), x0, dzdy, dzdx0, test.range * 1e-3) ;
      end
    end
  end
end
