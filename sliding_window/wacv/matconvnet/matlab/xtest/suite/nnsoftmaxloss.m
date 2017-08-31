classdef nnsoftmaxloss < nntest
  properties (TestParameter)
    weighed = {false true}
    multilab = {false true}
  end

  methods (Test)
    function basic(test, multilab, weighed)
      C = 10 ;
      n = 3 ;
      if multilab
        c = reshape(mod(0:3*4*n-1,C)+1, 3, 4, 1, n) ;
      else
        c = reshape([7 2 1],1,1,1,[]) ;
      end
      if weighed
        c = cat(3, c, test.rand(size(c))) ;
      end

      % compare direct and indirect composition; this cannot
      % take large test.ranges
      x = test.rand(3,4,C,n)/test.range + 0.001 ; % non-negative
      y = vl_nnsoftmaxloss(x,c) ;
      if size(c,3) == 1
        opts = {'loss','log'} ;
      else
        opts = {'loss','log','instanceWeights',c(:,:,2,:)} ;
      end
      y_ = vl_nnloss(vl_nnsoftmax(x),c(:,:,1,:),[],opts{:}) ;
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnsoftmaxloss(x,c,dzdy) ;
      dzdx_ = vl_nnsoftmax(x,vl_nnloss(vl_nnsoftmax(x),c(:,:,1,:),dzdy,opts{:})) ;
      test.eq(y,y_) ;
      test.eq(dzdx,dzdx_) ;
      test.der(@(x) vl_nnsoftmaxloss(x,c), x, dzdy, dzdx, 0.001, -5e1) ;

      % now larger input range
      x = test.rand(3,4,C,n) + test.range * 0.001 ; % non-negative
      y = vl_nnsoftmaxloss(x,c) ;
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnsoftmaxloss(x,c,dzdy) ;
      test.der(@(x) vl_nnsoftmaxloss(x,c), ...
               x, dzdy, dzdx, test.range * 0.001, -5e1) ;
    end
  end
end
