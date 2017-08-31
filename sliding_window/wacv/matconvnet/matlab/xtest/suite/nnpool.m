classdef nnpool < nntest
  properties
    x
  end

  properties (TestParameter)
    type = {'avg', 'max'}
    poolx = {1 2 3}
    pooly = {1 2}
    pool = {1 2 3}
    pad = {0 1 2}
    stride = {1 2 3 4}
    stridex = {1 2 3}
    stridey = {1 2}
    padLeft = {0 1 2}
    padRight = {0 1 2}
    padTop = {0 1 2}
    padBottom = {0 1 2}
  end

  methods (TestClassSetup)
    function data(test,device)
      % make sure that all elements in x are different. in this way,
      % we can compute numerical derivatives reliably by adding a delta < .5.
      x = test.randn(15,14,3,2) ;
      x(:) = randperm(numel(x))' ;
      test.x = x ;
      test.range = 10 ;
      if strcmp(device,'gpu'), test.x = gpuArray(test.x) ; end
    end
  end

  methods (Test)
    function basic(test,poolx,pooly)
      % Test whether the avg pool output is equal to its emulation with
      % convolutional layer
      x = test.x ;
      stride = 1 ;
      pad = 0 ;
      pool = [pooly poolx] ;
      args = {'stride',stride,'pad',pad, 'method', 'avg'};
      y = vl_nnpool(x,pool,args{:}) ;
      y_conv = vl_nnconv(gather(x), ...
                         ones(pooly,poolx,1,size(x,3),test.currentDataType)./poolx./pooly, ...
                         zeros(1,size(x,3),test.currentDataType), ...
                         'stride', stride, ...
                         'pad', pad);
      test.eq(y, y_conv, 1e-3); % Does not pass with 1e-4
    end

    function pool_type(test, type, pool, pad, stride)
      x = test.x ;
      if pad > pool-1, return ; end
      args = {'stride',stride,'pad',pad,'method',type};
      y = vl_nnpool(x,pool,args{:}) ;
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnpool(x,pool,dzdy,args{:}) ;
      test.der(@(x) vl_nnpool(x,pool,args{:}), ...
               x, dzdy, dzdx, test.range * 1e-2) ;
    end

    function pool_type_and_pad(test, type, poolx, pooly, stride)
      x = test.x ;
      stride = 1 ;
      pad = 0 ;
      pool = [pooly poolx] ;
      args = {'stride',stride,'pad',pad,'method',type};
      y = vl_nnpool(x,pool,args{:}) ;
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnpool(x,pool,dzdy,args{:}) ;
      test.der(@(x) vl_nnpool(x,pool,args{:}), ...
               x, dzdy, dzdx, test.range * 1e-2) ;
    end

    function pool_type_and_stride(test, type, stridex, stridey)
      x = test.x ;
      pad = 0 ;
      pool = [3 2] ;
      stride = [stridey stridex] ;
      args = {'stride',stride,'pad',pad,'method',type};
      y = vl_nnpool(x,pool,args{:}) ;
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnpool(x,pool,dzdy,args{:}) ;
      test.der(@(x) vl_nnpool(x,pool,args{:}), ...
               x, dzdy, dzdx, test.range * 1e-2) ;
    end

    function asym_pad1(test, type, padLeft, padRight)
      x = test.x ;
      pool = [3 4] ;
      stride = [2 1] ;
      pad = [0 0 padLeft padRight] ;
      args = {'stride',stride,'pad',pad,'method',type};
      y = vl_nnpool(x,pool,args{:}) ;
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnpool(x,pool,dzdy,args{:}) ;
      test.der(@(x) vl_nnpool(x,pool,args{:}), ...
               x, dzdy, dzdx, test.range * 1e-2) ;
    end

    function asym_pad2(test, type, padTop, padBottom)
      x = test.x ;
      pool = [3 4] ;
      stride = [2 1] ;
      pad = [padTop padBottom 2 1] ;
      args = {'stride',stride,'pad',pad,'method',type};
      y = vl_nnpool(x,pool,args{:}) ;
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnpool(x,pool,dzdy,args{:}) ;
      test.der(@(x) vl_nnpool(x,pool,args{:}), ...
               x, dzdy, dzdx, test.range * 1e-2) ;
    end
  end
end
