classdef nnroipool < nntest
  properties
    x
  end

  properties (TestParameter)
    method = {'avg', 'max'}
    subdivisions = {[1 1], [2 1], [1 2], [3 7], [16 16]}
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
    function basic(test,method,subdivisions)
      R = [1  1 1 2   2 2 1 1 ;
           0  1 2 0   1 2 1 1 ;
           0  4 3 0   1 2 1 1 ;
           15 5 6 15  4 2 9 0 ;
           14 7 9 14  4 8 1 0] ;
      R = test.toDevice(test.toDataType(R)) ;
      x = test.x ;
      args = {'method', method, 'subdivisions', subdivisions} ;
      y = vl_nnroipool(x,R,args{:}) ;
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnroipool(x,R,dzdy,args{:}) ;
      test.der(@(x) vl_nnroipool(x,R,args{:}), ...
               x, dzdy, dzdx, test.range * 1e-2) ;
    end

    function identity(test,method)
      x = test.toDevice(test.toDataType((2:10)'*(1:10))) ;
      R = test.toDevice(test.toDataType([1, 1, 1, 9, 10])) ;
      T = [0 1 0 ; 1 0 0] ;
      opts = {'method', method, ...
              'subdivisions', [9,10], ...
              'transform', T} ;
      y = vl_nnroipool(x,R,opts{:}) ;
      test.eq(x,y) ;
    end
  end
end
