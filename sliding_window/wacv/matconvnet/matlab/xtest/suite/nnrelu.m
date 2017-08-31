classdef nnrelu < nntest
  properties
    x
  end

  methods (TestClassSetup)
    function data(test,device)
      % make sure that all elements in x are different. in this way,
      % we can compute numerical derivatives reliably by adding a delta < .5.
      x = test.randn(15,14,3,2) ;
      x(:) = randperm(numel(x))' ;
      % avoid non-diff value for test
      x(x==0)=1 ;
      test.x = x ;
      test.range = 10 ;
      if strcmp(device,'gpu'), test.x = gpuArray(test.x) ; end
    end
  end

  methods (Test)
    function basic(test)
      x = test.x ;
      y = vl_nnrelu(x) ;
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnrelu(x,dzdy) ;
      test.der(@(x) vl_nnrelu(x), x, dzdy, dzdx, 1e-2 * test.range) ;
    end
  end
end
